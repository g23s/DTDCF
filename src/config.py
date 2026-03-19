import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert (word_emb_path is not None)

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosi)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_tse', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--head_num', type=int, default=8,
                        help='number of class')
    parser.add_argument('--d_prjh', type=int, default=64,
                        help='hidden size in projection network,32 for mosi 128 for mosei')
    parser.add_argument('--dropout_r', type=float, default=0.4)
    parser.add_argument('--multi_head', type=int, default=8)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 1.0)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_et', type=float, default=2e-3, )
    parser.add_argument('--lr_te', type=float, default=3e-3, )

    # ===== 损失函数权重 =====
    parser.add_argument('--lambda_cf', type=float, default=0,
                        help='weight for counterfactual loss')
    parser.add_argument('--lambda_single', type=float, default=1.0,
                        help='权重: 单模态预测损失 (single_loss)')
    parser.add_argument('--lambda_fusion', type=float, default=1.0,
                        help='权重: 融合预测损失 (fusion_loss)')
    parser.add_argument('--lambda_recon', type=float, default=0,
                        help='权重: 重构损失 (recon_loss)')
    parser.add_argument('--lambda_dis', type=float, default=0,
                        help='权重: 判别/对比损失 (dis_loss)')
    # 已有的 lambda_cf 保留，不要重复加
    parser.add_argument('--lambda_supcon', type=float, default=0,
                        help='weight for supervised contrastive loss')
    parser.add_argument('--lambda_mi', type=float, default=0,
                        help='weight for mutual information loss')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-5,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_et', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_te', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=5,
                        help='when to decay learning rate (default: 20)')

    # Logistics
    parser.add_argument('--tse_layers', type=int, default=1,
                        help='TSE layers')
    parser.add_argument('--ETlayers', type=int, default=2,
                        help='Encoder Tower layers')
    parser.add_argument('--step_ratio', type=int, default=30,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--a_size', type=int, default=150,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--v_size', type=int, default=150,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--dislen', type=int, default=90,
                        help='dislen (default: 30 for mosi 90 for mosei)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')

    # FlowAuto 参数（用于领域反事实注意力）
    parser.add_argument('--num_domains', type=int, default=3,
                        help='number of modality domains (text, audio, visual)')
    parser.add_argument('--domain_dim', type=int, default=8,
                        help='dimension of domain embedding for flow autoencoder')

    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    # # ===== 损失函数权重 =====
    # parser.add_argument('--lambda_single', type=float, default=1.0,
    #                     help='权重: 单模态预测损失 (single_loss)')
    # parser.add_argument('--lambda_fusion', type=float, default=1.0,
    #                     help='权重: 融合预测损失 (fusion_loss)')
    # parser.add_argument('--lambda_recon', type=float, default=1.0,
    #                     help='权重: 重构损失 (recon_loss)')
    # parser.add_argument('--lambda_dis', type=float, default=1.0,
    #                     help='权重: 判别/对比损失 (dis_loss)')
    # parser.add_argument('--lambda_cf', type=float, default=1.0,
    #                     help='权重: 反事实差异损失 (L_cf)')
    args = parser.parse_args()
    return args

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        """self.__dict__包括了self的内含属性"""
        return config_str


def get_config(dataset, mode, batch_size):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config