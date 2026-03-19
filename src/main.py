import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader



def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    #args.seed = 1000  # 想改成多少就写多少
    set_seed(args.seed)


    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when #weight decay的起点
    args.n_class = output_dim_dict.get(dataset, 1)#输出类别的个数
    args.criterion = criterion_dict.get(dataset, 'MSELoss')


    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)  # 实例化模型运行类
    solver.train_and_eval()
    # step_ratios = [50,5,10, 15, 20, 25]
    # step_ratios = [60, 75]
    # for ratio in step_ratios:
    #     # 复制基础参数并更新step_ratio
    #     print(f"当前ratio:{ratio}")
    #     current_args = argparse.Namespace(**vars(args))
    #     current_args.step_ratio = ratio
    #     solver = Solver(current_args, train_loader=train_loader, dev_loader=valid_loader,
    #                     test_loader=test_loader, is_train=True)  # 实例化模型运行类
    #     solver.train_and_eval()
    # -------------------------
    # 新增：提取特征并画图
    # -------------------------
    #
    # 训练全部结束后再可视化
    from extract_features_inside_main import plot_one_epoch

    plot_one_epoch(30)
