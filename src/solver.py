from torch import nn
import sys
import datetime
import time
from utils.eval_metrics import *
from utils.tools import *
from MSA import MSA
from transformers import get_linear_schedule_with_warmup
from src.utils.Sinkhorn import CustomMultiLossLayer
# from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn.functional as F

def irm_loss(fused, unimodal):
    return torch.mean((fused - unimodal)**2)

# ================================================================
# 因果敏感性驱动反事实损失函数 (sensitivity 模式)
# ================================================================
def causal_sensitivity_cf_loss(factual_attn, counter_attn, eps=1e-8):
    """
    因果敏感性驱动反事实损失（sensitivity 模式）
    factual_attn: [B, D] factual 分支输出
    counter_attn: [B, D] counterfactual 分支输出
    """
    with torch.no_grad():
        # (1) 计算注意力差距（反事实扰动强度）
        gap = torch.abs(factual_attn - counter_attn)
        # (2) 归一化得到重要性权重
        importance = gap / (gap.sum(dim=-1, keepdim=True) + eps)

    # (3) 计算加权的 -log(factual_attn)
    loss = -(importance * torch.log_softmax(factual_attn, dim=-1)).mean()
    return loss

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None,
                 pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # initialize the model
        if model is None:
            self.model = model = MSA(hp)  # 本文的模型在这里

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")
            # ✅ 在这里加（最正确位置） 计算 参数量（Parameter Count）
        #total_params = sum(p.numel() for p in model.parameters())
        #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f"\nTotal parameters: {total_params}")
        #print(f"Trainable parameters: {trainable_params}\n")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            main_param = []
            bert_param = []
            disnet_params = []
            encoder_tower_params = []
            TimeEncoder_params = []

            # 先收集所有需要的参数
            for name, p in model.named_parameters():  # 遍历输出参数
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'TDisNet' in name or 'ADisNet' in name or 'VDisNet' in name:
                        disnet_params.append(p)
                    elif 'ET_TA' in name or 'ET_TV' in name or 'ET_AV' in name:
                        encoder_tower_params.append(p)
                    elif 'a_encoder' in name or 'v_encoder' in name:
                        TimeEncoder_params.append(p)
                    else:
                        main_param.append(p)

            # 对main_param中的参数进行初始化
            for p in main_param:
                if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.kaiming_normal_(p, mode='fan_in')
            # for p in disnet_params:
            #     if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
            #         nn.init.kaiming_normal_(p, mode='fan_in')
            for p in encoder_tower_params:
                if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.kaiming_normal_(p, mode='fan_in')
            for p in TimeEncoder_params:
                if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.kaiming_normal_(p, mode='fan_in')
            # 创建优化器参数组

            self.optimizer_main_group = [
                {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
                {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
                {'params': encoder_tower_params, 'weight_decay': hp.weight_decay_et, 'lr': hp.lr_et},
                {'params': TimeEncoder_params, 'weight_decay': hp.weight_decay_te, 'lr': hp.lr_te},
            ]

            self.optimizer = getattr(torch.optim, self.hp.optim)(
                self.optimizer_main_group
            )
            num_training_steps = int(self.hp.n_train / self.hp.batch_size * self.hp.num_epochs)
            warmup_steps = int(self.hp.warmup_ratio * num_training_steps)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=hp.when, factor=0.5, verbose=True)
        # self.scheduler_main = ReduceLROnPlateau(self.optimizer, mode='min', patience=hp.when, factor=0.5, verbose=True)#这是一个学习率调度器，当模型的性能（如验证集上的损失）在一定数量的连续训练周期（epochs）内没有显著改善时，它会降低学习率。这种策略有助于在训练过程中避免陷入局部最小值，并且可以在训练后期通过减小学习率来细化模型参数。

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        print("Learning Rates for Each Parameter Group:")
        print(f"BERT learning rate: {self.hp.lr_bert}")
        print(f"Main learning rate: {self.hp.lr_main}")
        print(f"Encoder Tower learning rate: {self.hp.lr_et}")
        print()
        # writer = SummaryWriter('D:\Pycharm_project\MSA_run\src\experiment')
        scheduler = self.scheduler
        #自适应损失权重 (CustomMultiLossLayer)
        #custom_loss_layer = CustomMultiLossLayer(loss_num=4, device=self.device)
        def train(model, optimizer, scheduler):
            #epoch_loss = 0
            epoch_loss_sum = 0.0  # ✅ 初始化总损失累计器

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            # ✅ 定义累计器，初始为 0
            single_loss_sum, fusion_loss_sum, recon_loss_sum, dis_loss_sum = 0.0, 0.0, 0.0, 0.0
            proc_size = 0
            start_time = time.time()
            a_size = self.hp.a_size
            v_size = self.hp.v_size

            # s_loss, fusion_loss,proc_size,recon_loss, dis_loss,avg_loss1, avg_loss2,avg_loss3,avg_loss4 = 0, 0, 0, 0, 0, 0,0,0,0
            # start_time = time.time()
            # a_size = self.hp.a_size
            # v_size = self.hp.v_size

            for i_batch, batch_data in enumerate(self.train_loader):
                visual, vlens, audio, alens, r_labels,c_labels,l, bert_sent, bert_sent_mask, ids = batch_data
                model.zero_grad()
                with torch.cuda.device(0):
                    visual, audio, r_labels,c_labels, l, bert_sent, bert_sent_mask = \
                        visual.cuda(), audio.cuda(), r_labels.cuda(), c_labels.cuda(),l.cuda(), bert_sent.cuda(), \
                            bert_sent_mask.cuda()
                batch_size = r_labels.size(0)
                # 前向返回 recon_loss 和 dis_loss 是张量，不要和累计器混淆
                r_preds, r_preds_F, recon_loss, dis_loss, factual_text, counter_text = model(
                    visual, audio, v_size, a_size, bert_sent, bert_sent_mask
                )
                #r_preds, r_preds_F, recon_loss,dis_loss, factual_text, counter_text = model(visual, audio, v_size, a_size, bert_sent, bert_sent_mask)
                # KL 散度损失：L_cf = KL(factual || counterfactual)
                # log_p = torch.log_softmax(factual_text, dim=-1)  # factual 分支
                # q = torch.softmax(counter_text.detach(), dim=-1)  # counterfactual 分支
                # #L_cf = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
                # L_cf = F.kl_div(log_p, q, reduction="batchmean")

#下面这段改cf损失逻辑
                # # ===================== 因果对比损失 ===================== #
                # # factual 分支（anchor）
                # log_p = torch.log_softmax(factual_text, dim=-1)
                # p = torch.softmax(factual_text, dim=-1)
                # # counterfactual 分支（负样本）
                # log_q = torch.log_softmax(counter_text, dim=-1)
                # q = torch.softmax(counter_text, dim=-1)
                #
                # # 1. 对称 KL 散度 (保证双向约束)
                # kl_fq = F.kl_div(log_p, q, reduction="batchmean")  # KL(P||Q)
                # kl_qf = F.kl_div(log_q, p, reduction="batchmean")  # KL(Q||P)
                # L_cf_kl = (kl_fq + kl_qf) / 2.0
                #
                # # 2. InfoNCE / SupCon 损失 (factual = anchor, counterfactual = 负样本)
                # temperature = 0.07
                # f_norm = F.normalize(factual_text, dim=-1)
                # cf_norm = F.normalize(counter_text, dim=-1)
                #
                # # 正样本：factual 与同 batch 的标签一致样本
                # # 负样本：counterfactual + 其他类 factual
                # labels = r_labels.view(-1, 1)  # batch 标签
                # mask = torch.eq(labels, labels.T).float().to(self.device)  # 正样本掩码
                #
                # sim_matrix = torch.matmul(f_norm, f_norm.T) / temperature  # factual-factual 相似度
                # sim_matrix_cf = torch.matmul(f_norm, cf_norm.T) / temperature  # factual-counter 相似度
                #
                # # InfoNCE: anchor factual，对比所有 factual+cf
                # logits = torch.cat([sim_matrix, sim_matrix_cf], dim=1)  # 拼接正负样本
                # supcon_loss = -torch.log(
                #     torch.sum(torch.exp(sim_matrix) * mask, dim=1) /
                #     torch.sum(torch.exp(logits), dim=1)
                # ).mean()
                #
                # # 3. 互信息最大化 (避免 collapse)
                # # 使用 Jensen-Shannon MI estimator (近似 mutual information)
                # mi_loss = -torch.mean(torch.sum(p * torch.log(q + 1e-6), dim=-1))
                #
                # # 总的因果对比损失
                # L_cf = L_cf_kl + self.hp.lambda_supcon * supcon_loss + self.hp.lambda_mi * mi_loss
# ===================== 因果敏感性反事实损失 ===================== #
                # factual_attention = factual_text
                # counter_attention = counter_text
                L_cf = causal_sensitivity_cf_loss(factual_text, counter_text.detach())

                h_loss = nn.L1Loss()

                single_loss = h_loss(r_preds, r_labels)
                fusion_loss = h_loss(r_preds_F, r_labels)

                #如果是分类任务，使用交叉熵损失
                # criterion_cls = nn.CrossEntropyLoss()
                # classification_loss = criterion_cls(c_preds, c_labels)
                # 使用自定义损失层计算总损失

                #loss = custom_loss_layer([single_loss, fusion_loss, rec_loss, dis_loss])下面加了lcf
                # loss_core = custom_loss_layer([single_loss, fusion_loss, recon_loss, dis_loss])
                # loss = loss_core + self.hp.lambda_cf * L_cf
                # ✅ 使用固定加权和（超参数需在 config.py 里设置）
                loss = (
                        self.hp.lambda_single * single_loss +
                        self.hp.lambda_fusion * fusion_loss +
                        self.hp.lambda_recon * recon_loss +
                        self.hp.lambda_dis * dis_loss +
                        self.hp.lambda_cf * L_cf
                )

                # 反传更新
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()
                scheduler.step()

                #loss = custom_loss_layer([single_loss, fusion_loss, rec_loss, dis_loss]) + self.hp.lambda_cf * L_cf
                # ✅ 用累计器收集 batch 损失，注意加 .item()
                single_loss_sum += single_loss.item() * batch_size
                fusion_loss_sum += fusion_loss.item() * batch_size
                recon_loss_sum += recon_loss.item() * batch_size
                dis_loss_sum += dis_loss.item() * batch_size
                epoch_loss_sum += loss.item() * batch_size  # ✅ 总损失累计
                proc_size += batch_size
                # loss = custom_loss_layer([single_loss, fusion_loss])
                # loss = y_loss + classification_loss + nce_loss / 6
                # loss = fusion_loss
                # optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                # optimizer.step()
                # scheduler.step()
                # s_loss += single_loss.item() * batch_size
                # fusion_loss += fusion_loss.item() * batch_size
                # #recon_loss += recon_loss * batch_size
                # recon_loss_sum += recon_loss.item() * batch_size
                # dis_loss += dis_loss * batch_size
                # proc_size += batch_size
                # epoch_loss += loss.item() * batch_size

                # 记录损失
                # writer.add_scalar('Training Loss', loss.item(), epoch * num_batches + i_batch)
                # 记录当前学习率
                # 记录每个参数组的学习率
                # if (i_batch + 1) % 10 == 0:  # 每10个batch记录一次，可以根据需要调整
                #     for name, param in model.named_parameters():
                #         writer.add_histogram(f'{name}/weights', param.data.cpu().numpy(), epoch * num_batches + i_batch)
                # for i, param_group in enumerate(optimizer.param_groups):
                #     lr = param_group['lr']
                #     writer.add_scalar(f'Learning Rate/Group_{i}', lr, epoch * num_batches + i_batch)

                if i_batch % 10 == 0 and i_batch > 0:
                    # ✅ 用累计器计算平均
                    avg_loss1 = single_loss_sum / proc_size
                    avg_loss2 = fusion_loss_sum / proc_size
                    avg_loss3 = recon_loss_sum / proc_size
                    avg_loss4 = dis_loss_sum / proc_size
                    # avg_loss1 = s_loss / proc_size
                    # avg_loss2 = fusion_loss / proc_size
                    # avg_loss3 = recon_loss/ proc_size
                    # avg_loss4 = dis_loss/ proc_size
                    elapsed_time = time.time() - start_time
                    # print(
                    #     'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | S_loss  {:5.4f} | F_loss {:5.4f} | REC {:5.4f}| DIS {:5.4f}'.
                    #     format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss1,
                    #            avg_loss2,avg_loss3,avg_loss4))
                    print(
                        'Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} '
                        '| S_loss {:5.4f} | F_loss {:5.4f} | REC {:5.4f} | DIS {:5.4f} | CF {:5.4f}'.
                        format(i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                               avg_loss1, avg_loss2, avg_loss3, avg_loss4, L_cf.item())
                    )

                    # print(
                    #     'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} '
                    #     '| S_loss {:5.4f} | F_loss {:5.4f} | REC {:5.4f} | DIS {:5.4f} | CF {:5.4f}'.
                    #     format(epoch, i_batch, num_batches,
                    #            elapsed_time * 1000 / self.hp.log_interval,
                    #            avg_loss1, avg_loss2, avg_loss3, avg_loss4,
                    #            L_cf.item())
                    # )# ✅ 清零累计器，不要清张量
                    single_loss_sum, fusion_loss_sum, recon_loss_sum, dis_loss_sum, proc_size = 0, 0, 0, 0, 0
                    #s_loss, fusion_loss,recon_loss,dis_loss,proc_size = 0, 0, 0,0,0
                    start_time = time.time()
            return epoch_loss_sum / self.hp.n_train

            #return epoch_loss / self.hp.n_train

        def evaluate(model, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader

            # 计算推理时间 在函数开头加变量 添加2行
            total_infer_time = 0
            num_batches = 0

            total_loss = 0.0

            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(loader):
                    visual, vlens, audio, alens, r_labels,c_labels,lengths, bert_sent, bert_sent_mask, ids = batch_data

                    with torch.cuda.device(0):
                        audio, visual, r_labels,c_labels = audio.cuda(), visual.cuda(), r_labels.cuda(), c_labels.cuda()
                        # print(visual.size())
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_mask = bert_sent.cuda(), bert_sent_mask.cuda()
                    batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)
#添加2行
                    torch.cuda.synchronize()
                    start = time.time()

                    r_preds,r_preds_F,recon_loss,dis_loss, _, _  = model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)

#添加4行
                    torch.cuda.synchronize()
                    end = time.time()
                    total_infer_time += (end - start)
                    num_batches += 1



                    # criterion = nn.SmoothL1Loss()
                    criterion = nn.L1Loss()
                    # criterion_cls = nn.CrossEntropyLoss()
                    s_loss = criterion(r_preds, r_labels)
                    fusion_loss = criterion(r_preds_F, r_labels)
                    # classification_loss = criterion_cls(c_preds, c_labels)
                    # total_loss += (s_loss.item()+fusion_loss.item()+recon_loss+dis_loss)*batch_size
                    # total_loss += (s_loss.item()+fusion_loss)*batch_size
                    #total_loss += (s_loss.item()+fusion_loss.item()+recon_loss+dis_loss)*batch_size
                    total_loss += (s_loss.item()+fusion_loss.item()+recon_loss.item()+dis_loss.item())*batch_size

                    res = (r_preds_F+r_preds)/2
                    results.append(res)
                    # results.append(r_preds)
                    # results.append(r_preds_F)
                    truths.append(r_labels)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
# 在函数最后（return 前）加输出 添加5行
            avg_infer_time = total_infer_time / num_batches
            print(f"Average inference time per batch: {avg_infer_time:.6f} seconds")
            # 如果你想要 per sample（推荐）
            batch_size = lengths.size(0)
            print(f"Inference time per sample: {avg_infer_time / batch_size:.6f} seconds")

            return avg_loss, results, truths
        # 关闭TensorBoard writer
        # writer.close()
        best_accu = 1e-8
        best_mae = 1e8
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"log/training_log_{current_time}.txt"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with open(log_filename, 'w') as log_file:
            for epoch in range(1, self.hp.num_epochs + 1):
                start = time.time()

                # minimize all losses left
                train_loss = train(model, optimizer, scheduler)

                val_loss, _, _ = evaluate(model, test=False)
                test_loss, results, truths = evaluate(model, test=True)
#可视化触发点
                # ------------------------------------------------------
                # 保存 epoch 的特征（IMPORTANT）
                # ------------------------------------------------------
                if epoch in [1, 10, 20, 30, 40]:
                    print(f"\n>>> Extracting features for epoch {epoch} ...")

                    feats = []
                    labs = []

                    self.model.eval()
                    with torch.no_grad():
                        for batch_data in self.test_loader:
                            visual, vlens, audio, alens, r_labels, c_labels, lengths, bert_sent, bert_sent_mask, ids = batch_data

                            visual = visual.to(self.device)
                            audio = audio.to(self.device)
                            bert_sent = bert_sent.to(self.device)
                            bert_sent_mask = bert_sent_mask.to(self.device)

                            r_preds, r_preds_F, recon_loss, dis_loss, factual_text, counter_text = \
                                self.model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)

                            feats.append(factual_text.cpu().numpy())
                            labs.append(r_labels.cpu().numpy())

                    feats = np.concatenate(feats, axis=0)
                    labs = np.concatenate(labs, axis=0)

                    np.save(f"features_epoch_{epoch}.npy", feats)
                    np.save(f"labels_epoch_{epoch}.npy", labs)
                    print(f">>> Saved features_epoch_{epoch}.npy / labels_epoch_{epoch}.npy")

                end = time.time()
                duration = end - start
                # scheduler.step(val_loss)

                epoch_info = f"Epoch {epoch} | Time {duration:.4f} sec | Train Loss {train_loss:.4f} | Valid Loss {val_loss:.4f} | Test Loss {test_loss:.4f}\n"
                log_file.write(epoch_info)
                print(epoch_info)

                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    accu, mae, res_dict = eval_mosei_senti(results, truths, True)
                elif self.hp.dataset == 'mosi':
                    accu, mae, res_dict = eval_mosei_senti(results, truths, True)
                print(f'accu: {accu}')
                print(f'best_accu: {best_accu}')
                print(f'mae: {mae}')
                print(f'best_mae: {best_mae}')
                if mae <= best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models of best_mae/MM.pt!")
                    save_model(self.hp, model, type='mae')

                if accu >= best_accu:
                    best_accu = accu
                    best_epoch = epoch
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models of best_acc/MM.pt!")
                    save_model(self.hp, model, type='acc')

                # 将每个epoch的评估结果写入文件
                if self.hp.dataset in ["mosi", "mosei"]:
                    accu, mae, eval_results = eval_mosei_senti(results, truths, False)
                    print('log generated')
                    log_file.write(f"BERT learning rate = {self.hp.lr_bert}\n")
                    log_file.write(f"Main learning rate = {self.hp.lr_main}\n")
                    log_file.write(f"Encoder Tower learning rate = {self.hp.lr_et}\n")
                    log_file.write(f"TSE learning rate = {self.hp.lr_te}\n")
                    log_file.write(f"a_size = {self.hp.a_size}\n")
                    log_file.write(f"v_size = {self.hp.v_size}\n")
                    log_file.write(f"step_ratio = {self.hp.step_ratio}\n")
                    log_file.write(f"TSElayer = {self.hp.tse_layers}\n")
                    log_file.write(f"ETlayer = {self.hp.ETlayers}\n")
                    log_file.write(f"weight_decay_te = {self.hp.weight_decay_te}\n")
                    for key, value in eval_results.items():
                        log_file.write(f"Epoch {epoch + 1}: {key} = {value}\n")
                    print('log wrote')
                log_file.write("-" * 50 + "\n")

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            eval_mosei_senti(best_results, best_truths, True)
        sys.stdout.flush()