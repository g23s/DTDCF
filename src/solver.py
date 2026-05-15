from torch import nn
import sys
import datetime
import time
from utils.eval_metrics import *
from utils.tools import *
from MSA import MSA
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    from transformers import WarmupLinearSchedule

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        return WarmupLinearSchedule(
            optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_training_steps
        )
from src.utils.Sinkhorn import CustomMultiLossLayer
# from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn.functional as F

def irm_loss(fused, unimodal):
    return torch.mean((fused - unimodal)**2)
#用 MSE 做稳健性约束
def robustness_loss(factual_repr, perturbed_repr):
    return F.mse_loss(factual_repr, perturbed_repr)

def format_dtm_debug(model):
    debug = getattr(model, "dtm_debug", None)
    if not debug or debug.get("g_a") is None or debug.get("g_v") is None:
        return ""

    tau_a = debug["tau_a"].mean().item()
    tau_v = debug["tau_v"].mean().item()
    g_a = debug["g_a"].mean(dim=0).detach().cpu().tolist()
    g_v = debug["g_v"].mean(dim=0).detach().cpu().tolist()
    g_a_str = ",".join(f"{x:.2f}" for x in g_a)
    g_v_str = ",".join(f"{x:.2f}" for x in g_v)
    return f" | tau_a {tau_a:.3f} | tau_v {tau_v:.3f} | g_a [{g_a_str}] | g_v [{g_v_str}]"

def set_dtm_gumbel_temperature(model, epoch, total_epochs, start_tau, min_tau):
    if total_epochs <= 1:
        tau = min_tau
    else:
        progress = (epoch - 1) / (total_epochs - 1)
        tau = start_tau + (min_tau - start_tau) * progress
    tau = max(min_tau, tau)
    for module in model.modules():
        if hasattr(module, "gumbel_temperature"):
            module.gumbel_temperature = tau
    return tau

def set_dtm_hard_selection(model, enabled):
    for module in model.modules():
        if hasattr(module, "dtm_hard_selection"):
            module.dtm_hard_selection = enabled

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
            # =========================
            # 打印模型参数量（新增）
            # =========================
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("========== Model Parameters ==========")
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
            print(f"Total parameters (Million): {total_params / 1e6:.2f} M")
            print("======================================")
            # =========================
            # 单独统计 BERT 参数量
            # =========================
            bert_params = sum(p.numel() for name, p in self.model.named_parameters() if 'bert' in name.lower())

            print(f"BERT parameters: {bert_params}")
            print(f"BERT parameters (Million): {bert_params / 1e6:.2f} M")





        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)
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

        SAVE_TSNE_EPOCHS = [1, 5, 10]

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
                visual = visual.to(self.device)
                audio = audio.to(self.device)
                r_labels = r_labels.to(self.device)
                c_labels = c_labels.to(self.device)
                l = l.to(self.device)
                bert_sent = bert_sent.to(self.device)
                bert_sent_mask = bert_sent_mask.to(self.device)
                batch_size = r_labels.size(0)
                # 前向返回 recon_loss 和 dis_loss 是张量，不要和累计器混淆
                #r_preds, r_preds_F, recon_loss, dis_loss, factual_text, counter_text = model(visual, audio, v_size, a_size, bert_sent, bert_sent_mask)

                model_out = model(visual, audio, v_size, a_size, bert_sent, bert_sent_mask)
                if len(model_out) == 8:
                    r_preds, r_preds_F, recon_loss, dis_loss, factual_text, counter_text, F_feat, irm_loss = model_out
                else:
                    r_preds, r_preds_F, recon_loss, dis_loss, factual_text, counter_text, F_feat = model_out
                    irm_loss = torch.tensor(0.0, device=self.device)

                L_rob = robustness_loss(factual_text, counter_text.detach())


                #L_rob = torch.tensor(0.0, device=self.device)
                h_loss = nn.L1Loss()

                single_loss = h_loss(r_preds, r_labels)
                fusion_loss = h_loss(r_preds_F, r_labels)

                #  使用固定加权和（超参数需在 config.py 里设置）
                loss = (
                        self.hp.lambda_single * single_loss +
                        self.hp.lambda_fusion * fusion_loss +
                        self.hp.lambda_recon * recon_loss +
                        self.hp.lambda_dis * dis_loss +
                        self.hp.lambda_rob * L_rob +
                        self.hp.lambda_irm * irm_loss
                )
                #+self.hp.lambda_rob * L_rob

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

                    elapsed_time = time.time() - start_time

                    print(
                        'Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} '
                        '| S_loss {:5.4f} | F_loss {:5.4f} | REC {:5.4f} | DIS {:5.4f} | ROB {:5.4f} | IRM {:5.4f}{}'.
                        format(i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                               avg_loss1, avg_loss2, avg_loss3, avg_loss4, L_rob.item(), irm_loss.item(), format_dtm_debug(model))
                    )


                    single_loss_sum, fusion_loss_sum, recon_loss_sum, dis_loss_sum, proc_size = 0, 0, 0, 0, 0

                    start_time = time.time()
            return epoch_loss_sum / self.hp.n_train

            #return epoch_loss / self.hp.n_train

        def evaluate(model, test=False, epoch=None):

        #def evaluate(model, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader

            # =========================
            # Inference efficiency metrics
            # =========================
            measure_efficiency = test and (epoch == self.hp.num_epochs)

            total_infer_time = 0.0
            total_infer_samples = 0
            total_infer_batches = 0

            if measure_efficiency and torch.cuda.is_available():
                 torch.cuda.empty_cache()
                 torch.cuda.reset_peak_memory_stats()
                 torch.cuda.synchronize()

            # 计算推理时间 在函数开头加变量 添加2行
           # total_infer_time = 0
           # num_batches = 0

            total_loss = 0.0

            results = []
            truths = []
            feat_list = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(loader):
                    visual, vlens, audio, alens, r_labels,c_labels,lengths, bert_sent, bert_sent_mask, ids = batch_data

                    audio = audio.to(self.device)
                    visual = visual.to(self.device)
                    r_labels = r_labels.to(self.device)
                    c_labels = c_labels.to(self.device)
                    # print(visual.size())
                    lengths = lengths.to(self.device)
                    bert_sent = bert_sent.to(self.device)
                    bert_sent_mask = bert_sent_mask.to(self.device)
                    batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)
#添加2行
                  #  torch.cuda.synchronize()
                  #  start = time.time()

                    #r_preds,r_preds_F,recon_loss,dis_loss, _, _  = model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)


                    #r_preds, r_preds_F, recon_loss, dis_loss, _, _, F_feat = model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)

                    # =========================
                    # Measure pure inference time
                    # =========================
                    if measure_efficiency and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        infer_start = time.time()

                    model_out = model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)
                    r_preds, r_preds_F, recon_loss, dis_loss, _, _, F_feat = model_out[:7]

                    if measure_efficiency and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        infer_end = time.time()

                        total_infer_time += (infer_end - infer_start)
                        total_infer_samples += batch_size
                        total_infer_batches += 1

#添加4行
                  #  torch.cuda.synchronize()
                  #  end = time.time()
                  #  total_infer_time += (end - start)
                   # num_batches += 1



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
                    if test:
                        feat_list.append(F_feat.detach().cpu())

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

        # =========================
        # Print inference efficiency metrics
        # =========================
            if measure_efficiency and total_infer_batches > 0:
                 avg_infer_time_per_batch = total_infer_time / total_infer_batches
                 throughput = total_infer_samples / total_infer_time

                 if torch.cuda.is_available():
                      peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                 else:
                      peak_memory_gb = 0.0

                 print("========== Inference Efficiency ==========")
                 print(f"Peak GPU Memory: {peak_memory_gb:.3f} GB")
                 print(f"Total inference time: {total_infer_time:.6f} s")
                 print(f"Average inference time per batch: {avg_infer_time_per_batch:.6f} s/batch")
                 print(f"Throughput: {throughput:.2f} samples/s")
                 print("==========================================")




            results = torch.cat(results)
            truths = torch.cat(truths)

# 在函数最后（return 前）加输出 添加5行
          #  avg_infer_time = total_infer_time / num_batches
           # print(f"Average inference time per batch: {avg_infer_time:.6f} seconds")
            # 如果你想要 per sample（推荐）
         #   batch_size = lengths.size(0)
         #   print(f"Inference time per sample: {avg_infer_time / batch_size:.6f} seconds")

            #return avg_loss, results, truths

            # if test and len(feat_list) > 0:
            #     import numpy as np
            #     import os
            #
            #     save_dir = r"H:\why\MSATASE2\tsne_cache"
            #     os.makedirs(save_dir, exist_ok=True)
            #
            #     F_feats = torch.cat(feat_list, dim=0).numpy()
            #     y_labels = truths.detach().cpu().numpy()
            #
            #
            #
            #     np.save(os.path.join(save_dir, "DTDCF_F_feat.npy"), F_feats)
            #     np.save(os.path.join(save_dir, "DTDCF_labels.npy"), y_labels)
            #
            #     print("t-SNE features saved to:", save_dir)

            if test and epoch in SAVE_TSNE_EPOCHS and len(feat_list) > 0:
                  import numpy as np
                  import os

                  save_dir = r"H:\why\MSATASE2\tsne_epoch_cache"
                  os.makedirs(save_dir, exist_ok=True)

                  F_feats = torch.cat(feat_list, dim=0).numpy()
                  y_labels = truths.detach().cpu().numpy()

                  np.save(os.path.join(save_dir, f"F_feat_epoch_{epoch}.npy"), F_feats)
                  np.save(os.path.join(save_dir, f"labels_epoch_{epoch}.npy"), y_labels)

                  print(f"t-SNE features saved for epoch {epoch} to:", save_dir)






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
                dtm_tau = set_dtm_gumbel_temperature(
                    model,
                    epoch,
                    self.hp.num_epochs,
                    self.hp.dtm_gumbel_tau,
                    self.hp.dtm_gumbel_tau_min
                )
                hard_after = int(getattr(self.hp, "dtm_hard_after_epoch", 0))
                hard_enabled = bool(getattr(self.hp, "use_dtm_hard_train", 1)) and hard_after > 0 and epoch >= hard_after
                set_dtm_hard_selection(model, hard_enabled)
                tau_info = f"DTM Gumbel temperature: {dtm_tau:.4f} | hard_selection: {int(hard_enabled)}\n"
                log_file.write(tau_info)
                print(tau_info.strip())

                # minimize all losses left
                train_loss = train(model, optimizer, scheduler)

                # val_loss, _, _ = evaluate(model, test=False)
                # test_loss, results, truths = evaluate(model, test=True)
                val_loss, _, _ = evaluate(model, test=False, epoch=epoch)
                test_loss, results, truths = evaluate(model, test=True, epoch=epoch)

#可视化触发点
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
