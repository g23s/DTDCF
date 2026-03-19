from torch import nn
import torch
from modules.encoders import LanguageEmbeddingLayer, CPC, SubNet
from src.utils.Sinkhorn import SinkhornDistance,CPC,kl_divergence
from utils.Tri_aug_attention import Tri_aug_attlayer,F_S_Decoder
from utils.TimeEncoder import TimeEncoder,ConvLayer
from utils.debias import TextDebias

class MSA(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """

        # Base Encoders
        super(MSA,self).__init__()
        self.hp = hp
        self.text_enc = LanguageEmbeddingLayer(hp)
        # Trimodal Settings

        # self.pre_t = nn.Linear(hp.d_prjh, 1)
        # self.pre_a = nn.Linear(hp.d_prjh, 1)
        # self.pre_v = nn.Linear(hp.d_prjh, 1)
        # self.pre_f = nn.Linear(hp.d_prjh, hp.n_class)

        self.act_t = SubNet(
            in_size= 2*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        # self.fusion_prj2 = SubNet(
        #     in_size= 2*hp.d_prjh,
        #     hidden_size=hp.d_prjh,
        #     n_class=hp.n_class,
        #     dropout=hp.dropout_prj
        # )
        self.fusion_prj_s = SubNet(
            in_size= 3*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_f =SubNet(
            in_size= hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fc1=nn.Linear(768,hp.d_prjh)
        self.a_shape = nn.AdaptiveMaxPool1d(hp.a_size)
        self.v_shape = nn.AdaptiveMaxPool1d(hp.v_size)

        # 新增映射到 64
        self.proj_to64 = nn.Linear(768, 64)

        self.t_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)
        self.a_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)
        self.v_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)

        # self.t_shrink = nn.AdaptiveMaxPool1d(hp.dislen)
        # self.a_shrink = nn.AdaptiveMaxPool1d(hp.dislen)
        # self.v_shrink = nn.AdaptiveMaxPool1d(hp.dislen)

        self.ET_TA = Tri_aug_attlayer(hp)
        self.ET_TV = Tri_aug_attlayer(hp)
#文本去偏模块
        self.text_debias = TextDebias(
            "H:\why\MSATASE2\src\dict_npy\kmeans_mosi-200_text.npy",
            hidden_size=768
        )
        self.fc1 = nn.Linear(768, 768)  # 保持维度不变

        #在 __init__ 加 FusionMLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hp.d_prjh, hp.d_prjh),
            nn.ReLU(),
            nn.Linear(hp.d_prjh, hp.d_prjh)
        )
        #self.text_proj_to64 = nn.Linear(128, 64)

        self.f_proj_to192 = nn.Linear(hp.d_prjh, hp.d_prjh * 3)
        # 64 → 192，如果 hp.d_prjh = 64
        # 让 t_emb(128) 在接入 Decoder 之前对齐到 64 维
        self.t_student_align_to64 = nn.Linear(128, 64)

        # self.ET_AV = Tri_aug_attlayer(hp)


        self.decoder_t = F_S_Decoder(hp)
        self.decoder_a = F_S_Decoder(hp)
        self.decoder_v = F_S_Decoder(hp)


        #print(">>> decoder_t type:", type(self.decoder_t))

        # self.sinkhorn = SinkhornDistance()
        self.sentiment_fc1 = nn.Sequential(
            nn.Linear(hp.d_prjh, hp.d_prjh),
            nn.Tanh(),
            nn.Linear(hp.d_prjh, hp.d_prjh)
        )
        self.a_encoder = TimeEncoder(self.hp,encoder_layers=hp.tse_layers,step_ratio=hp.step_ratio, data_type='a')
        self.v_encoder = TimeEncoder(self.hp,encoder_layers=hp.tse_layers,step_ratio=hp.step_ratio, data_type='v')

        #self.num_segments = 8
        #self.tau = 1.0

        #self.granularity_proj_a = nn.Linear(hp.d_prjh, self.num_segments)
        #self.granularity_proj_v = nn.Linear(hp.d_prjh, self.num_segments)



    def forward(self,visual, acoustic, v_len, a_len, bert_sent,  bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(bert_sent,bert_sent_mask) # (batch_size, seq_len, emb_size)
        lang_emb=self.fc1(enc_word)
        #print("enc_word:", enc_word.shape)  # BERT 输出
        #print("lang_emb (fc1):", lang_emb.shape)  # fc1 之后
        # ★ 在 fc1 之后，shrink 之前去偏
        lang_content, lang_bias = self.text_debias(lang_emb)  # (B,L,128)# ★ 去偏后的结果再 shrink
        lang_content = self.proj_to64(lang_content)  # (B,L,64)
        #print("lang_content (after debias):", lang_content.shape)
        #print("lang_bias:", lang_bias.shape)
        lang_content = self.t_shrink(lang_content)  # (B,L,64)
        # aco_emb = self.a_shrink(aco_emb)  # (B,L,64)
        # vis_emb = self.v_shrink(vis_emb)
        acoustic = self.a_shape(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
        visual = self.v_shape(visual.permute(0, 2, 1)).permute(0, 2, 1)
        aco_emb=self.a_encoder(acoustic)
        vis_emb=self.v_encoder(visual)

        # ====== Gumbel Softmax 自适应粒度 ======

        # audio
        #logits_a = self.granularity_proj_a(aco_emb)  # (B,T,K)
        #g_a = torch.nn.functional.gumbel_softmax(
        #    logits_a, tau=self.tau, hard=False, dim=-1
        #)
        #aco_emb = torch.einsum('btk,btd->bkd', g_a, aco_emb)

        # visual
        #logits_v = self.granularity_proj_v(vis_emb)
        #g_v = torch.nn.functional.gumbel_softmax(
        #    logits_v, tau=self.tau, hard=False, dim=-1
        #)
        #vis_emb = torch.einsum('btk,btd->bkd', g_v, vis_emb)


        # lang_emb = self.t_shrink(lang_emb)
        aco_emb = self.a_shrink(aco_emb)
        vis_emb = self.v_shrink(vis_emb)

        #print("S_t input:", lang_content.shape)
        #print("S_a input:", aco_emb.shape)
        #print("S_v input:", vis_emb.shape)
        S_t =torch.mean(lang_content, dim=1)
        S_a =torch.mean(aco_emb, dim=1)
        S_v =torch.mean(vis_emb, dim=1)
        #print("S_t:", S_t.shape)
        #print("S_a:", S_a.shape)
        #print("S_v:", S_v.shape)
        S_feat = torch.cat((S_t,S_a,S_v), dim=-1)
        #print("S_feat:", S_feat.shape)
        _,r_preds = self.fusion_prj_s(S_feat)#通过单模态标签损失促使TSE捕获模态特定的情感信息
        #文本主导
        # TF_v, VF_t, TM, IM = self.ET_TV(lang_emb, vis_emb)
        # TF_a, AF_t, TM, IM = self.ET_TA(lang_emb, aco_emb)
        # 文本去偏
        #lang_content, lang_bias = self.text_debias(lang_emb)
    #加了反事实
        # 用 lang_content 进入 cross-attention
        TF_v, VF_t, TM, IM, CF_TF_v, CF_VF_t = self.ET_TV(lang_content, vis_emb)
        TF_a, AF_t, TM, IM, CF_TF_a, CF_AF_t = self.ET_TA(lang_content, aco_emb)
        # AF_v, VF_a, TM, IM = self.ET_AV(aco_emb, vis_emb)
        #print("TF_a:", TF_a.shape)
        #print("TF_v:", TF_v.shape)
        t_emb = torch.cat((TF_a, TF_v), dim=-1)
        #print("t_emb (concat):", t_emb.shape)
        #t_emb,_ = self.act_t(t_emb)
    #加了反事实
        T_dfeat = torch.mean(t_emb, dim=1)
        #T_dfeat = self.text_proj_to64(T_dfeat)  # [B, 64] ✅ 匹配 sentiment_fc1
        #print("T_dfeat (pooled):", T_dfeat.shape)
        factual_text = self.fusion_mlp(T_dfeat)
        #print("fusion_mlp input dim:", self.fusion_mlp[0].in_features)

    # counterfactual
        cf_t_emb = torch.cat((CF_TF_a, CF_TF_v), dim=-1)
        #cf_t_emb, _ = self.act_t(cf_t_emb)
        CF_dfeat = torch.mean(cf_t_emb, dim=1)  # [B, 2*D]  ★ 池化
        #CF_dfeat = self.text_proj_to64(CF_dfeat)  # [B, 64]
        counter_text = self.fusion_mlp(CF_dfeat)  # 分类头

        # # 音频主导
        # AF_v, VF_a, TM, IM = self.ET_TV(aco_emb, vis_emb)
        # AF_t, TF_a, TM, IM = self.ET_TA(aco_emb, lang_emb)
        # a_emb = torch.cat((AF_v, AF_t), dim=-1)
        # a_emb, _ = self.act_t(a_emb)
        # T_dfeat = torch.mean(TF_a, dim=1)
        # A_dfeat = torch.mean(a_emb, dim=1)
        # V_dfeat = torch.mean(VF_a, dim=1)

        # # 视频主导
        # VF_a, AF_v, TM, IM = self.ET_TV(vis_emb, aco_emb)
        # VF_t, TF_v, TM, IM = self.ET_TA(vis_emb, lang_emb)
        # v_emb = torch.cat((VF_a, VF_t), dim=-1)
        # v_emb, _ = self.act_t(v_emb)
        # T_dfeat = torch.mean(TF_v, dim=1)
        # A_dfeat = torch.mean(AF_v, dim=1)
        # V_dfeat = torch.mean(v_emb, dim=1)
        # F_feat,r_preds_F = self.fusion_prj4(F_feat)

        A_dfeat = torch.mean(AF_t, dim=1)
        V_dfeat = torch.mean(VF_t, dim=1)
        T_dfeat_for_sentiment = torch.mean(lang_content, dim=1)  # [B, 64]
        #print("T_dfeat_for_sentiment:", T_dfeat_for_sentiment.shape)
        #print("A_dfeat:", A_dfeat.shape)
        #print("V_dfeat:", V_dfeat.shape)
         # 将特征投影到情感空间
        T_emb = self.sentiment_fc1(T_dfeat_for_sentiment)   # 64 → 64
        A_emb = self.sentiment_fc1(A_dfeat)
        V_emb = self.sentiment_fc1(V_dfeat)
        #print("T_emb (sentiment space):", T_emb.shape)
        #print("A_emb (sentiment space):", A_emb.shape)
        #print("V_emb (sentiment space):", V_emb.shape)
        # 使用欧氏距离计算距离
        cost1 = torch.norm(T_emb - A_emb, dim=-1, p=2)
        cost2 = torch.norm(T_emb - V_emb, dim=-1, p=2)
        cost3 = torch.norm(A_emb - V_emb, dim=-1, p=2)
        # 扩展距离以匹配特征维度
        cost_ta = cost1.unsqueeze(-1)
        cost_tv = cost2.unsqueeze(-1)
        cost_av = cost3.unsqueeze(-1)

        # 计算距离的倒数，避免除以零
        epsilon = 1e-6  # 避免除以零的小正则项
        inv_T = 1 / (cost_ta + cost_tv + epsilon)
        inv_A = 1 / (cost_ta + cost_av + epsilon)
        inv_V = 1 / (cost_tv + cost_av + epsilon)

        # 归一化倒数以得到权重
        total_inv_cost = inv_T + inv_A + inv_V
        weight_T = inv_T / total_inv_cost
        weight_A = inv_A / total_inv_cost
        weight_V = inv_V / total_inv_cost

        #特征融合
        F_feat = weight_T * T_dfeat_for_sentiment + weight_A * A_dfeat + weight_V * V_dfeat

        #print("F_feat before proj:", F_feat.shape)  # [B, 64]
        _, r_preds_F = self.fusion_prj_f(F_feat)  # ✅ logits丢弃，只保留预测
        F_feat_proj = self.f_proj_to192(F_feat)
        #print("F_feat_proj:", F_feat_proj.shape)  # [B, 192]


        #F_feat = self.f_proj_to192(F_feat)  # [B, 192]

        # F_feat = torch.cat((lang_emb, aco_emb, vis_emb), dim=1)
        # F_feat = self.f_shrink(F_feat)
        # 先把 student 端的 t_emb 从 128 → 64，对齐 teacher 的 lang_content(… ,64)
        t_student_64 = self.t_student_align_to64(t_emb)  # [B, 90, 64]
        t_recon = self.decoder_t(t_student_64,lang_content).mean(dim=1)

        a_recon = self.decoder_a(AF_t,aco_emb).mean(dim=1)
        v_recon = self.decoder_v(VF_t,vis_emb).mean(dim=1)
        kl_div_t_recon = kl_divergence(S_t, t_recon)
        kl_div_a_recon = kl_divergence(S_a, a_recon)
        kl_div_v_recon = kl_divergence(S_v, v_recon)
        kl_div_t_distill = kl_divergence(F_feat, S_t)
        kl_div_a_distill = kl_divergence(F_feat, S_a)
        kl_div_v_distill = kl_divergence(F_feat, S_v)#双向kl散度
        L_recon = (kl_div_t_recon + kl_div_a_recon + kl_div_v_recon) / 3
        L_dis = (kl_div_t_distill + kl_div_a_distill + kl_div_v_distill) / 3
        # return r_preds,r_preds_F,L_recon,L_dis,weight_T, weight_A, weight_V
        return r_preds,r_preds_F,L_recon,L_dis, factual_text, counter_text#L_recon,


#加了, factual_text, counter_text