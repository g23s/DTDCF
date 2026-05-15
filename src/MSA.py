from torch import nn
import torch
from modules.encoders import LanguageEmbeddingLayer, CPC, SubNet
from src.utils.Sinkhorn import SinkhornDistance,CPC,kl_divergence
from utils.Tri_aug_attention import Tri_aug_attlayer,F_S_Decoder
from utils.TimeEncoder import TimeEncoder,ConvLayer

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

        self.act_t = SubNet(
            in_size= 2*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

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

        self.ET_TA = Tri_aug_attlayer(hp)
        self.ET_TV = Tri_aug_attlayer(hp)
#文本去偏模块
        self.fc1 = nn.Linear(768, 768)  # 保持维度不变

        #在 __init__ 加 FusionMLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hp.d_prjh, hp.d_prjh),
            nn.ReLU(),
            nn.Linear(hp.d_prjh, hp.d_prjh)
        )

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

# ===== 时间可靠性头：给 DTM 输出一个可靠性分数 =====在 __init__ 里加“时间可靠性头”
        self.a_reliability_head = nn.Sequential(
            nn.Linear(hp.d_prjh, hp.d_prjh),
            nn.ReLU(),
            nn.Linear(hp.d_prjh, 1),
            nn.Sigmoid()
        )

        self.v_reliability_head = nn.Sequential(
            nn.Linear(hp.d_prjh, hp.d_prjh),
            nn.ReLU(),
            nn.Linear(hp.d_prjh, 1),
            nn.Sigmoid()
        )

# ===== Gaussian 分布参数头：每个模态输出 mean 和 std =====
        self.mu_t_head = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.logvar_t_head = nn.Linear(hp.d_prjh, hp.d_prjh)

        self.mu_a_head = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.logvar_a_head = nn.Linear(hp.d_prjh, hp.d_prjh)

        self.mu_v_head = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.logvar_v_head = nn.Linear(hp.d_prjh, hp.d_prjh)

# ===== 扰动分支的融合 MLP =====
        self.perturb_fusion_mlp = nn.Sequential(
            nn.Linear(3 * hp.d_prjh, hp.d_prjh),
            nn.ReLU(),
            nn.Linear(hp.d_prjh, hp.d_prjh)
        )
        self.use_dcf_qkv_confidence = bool(getattr(hp, "use_dcf_qkv_confidence", 1))
        self.use_dcf_output_gate = bool(getattr(hp, "use_dcf_output_gate", 0))
        self.dcf_perturb_eta = float(getattr(hp, "dcf_perturb_eta", 1.0))
        self.use_irm = bool(getattr(hp, "use_irm", 1))
        self.irm_sigma = float(getattr(hp, "irm_sigma", 1.0))
        self.use_irm_subset_gate = bool(getattr(hp, "use_irm_subset_gate", 1))
        self.irm_text_gate = nn.Sequential(
            nn.Linear(2 * hp.d_prjh, hp.d_prjh),
            nn.Sigmoid()
        )
        self.irm_audio_gate = nn.Sequential(
            nn.Linear(2 * hp.d_prjh, hp.d_prjh),
            nn.Sigmoid()
        )
        self.irm_visual_gate = nn.Sequential(
            nn.Linear(2 * hp.d_prjh, hp.d_prjh),
            nn.Sigmoid()
        )

#这是高斯采样的核心函数 从 mu 和 std 生成扰动表示 新第二分支的起点
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + self.dcf_perturb_eta * std * eps

    def _cauchy_schwarz_loss(self, x, y):
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)
        xx = torch.cdist(x, x, p=2).pow(2)
        yy = torch.cdist(y, y, p=2).pow(2)
        xy = torch.cdist(x, y, p=2).pow(2)
        denom = 2 * (self.irm_sigma ** 2)
        k_xx = torch.exp(-xx / denom).mean().clamp_min(1e-8)
        k_yy = torch.exp(-yy / denom).mean().clamp_min(1e-8)
        k_xy = torch.exp(-xy / denom).mean().clamp_min(1e-8)
        return -2 * torch.log(k_xy) + torch.log(k_xx) + torch.log(k_yy)

    def _irm_loss(self, fused_seq, text_seq, audio_seq, visual_seq, c_t, c_a, c_v):
        if not self.use_irm:
            return fused_seq.new_tensor(0.0)
        ct = c_t.mean()
        ca = c_a.mean()
        cv = c_v.mean()
        l_ccs1 = (
            ct * self._cauchy_schwarz_loss(fused_seq, text_seq) +
            ca * self._cauchy_schwarz_loss(fused_seq, audio_seq) +
            cv * self._cauchy_schwarz_loss(fused_seq, visual_seq)
        ) / 3
        if self.use_irm_subset_gate:
            text_gate = self.irm_text_gate(torch.cat((text_seq, fused_seq), dim=-1))
            audio_gate = self.irm_audio_gate(torch.cat((audio_seq, fused_seq), dim=-1))
            visual_gate = self.irm_visual_gate(torch.cat((visual_seq, fused_seq), dim=-1))
            conditioned_text = c_t * text_gate * text_seq
            conditioned_audio = c_a * audio_gate * audio_seq
            conditioned_visual = c_v * visual_gate * visual_seq
        else:
            conditioned_text = c_t * text_seq
            conditioned_audio = c_a * audio_seq
            conditioned_visual = c_v * visual_seq
        l_ccs2 = (
            ct * self._cauchy_schwarz_loss(conditioned_text, text_seq) +
            ca * self._cauchy_schwarz_loss(conditioned_audio, audio_seq) +
            cv * self._cauchy_schwarz_loss(conditioned_visual, visual_seq)
        ) / 3
        return l_ccs1 + l_ccs2

    def forward(self,visual, acoustic, v_len, a_len, bert_sent,  bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(bert_sent,bert_sent_mask) # (batch_size, seq_len, emb_size)
        lang_emb=self.fc1(enc_word)

        lang_content = self.proj_to64(lang_emb)  # (B,L,64)

        lang_content = self.t_shrink(lang_content)  # (B,L,64)

        acoustic = self.a_shape(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
        visual = self.v_shape(visual.permute(0, 2, 1)).permute(0, 2, 1)
        a_encoder_out = self.a_encoder(acoustic)
        v_encoder_out = self.v_encoder(visual)
        if isinstance(a_encoder_out, tuple):
            aco_emb, tau_a, g_a = a_encoder_out
        else:
            aco_emb = a_encoder_out
            tau_a = torch.ones(aco_emb.size(0), 1, device=aco_emb.device)
            g_a = None
        if isinstance(v_encoder_out, tuple):
            vis_emb, tau_v, g_v = v_encoder_out
        else:
            vis_emb = v_encoder_out
            tau_v = torch.ones(vis_emb.size(0), 1, device=vis_emb.device)
            g_v = None
        self.dtm_debug = {
            "tau_a": tau_a.detach(),
            "tau_v": tau_v.detach(),
            "g_a": None if g_a is None else g_a.detach(),
            "g_v": None if g_v is None else g_v.detach(),
        }

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

        # ===== 1. 时间可靠性（只给视频/音频） =====
        u_a = self.a_reliability_head(S_a)  # [B, 1]
        u_v = self.v_reliability_head(S_v)  # [B, 1]

        # ===== 2. Gaussian 分布参数 =====
        mu_t_seq = self.mu_t_head(lang_content)
        std_t_seq = torch.nn.functional.softplus(self.logvar_t_head(lang_content)) + 1e-6
        mu_a_seq = self.mu_a_head(aco_emb)
        std_a_seq = torch.nn.functional.softplus(self.logvar_a_head(aco_emb)) + 1e-6
        mu_v_seq = self.mu_v_head(vis_emb)
        std_v_seq = torch.nn.functional.softplus(self.logvar_v_head(vis_emb)) + 1e-6

        mu_t = torch.mean(mu_t_seq, dim=1)
        std_t = torch.mean(std_t_seq, dim=1)
        mu_a = torch.mean(mu_a_seq, dim=1)
        std_a = torch.mean(std_a_seq, dim=1)
        mu_v = torch.mean(mu_v_seq, dim=1)
        std_v = torch.mean(std_v_seq, dim=1)

        # ===== 3. 分布置信度 =====
        r_t_seq = torch.exp(-std_t_seq)
        r_a_seq = tau_a.unsqueeze(1) * u_a.unsqueeze(1) * torch.exp(-std_a_seq)
        r_v_seq = tau_v.unsqueeze(1) * u_v.unsqueeze(1) * torch.exp(-std_v_seq)

        S_feat = torch.cat((S_t,S_a,S_v), dim=-1)
        #print("S_feat:", S_feat.shape)
        _,r_preds = self.fusion_prj_s(S_feat)#通过单模态标签损失促使TSE捕获模态特定的情感信息

        qkv_r_t = r_t_seq if self.use_dcf_qkv_confidence else None
        qkv_r_a = r_a_seq if self.use_dcf_qkv_confidence else None
        qkv_r_v = r_v_seq if self.use_dcf_qkv_confidence else None
        TF_v, VF_t, TM, IM, _, _ = self.ET_TV(lang_content, vis_emb, qkv_r_t, qkv_r_v)
        TF_a, AF_t, TM, IM, _, _ = self.ET_TA(lang_content, aco_emb, qkv_r_t, qkv_r_a)
        # ===== 可靠性门控：让模态可靠性直接调节 cross-attention 输出 =====
        if self.use_dcf_output_gate:
            TF_v = r_v_seq * TF_v
            VF_t = r_t_seq * VF_t

            TF_a = r_a_seq * TF_a
            AF_t = r_t_seq * AF_t

        # ===== 主分支：原始表示融合 =====
        t_emb = torch.cat((TF_a, TF_v), dim=-1)
        factual_text = self.fusion_mlp(t_emb)
        T_dfeat = torch.mean(factual_text, dim=1)

        # ===== 分布扰动分支：先采样，再走同样的 attention 结构 =====扰动表示 → attention → fusion 这样两边就是同构的 L_rob更有意义
        #1. 从 Gaussian 分布采样扰动后的全局表示
        z_t_pert_seq = self.reparameterize(mu_t_seq, std_t_seq)  # [B, Lt, D]
        z_a_pert_seq = self.reparameterize(mu_a_seq, std_a_seq)  # [B, La, D]
        z_v_pert_seq = self.reparameterize(mu_v_seq, std_v_seq)  # [B, Lv, D]
# #扰动更大
#         noise_scale = 2.0  # 你可以试 1.5 / 2.0 / 3.0
#         z_t_pert = mu_t + noise_scale * std_t * torch.randn_like(std_t)
#         z_a_pert = mu_a + noise_scale * std_a * torch.randn_like(std_a)
#         z_v_pert = mu_v + noise_scale * std_v * torch.randn_like(std_v)

        # 2. 把全局向量扩展成“伪序列”，长度与原序列保持一致

        # 3. 走和主分支一样的 cross-attention
        TF_v_p, VF_t_p, _, _, _, _ = self.ET_TV(z_t_pert_seq, z_v_pert_seq, qkv_r_t, qkv_r_v)
        TF_a_p, AF_t_p, _, _, _, _ = self.ET_TA(z_t_pert_seq, z_a_pert_seq, qkv_r_t, qkv_r_a)

        # 4. 对扰动分支也施加同样的 reliability gating
        if self.use_dcf_output_gate:
            TF_v_p = r_v_seq * TF_v_p
            TF_a_p = r_a_seq * TF_a_p

        # 如果你想让文本可靠性也参与，可以打开下面两行；第一版先不要开
        # VF_t_p = r_t_seq * VF_t_p
        # AF_t_p = r_t_seq * AF_t_p

        # 5. 用和主分支相同的方式融合
        t_emb_p = torch.cat((TF_a_p, TF_v_p), dim=-1)
        counter_text = self.fusion_mlp(t_emb_p)

        A_dfeat = torch.mean(AF_t, dim=1)
        V_dfeat = torch.mean(VF_t, dim=1)
        T_dfeat_for_sentiment = torch.mean(lang_content, dim=1)  # [B, 64]

        T_emb = self.sentiment_fc1(T_dfeat_for_sentiment)   # 64 → 64
        A_emb = self.sentiment_fc1(A_dfeat)
        V_emb = self.sentiment_fc1(V_dfeat)

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


        _, r_preds_F = self.fusion_prj_f(F_feat)  # ✅ logits丢弃，只保留预测
        F_feat_proj = self.f_proj_to192(F_feat)

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
        irm_loss = self._irm_loss(factual_text, lang_content, AF_t, VF_t,
                                  r_t_seq, r_a_seq, r_v_seq)
        # return r_preds,r_preds_F,L_recon,L_dis,weight_T, weight_A, weight_V


        #return r_preds,r_preds_F,L_recon,L_dis, factual_text, counter_text

        return r_preds, r_preds_F, L_recon, L_dis, factual_text, counter_text, F_feat, irm_loss
