import torch
from torch import nn
import math
from transformers import BertConfig
from .bert_model import BertCrossLayer
#在文件顶部增加 CounterfactualOps
class CounterfactualOps:
    def __init__(self, method="random"):
        self.method = method

    def __call__(self, attn_weights):
        if self.method == "random":
            rand = torch.rand_like(attn_weights)
            attn_weights = rand * (attn_weights != 0)
        elif self.method == "shuffle":
            idx = torch.randperm(attn_weights.size(-1), device=attn_weights.device)
            attn_weights = attn_weights[..., idx]
        elif self.method == "reversed":
            attn_weights = torch.flip(attn_weights, dims=[-1])
        elif self.method == "mask":
            mask = (torch.rand_like(attn_weights) > 0.3).float()
            attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        return attn_weights


class Tri_aug_attlayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.opt = dict(
            EThidden_size=args.d_prjh,
            multihead=args.multi_head,
            ETlayers=args.ETlayers
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_config = BertConfig(
                    hidden_size=self.opt['EThidden_size'],
                    num_attention_heads=self.opt['multihead'],
                    intermediate_size=self.opt['EThidden_size'] * 4,
                    hidden_dropout_prob=0.3,
                    attention_probs_dropout_prob=0.3,
        )
        self.cross_modal_layers1 = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(self.opt['ETlayers'])])
        self.cross_modal_layers2 = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(self.opt['ETlayers'])])
        # self.x_shrink = ConvLayer(args.d_prjh)
        # self.y_shrink = ConvLayer(args.d_prjh)
    def forward(

        self,
        lang_emb,
        img_emb,
        lang_conf=None,
        img_conf=None,
        ):
        x = lang_emb
        y = img_emb
        text_masks = torch.zeros([x.shape[0], 1, 1, x.shape[1]], dtype=torch.bool,device=lang_emb.device)
        extend_image_masks = torch.ones((y.size(0),1,1,y.size(1)), dtype=torch.long, device=self.device)
        link_layer_index = 0
        last_layer=False
        for i in range(self.opt['ETlayers']):
            if i == self.opt['ETlayers'] - 1:
                last_layer=True
            #DBIT
            # image_textaug_embeds = self.cross_modal_layers1[link_layer_index](x, y, text_masks,extend_image_masks,last_layer)
            # text_imageaug_embeds = self.cross_modal_layers2[link_layer_index](y, x,extend_image_masks, text_masks,last_layer)
            # #普通CA
            # text_imageaug_embeds = self.cross_modal_layers1[link_layer_index](x, y, text_masks,extend_image_masks,last_layer, mode="factual")
            # image_textaug_embeds = self.cross_modal_layers2[link_layer_index](y, x,extend_image_masks, text_masks,last_layer, mode="factual")
            text_imageaug_embeds, cf_text_embeds = self.cross_modal_layers1[link_layer_index](
                x, y, text_masks, extend_image_masks, last_layer,
                query_confidence=lang_conf,
                key_value_confidence=img_conf
            )
            image_textaug_embeds, cf_image_embeds = self.cross_modal_layers2[link_layer_index](
                y, x, extend_image_masks, text_masks, last_layer,
                query_confidence=img_conf,
                key_value_confidence=lang_conf
            )

            # text_feats = self.x_shrink(x)
        # image_feats= self.y_shrink(y)

            # cf_text_embeds = self.cross_modal_layers1[link_layer_index](x, y, text_masks, extend_image_masks, last_layer,
            #                                                         mode="counterfactual")
            # cf_image_embeds = self.cross_modal_layers2[link_layer_index](y, x, extend_image_masks, text_masks, last_layer,
            #                                                          mode="counterfactual")
            x = text_imageaug_embeds
            y = image_textaug_embeds
            link_layer_index += 1
        text_feats, image_feats = x, y

        return text_feats, image_feats, text_masks, extend_image_masks, cf_text_embeds, cf_image_embeds
    #加了, cf_text_embeds, cf_image_embeds
# class BertCrossLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention = Attention(config,type='sa')
#         # self.is_decoder = config.is_decoder
#         # self.add_cross_attention = config.add_cross_attention
#         self.crossattention = Attention(config,type='ca')
#         # self.intermediate1 = FFN(config)
#         self.intermediate2 = FFN(config)
#         # self.LN1 =nn.LayerNorm(config.hidden_size)
#         self.LN2 =nn.LayerNorm(config.hidden_size)
#         self.shrink = ConvLayer(config.hidden_size)
#     def forward(
#         self,
#         hidden_states,
#         encoder_hidden_states,
#         attention_mask=None,
#         encoder_attention_mask=None,
#         output_attentions=True,
#         last_layer=False,
#     ):
#         # self_attention_outputs = self.attention(
#         #     hidden_states,
#         #     attention_mask,
#         #     head_mask=None,
#         #     output_attentions=output_attentions,
#         #     past_key_value=None,
#         # )
#         # attention_output = self.intermediate1(self_attention_outputs)
#         # attention_output = self.LN1(hidden_states + attention_output)
#         # if decoder, the last output is tuple of self-attn cache
#         # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#
#         # cross_attn_present_key_value = None
#         cross_attention_outputs = self.crossattention(
#             hidden_states,
#             # attention_output,
#             attention_mask,
#             None,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             None,
#             output_attentions,
#         )
#         intermediate_output = self.intermediate2(cross_attention_outputs)
#         attention_output = self.LN2(intermediate_output+cross_attention_outputs)
#         if last_layer:
#             attention_output = self.shrink(attention_output)
#         return attention_output
class Attention(nn.Module):
    def __init__(self, config,type):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate_activation = nn.Sigmoid()
        self.res_type=type
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            mode="factual"  # ★ 新增参数
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None


        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # if mode == "counterfactual":  # ★ 插入反事实扰动
        #     cf_ops = CounterfactualOps(method="mask")  # 你可以配置 random/shuffle/reversed/mask
        #     attention_scores = cf_ops(attention_scores)

        if mode == "counterfactual":
            if not hasattr(self, "cf_generator"):
                from utils.domain_cf_generator import DomainCFGenerator
                self.cf_generator = DomainCFGenerator(config=self.config)  # 初始化 FlowAuto

            # 这里假设 factual=Audio(0)，counterfactual=Visual(1)
            attention_scores = self.cf_generator.generate_cf(
                attention_scores, factual_d=0, counter_d=1
            )

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        dense_output = self.dense(outputs[0])
        dense_output = self.dropout(dense_output)
        gate = self.gate_activation(self.gate_linear(hidden_states))
        gated_output = dense_output * gate
        if self.res_type=='ca':
            hidden_states = self.LayerNorm(gated_output + encoder_hidden_states)
        elif self.res_type == 'sa':
            hidden_states = self.LayerNorm(hidden_states+context_layer)

        return hidden_states

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ELU()
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states):
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.norm2 = nn.LayerNorm(c_in)
        self.activation = nn.ELU()
        # self.adaptive_pool = nn.AdaptiveMaxPool1d(dis_len)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.avgPool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # x = self.norm2(x1+x2)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        return x
#class F_S_Decoder(nn.Module):
#    def __init__(self,in_dim=64):
#        super().__init__()
        #self.args =args
#        self.f_proj_to192 = nn.Linear(in_dim, 192)
        # self.LN2 = nn.LayerNorm(args.d_prjh)
        # self.query = nn.Linear(args.d_prjh, args.d_prjh)
        # self.key = nn.Linear(args.d_prjh, args.d_prjh)
        # self.value = nn.Linear(args.d_prjh, args.d_prjh)
        # self.dropout = nn.Dropout(args.dropout_r)
        # self.correlation_conv = nn.Sequential(
        #     nn.Conv2d(1, 64, 3, stride=1, padding=1),
        #     nn.Conv2d(64, 1, 3, stride=1, padding=1),
        #     nn.ReLU()
        # )
    # def infoNCE_loss(self,x, x_pred):
    #     # normalize to unit sphere
    #     x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
    #     x = x / x.norm(dim=1, keepdim=True)
    #     pos = torch.sum(x * x_pred, dim=-1)  # bs
    #     neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
    #     nce = -(pos - neg).mean()
    #     return nce
    #
    #     return loss
#    def forward(
#            self,
#            F_feat,
#            S_feat,
#    ):
        # query_layer=self.query(F_feat)
        # key_layer = self.key(S_feat)
        # value_layer = self.value(S_feat)
#        F_feat_proj = self.f_proj_to192(F_feat)
#        attention_map = torch.matmul(F_feat_proj, S_feat.transpose(-1, -2))
        # attention_map = self.correlation_conv(attention_map.unsqueeze(1)).squeeze()
#        vision_c, text_c = attention_map.size(1), attention_map.size(2)

#        vision_attention, text_attention = torch.sum(attention_map, dim=2) / text_c, torch.sum(attention_map,dim=1) / vision_c
#        vision_attention, text_attention = torch.sigmoid(vision_attention), torch.sigmoid(text_attention)
#        F_aug = vision_attention.unsqueeze(-1) * F_feat#概率乘上去，计算保留信息量,衡量对齐概率分布
        # aligned_text_embeddings = text_attention.unsqueeze(-1) * S_feat
        # attention_scores = attention_map / math.sqrt(self.args.d_prjh)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dropout(attention_probs)
        # context_layer = torch.matmul(attention_probs, value_layer)
        # attention_output = self.LN2(F_feat + context_layer)

#        return F_aug

class F_S_Decoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    def forward(self, student_feat, teacher_feat):
        """
        student_feat: [B, L1, D]  -> 待对齐的特征
        teacher_feat: [B, L2, D]  -> 指导特征（teacher）
        return: [B, L1, D]        -> 加权后的学生特征
        """
        # 计算注意力相似度
        attention_map = torch.matmul(student_feat, teacher_feat.transpose(-1, -2))  # [B, L1, L2]

        vision_c, text_c = attention_map.size(1), attention_map.size(2)
        vision_attention = torch.sum(attention_map, dim=2) / text_c   # [B, L1]
        text_attention = torch.sum(attention_map, dim=1) / vision_c   # [B, L2]

        vision_attention = torch.sigmoid(vision_attention)
        text_attention = torch.sigmoid(text_attention)

        # 加权学生特征
        student_aug = vision_attention.unsqueeze(-1) * student_feat   # [B, L1, D]

        return student_aug

