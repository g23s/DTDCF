
import torch
import math
from torch import nn
from transformers.modeling_utils import prune_linear_layer


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask), dtype=torch.long)[mask]
    return heads, index


class BertSelfAttention(nn.Module):
    def __init__(self, config):
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
        self.is_decoder = getattr(config, "is_decoder", False)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

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
            query_confidence=None,
            key_value_confidence=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        if query_confidence is not None:
            mixed_query_layer = mixed_query_layer * query_confidence

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None


        if is_cross_attention:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            if key_value_confidence is not None:
                mixed_key_layer = mixed_key_layer * key_value_confidence
                mixed_value_layer = mixed_value_layer * key_value_confidence
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask


        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.gate_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate_activation = nn.Sigmoid()
    def forward(self, hidden_states, input_tensor):
        dense_output = self.dense(hidden_states)
        dense_output = self.dropout(dense_output)
        gate = self.gate_activation(self.gate_linear(hidden_states))
        gated_output = dense_output * gate
        hidden_states = self.LayerNorm(gated_output + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)#它接受原始的线性层和剪枝索引，返回剪枝后的线性层。
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_confidence=None,
        key_value_confidence=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            query_confidence,
            key_value_confidence,
        )
        if encoder_attention_mask is not None:
            attention_output = self.output(self_outputs[0], encoder_hidden_states)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0)
        self.seq_len_dim = 1
        # self.attention = BertAttention(config)
        # self.is_decoder = config.is_decoder
        # self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=True,
        query_confidence=None,
        key_value_confidence=None,

    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        #
        # self_attention_outputs = self.attention(
        #     hidden_states,
        #     attention_mask,
        #     head_mask=None,
        #     output_attentions=output_attentions,
        #     past_key_value=None,
        # )
        # attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # cross_attn_present_key_value = None
        # 1. 计算 cross-attention
        cross_attention_outputs = self.crossattention(
            hidden_states,
            # attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions=True,  # ★ 强制要求输出 attn_weights
            query_confidence=query_confidence,
            key_value_confidence=key_value_confidence,
        )
        attention_output = cross_attention_outputs[0]# ★ 原始 factual 输出
        attn_weights = cross_attention_outputs[1]
        #attn_weights 不是特征向量，而是 注意力分布 (softmax(QKᵀ))，用来表示 Query 对 Key 的注意力分配。

        attention_output_cf = None

        #print("attn_weights:", attn_weights.shape)  # [B, heads, Q_len, K_len]
        #print("encoder_hidden_states:", encoder_hidden_states.shape)  # [B, L, D]


# 更改“反事实注意力扰动策略”的位置！！！！
        if attn_weights is not None:
            perturbed_weights = self._perturb_attention(attn_weights, strategy="shuffle") #更改“反事实注意力扰动策略”的位置！！！！
            #print("perturbed_weights:", perturbed_weights.shape)
            #attention_output_cf = torch.matmul(perturbed_weights, encoder_hidden_states)
            #这样保证了：attention_output_cf 和 attention_output 都是 [B, Q_len, D]
            B, L, D = encoder_hidden_states.size()
            H = perturbed_weights.size(1)  # num_heads
            d_head = D // H

            # reshape encoder_hidden_states -> 多头
            encoder_states_heads = encoder_hidden_states.view(B, L, H, d_head).transpose(1, 2)
            # [B, H, L, d_head]

            # 扰动后的注意力应用
            attention_output_cf = torch.matmul(perturbed_weights, encoder_states_heads)
            # [B, H, Q_len, d_head]
            #print("attention_output_cf:", attention_output_cf.shape)

            # 合并 heads
            attention_output_cf = attention_output_cf.transpose(1, 2).contiguous().view(B, -1, D)
            # [B, Q_len, D]

#这里 _perturb_attention 就是你根据论文定义的 4 种扰动方法（shuffle / reverse / random / zero-mask）。
        # 3. FFN
        intermediate_output = self.intermediate(attention_output)
        attention_output = self.output(intermediate_output,attention_output)
        # att = cross_attention_outputs[1]  # add cross attentions if we output attention weights

        if attention_output_cf is not None:
            intermediate_output_cf = self.intermediate(attention_output_cf)
            attention_output_cf = self.output(intermediate_output_cf, attention_output_cf)

        # return attention_output,att
        return attention_output, attention_output_cf

    def _perturb_attention(self, attn_weights, strategy="shuffle"):
        """
        对注意力矩阵进行反事实扰动
        attn_weights: (B, heads, Q_len, K_len)
        """

        if strategy == "shuffle":
            # 在最后一维对注意力分布打乱
            idx = torch.randperm(attn_weights.size(-1), device=attn_weights.device)
            return attn_weights.index_select(-1, idx)

        elif strategy == "reverse":
            # 在最后一维反转
            return torch.flip(attn_weights, dims=[-1])

        elif strategy == "random":
            # 替换为随机分布（保持非负 & 归一化）
            rand = torch.rand_like(attn_weights)
            return rand / rand.sum(dim=-1, keepdim=True)

        elif strategy == "zero-mask":
            # 将一半位置置零，再归一化
            mask = torch.zeros_like(attn_weights)
            half = attn_weights.size(-1) // 2
            mask[..., :half] = 1
            masked = attn_weights * mask
            return masked / (masked.sum(dim=-1, keepdim=True) + 1e-6)

        else:
            raise ValueError(f"Unknown counterfactual strategy: {strategy}")
