# modules/debias.py功能：加载 .npy 聚类中心，对输入文本特征做投影与偏差分解。
#文本去偏模块：使用 text dictionary (kmeans_mosi-200_text.npy) 将文本 embedding 分解为 content 和 bias，content 用于 cross-attention。
#对应 AtCAF 公式 (h_t = h_t^content + h_t^bias)。
#修改点：新增 utils/debias.py，在 MSA.forward 中调用。
import torch
import torch.nn as nn
import numpy as np

class TextDebias(nn.Module):
    def __init__(self, dict_path, hidden_size):
        super().__init__()
        cluster_dict = np.load(dict_path)   # 例如 kmeans_mosi-200_text.npy
        self.cluster_centers = torch.tensor(cluster_dict, dtype=torch.float32)
        self.proj = nn.Linear(hidden_size, self.cluster_centers.shape[1])

    def forward(self, text_feat):
        """
        text_feat: (B, L, D)
        return: debiased_text, bias_component
        """
        #print(">> Debias input:", text_feat.shape)  # 输入 lang_emb
        #print(">> Cluster centers:", self.cluster_centers.shape)  # 你的 npy 文件 shape
        proj_feat = self.proj(text_feat)                  # (B, L, n_clusters)
        sim = torch.matmul(proj_feat, self.cluster_centers.T)
        bias = torch.matmul(sim, self.cluster_centers)    # 重构 bias
        debiased_text = text_feat - bias
        #print(">> Debiased text:", debiased_text.shape)
        return debiased_text, bias
