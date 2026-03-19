# utils/flow_auto.py
# ---------------------------------------------------------
# 轻量版 FlowAuto 模块 —— 领域反事实注意力生成器
# ---------------------------------------------------------
# 作者: ChatGPT（为 MSATASE2 项目适配）
# 功能:
#   1. 模拟可逆流体的 "inverse/forward" 双向映射。
#   2. 在潜空间中去除领域特征并生成跨域反事实注意力。
#   3. 保留原论文的核心思想：潜变量 + 域条件映射。
# ---------------------------------------------------------

import torch
import torch.nn as nn

class FlowAuto(nn.Module):
    """
    简化版 FlowAuto，用于领域反事实注意力映射。

    主要接口：
        - inverse(x, domain_id): 从 factual 注意力生成潜变量 z
        - forward(z, domain_id): 从潜变量生成目标领域反事实注意力

    参数说明:
        config.d_prjh: attention 隐层维度
        config.num_domains: 领域数 (通常为 3: text/audio/visual)
        config.domain_dim: 每个领域嵌入维度 (默认为 8)
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'd_prjh', 64)
        self.num_domains = getattr(config, 'num_domains', 3)
        self.domain_dim = getattr(config, 'domain_dim', 8)

        # 可学习的领域嵌入 (Text=0, Audio=1, Visual=2)
        self.domain_embed = nn.Embedding(self.num_domains, self.domain_dim)

        # 编码器: 近似 inverse 流 (去领域 -> 潜空间)
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2)
        )

        # 解码器: 近似 forward 流 (潜空间 + 域嵌入 -> 注意力)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 + self.domain_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

    # -------------------------------
    # 逆变换: factual_x -> z
    # -------------------------------
    def inverse(self, x, domain_id):
        """
        将事实注意力映射到潜空间 (z)。
        x: [B, L, D] factual attention
        domain_id: int or tensor
        """
        if x.dim() != 3:
            raise ValueError(f"[FlowAuto.inverse] 输入维度错误: 期望 (B, L, D)，实际 {x.shape}")
        z = self.encoder(x)
        return z

    # -------------------------------
    # 正变换: z -> counterfactual_x
    # -------------------------------
    def forward(self, z, domain_id):
        """
        将潜变量投射到目标领域，生成反事实注意力。
        z: [B, L, D_z]
        domain_id: int or tensor
        """
        if z.dim() != 3:
            raise ValueError(f"[FlowAuto.forward] 输入维度错误: 期望 (B, L, D_z)，实际 {z.shape}")

        if isinstance(domain_id, int):
            domain_id = torch.tensor([domain_id], device=z.device)
        if domain_id.dim() == 1:
            domain_id = domain_id.unsqueeze(0).expand(z.size(0), -1)

        domain_vec = self.domain_embed(domain_id.squeeze(1))  # [B, domain_dim]
        domain_vec = domain_vec.unsqueeze(1).expand(z.size(0), z.size(1), -1)  # [B, L, domain_dim]

        z_cat = torch.cat([z, domain_vec], dim=-1)  # [B, L, D_z + domain_dim]
        x_cf = self.decoder(z_cat)
        return x_cf

# ---------------------------------------------------------
# ✅ 示例调用（集成验证）
# ---------------------------------------------------------
if __name__ == "__main__":
    class DummyConfig:
        d_prjh = 64
        num_domains = 3
        domain_dim = 8

    config = DummyConfig()
    flow = FlowAuto(config)

    B, L, D = 4, 10, 64
    factual_x = torch.randn(B, L, D)

    # factual 域=Audio(1)，counter 域=Visual(2)
    z = flow.inverse(factual_x, domain_id=1)
    cf_x = flow.forward(z, domain_id=2)

    print("输入 factual_x:", factual_x.shape)
    print("潜空间 z:", z.shape)
    print("输出 counterfactual_x:", cf_x.shape)
