import torch
import torch.nn as nn
from utils.flow_auto import FlowAuto

class DomainCFGenerator(nn.Module):
    """
    用于领域反事实注意力生成的封装模块。
    内部包含 FlowAuto，可执行双向映射：
      factual_x -> z -> counterfactual_x
    """
    def __init__(self, config):
        super().__init__()
        self.flowauto = FlowAuto(config)

    @torch.no_grad()
    def debug_print(self, x, name=""):
        print(f"[DomainCFGenerator] {name} shape: {tuple(x.shape)}")

    def generate_cf(self, factual_x, factual_d, counter_d):
        """
        factual_x: factual attention (B, L, D)
        factual_d: 源领域 id（如 audio）
        counter_d: 目标领域 id（如 visual）
        return: 反事实注意力分布 cf_x
        """
        device = factual_x.device

        if isinstance(factual_d, int):
            factual_d = torch.tensor([factual_d], device=device)
        if isinstance(counter_d, int):
            counter_d = torch.tensor([counter_d], device=device)

        z = self.flowauto.inverse(factual_x, factual_d)
        cf_x = self.flowauto.forward(z, counter_d)

        # 调试信息（可选）
        # self.debug_print(factual_x, "factual_x")
        # self.debug_print(z, "latent_z")
        # self.debug_print(cf_x, "counterfactual_x")

        return cf_x
