import torch
import torch.nn as nn


class DreamRMSNorm(nn.Module):
    """
    RMSNorm(Root Mean Square Layer Normalization)を実装する。
    標準的なLayerNormよりも計算効率が良いとされる正規化手法
    Note: そんなこともなさそう　https://zenn.dev/bilzard/articles/why-rmsnorm-is-used-in-recent-llms
    Pytorch Document: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 入力テンソルのデータ型をfloat32に変換して計算精度を確保する
        input_type = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # RMSNormの計算
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            input=variance + self.variance_epsilon
        )

        # 成果されたテンソルに重みを掛け合わせ、元のデータ型に戻す
        return (self.weight * hidden_states).to(input_type)
