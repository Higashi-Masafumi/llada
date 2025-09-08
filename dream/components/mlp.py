import torch
import torch.nn as nn

from dream.config import DreamConfig


class DreamMLP(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        # 隠れ層から中間層への拡張
        self.gate_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        )
        # 中間層から隠れ層への縮小
        self.down_proj = nn.Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
        )
        # ゲート機構のための層
        self.up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        )
        # 活性化関数
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated MLP: 二つの線形層の出力を要素ごとに掛け合わせる
        return self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))
