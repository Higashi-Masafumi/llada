import torch
import torch.nn as nn

from dream.config import DreamConfig

from .attention import DreamAttention
from .mlp import DreamMLP
from .rms_norm import DreamRMSNorm


class DreamDecoder(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DreamAttention(config=config)
        self.mlp = DreamMLP(config=config)
        self.input_layernorm = DreamRMSNorm(
            hidden_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DreamRMSNorm(
            hidden_size=self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Decoderを実行する。

        Args:
            hidden_states (torch.Tensor): 入力テンソル
            attention_mask (torch.Tensor | None, optional): 注意マスク. Defaults to None.
            position_ids (torch.LongTensor | None, optional): 位置ID. Defaults to None.
            past_key_value (tuple[torch.Tensor, torch.Tensor] | None, optional): 過去のキーと値. Defaults to None.
            use_cache (bool, optional): キャッシュを使用するかどうか. Defaults to False.
        """
        # 1. 残差結合のための入力を保持
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # 2. Self-Attention
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # 4. 一つ目の残差結合
        hidden_states = residual + hidden_states

        # 5. 二つ目の残差結合のための入力を保持する
        residual = hidden_states

        # 6. MLP層への入力正規化
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 7. MLP層の計算
        hidden_states = self.mlp(hidden_states)

        # 8. 二つ目の残差結合
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
