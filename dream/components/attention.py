import torch
import torch.nn as nn

from dream.config import DreamConfig

from .embedding import DreamRotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """半分に回転する

    Args:
        x (torch.Tensor): 入力テンソル

    Returns:
        torch.Tensor: 回転後のテンソル
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPEを適用して位置情報を埋め込む

    Args:
        q (torch.Tensor): Query
        k (torch.Tensor): Key
        cos (torch.Tensor): cos
        sin (torch.Tensor): sin
        position_ids (torch.Tensor): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 埋め込み後のQueryとKey
    """
    cos = cos[position_ids].unsqueeze(dim=1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(dim=1)  # [batch_size, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DreamAttention(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True

        # Query, Key, Claue, Outputのための線形層を定義
        # 本当はmulti headだが、まとめて計算するために1つの線形層で定義
        self.q_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            bias=False,
        )
        # 位置埋め込みのためにRoPEを定義
        self.rotary_emb = DreamRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None
    ]:
        """Attentionを実行する。

        Args:
            hidden_states (torch.Tensor): 入力テンソル
            attention_mask (torch.Tensor | None, optional): 注意マスク. Defaults to None.
            position_ids (torch.LongTensor | None, optional): 位置ID. Defaults to None.
            past_key_value (tuple[torch.Tensor, torch.Tensor] | None, optional): 過去のキーと値. Defaults to None.
            use_cache (bool, optional): キャッシュを使用するかどうか. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]: 出力テンソル
        """
        batch_size, q_len, _ = hidden_states.size()
        # 1. 入力テンソルを線形層に通して、Query, Key, Valueを取得
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        value_states: torch.Tensor = self.v_proj(hidden_states)

        # 2. Q, K, Vをマルチヘッドの形に変形
        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 3. RoPEを適用して位置情報を埋め込む
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )
        #  4. （オプション） KVキャッシュの処理
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # use_cacheを使用している場合は、過去のキーと値を使用
        past_key_value = (key_states, value_states) if use_cache else None

        # 5. Scaled Dot-Product Attentionの計算
        # QとKの内積を計算し、head_dimの平方根でスケーリング
        attn_weights = torch.matmul(
            input=query_states, other=key_states.transpose(2, 3)
        ) / (self.head_dim**0.5)

        # attention_maskを適用して
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # SoftmaxでAttentionスコアを確立に変換
        attn_output = nn.functional.softmax(
            input=attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # AttentionスコアとVを掛け合わせてコンテキストベクトルを計算する
        attn_output = torch.matmul(input=attn_weights, other=value_states)

        # 6. 出力整形
        attn_output = attn_output.transpose(dim0=1, dim1=2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)

        # 7. 出力層に通して、最終的な出力を計算
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
