import torch
import torch.nn as nn


class DreamRotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: str | None = None,
    ):
        """Rotary Embeddingを実装する。

        Args:
            dim (int): 次元数
            max_position_embeddings (int, optional): 最大の位置. Defaults to 2048.
            base (int, optional): ベース. Defaults to 10000.
            device (str | None, optional): デバイス. Defaults to None.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 周波数シータを計算、位置エンコーディングの周期性を決定する
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(start=0, end=self.dim, step=2).float().to(device=device)
                / self.dim
            )
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistent=False)
        # 事前にcosとsinの値を計算しておく
        self._set_cos_sin_cache(self.max_position_embeddings, device, self.dtype)

    def _set_cos_sin_cache(self, seq_len: int, device: str, dtype: torch.dtype) -> None:
        """cosとsinの値をキャッシュとして保存する

        Args:
            seq_len (int): キャッシュする最大の位置
            device (str): デバイス
            dtype (torch.dtype): データ型
        Returns:
            None
        >>>Examples:
        >>>>>> _set_cos_sin_cache(1024, "cuda", torch.float32)
        >>>>>> self.max_seq_len_cached
        1024
        >>>>>> self.cos_cached.shape
        torch.Size([1024, 8192])
        >>>>>> self.sin_cached.shape
        torch.Size([1024, 8192])
        >>>>>> self.cos_cached[0, :10]
        tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000], device='cuda:0', dtype=torch.float32)
        >>>>>> self.sin_cached[0, :10]
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], device='cuda:0', dtype=torch.float32)
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            start=self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        # 各位置(t)と各周波数(inv_freq)を掛け合わせ、位置ごとの角度を計算する
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 2次元のテンソルを結合して[seq_len, dim]の形にする
        emb = torch.cat((freqs, freqs), dim=-1)

        # cosとsinの値をキャッシュとして保存する
        self.register_buffer(
            name="cos_cached", tensor=emb.cos().to(dtype), persistent=False
        )
        self.register_buffer(
            name="sin_cached", tensor=emb.sin().to(dtype), persistent=False
        )

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cosとsinの値を返す

        Args:
            x (torch.Tensor): 入力テンソル
            seq_len (int | None, optional): キャッシュする最大の位置. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: cosとsinの値
        >>>Examples:
        >>>>>> x = torch.randn(1024, 8192)
        >>>>>> seq_len = 1024
        >>>>>> cos, sin = DreamRotaryEmbedding(dim=8192)(x, seq_len)
        >>>>>> cos.shape
        torch.Size([1024, 8192])
        >>>>>> sin.shape
        torch.Size([1024, 8192])
        >>>>>> cos[0, :10]
        tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000], device='cuda:0', dtype=torch.float32)
        >>>>>> sin[0, :10]
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], device='cuda:0', dtype=torch.float32)
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        # キャッシュされたcosとsinの値を返す
        return (
            self.cos_cached[:seq_len, :],
            self.sin_cached[:seq_len, :],
        )
