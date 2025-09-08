import torch
import torch.distributions as dists
from torch.nn import functional as F


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p (Nucleus) サンプリングのためにロジットをフィルタリングする。"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = torch.finfo(logits.dtype).min
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k サンプリングのためにロジットをフィルタリングする。"""
    top_k = min(top_k, logits.size(-1))
    if top_k == 0:
        return logits

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = torch.finfo(logits.dtype).min
    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与えられたロジットからトークンをサンプリングし、その確信度（選ばれたトークンの確率）を返す。"""
    # 1. ロジットの加工 (temperature, top_k, top_p)
    if temperature > 0:
        logits = logits / temperature
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    # 2. 確率分布の計算とサンプリング
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        sampled_tokens = dists.Categorical(probs=probs).sample()
    else:
        _, sampled_tokens = probs.max(dim=-1)

    # 3. 確信度（サンプリングされたトークンの確率）を計算
    confidence = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    return confidence, sampled_tokens
