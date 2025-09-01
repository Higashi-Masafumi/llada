from typing import Literal
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from utils import log_print


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise to logits for improved sampling.

    Args:
        logits (torch.Tensor): モデルが出力した各単語のスコア（確率分布に変換する前）
        temperature (float): サンプリングのランダム性を制御するパラメータ。０の場合はノイズを加えない。

    Returns:
        torch.Tensor: The logits with added Gumbel noise.

    Note:
        temperatureは言語モデルのランダム性、言ってしまえば創造性を制御するパラメータです。
        ここではtemperatureに応じてガンベルノイズを付与します、これによりlogitsの中でスコアの最大の単語が
        100%で選ばれることがなくなり、より多様な単語が選ばれるようになります。
    """

    if temperature <= 0:
        return logits

    # 高精度な計算のためにlogitsをfloat64に変換
    logits = logits.to(torch.float64)
    log_print(log_name="input logits", content=f"{logits}")
    # 元のlogitsと同じ形状のランダムなノイズを生成
    noise = torch.rand_like(logits, dtype=torch.float64)
    # Gumbelノイズを計算(temperatureによって変化させる)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Calculate the number of tokens to transfer based on the mask index and steps.

    Args:
        mask_index (torch.Tensor): マスクされている位置を示すブール値のテンソル
        steps (int): 生成プロセスの総ステップ数

    Returns:
        torch.Tensor: 各ステップで確定させるべきトークン数を示すテンソル
    """
    log_print(log_name="mask_index", content=f"{mask_index}")
    # マスクされているトークンの総数を計算する
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # 総マスク数をステップ数で割って、各ステップの基本的な確定数（デノイズするトークン数）を計算する
    base = mask_num // steps
    # 余りを計算する
    remainder = mask_num % steps

    # 各ステップの確定数をbaseで初期化する
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    # バッチ内の各sequenceに対して余りの確定数を分配する
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model: AutoModel,
    device: str,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temerature: int = 0,
    cfg_scale: int = 0,
    remasking: Literal["low_confidence", "random"] = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    """Generate text using the model.

    Args:
        model (AutoModel): The model to use for generation.
        prompt (str): The input prompt for the model.
        cfg_scale (int): The classifier-free guidance scale.
        steps (int, optional): The number of steps for generation. Defaults to 128.
        gen_length (int, optional): The length of the generated text. Defaults to 128.
        block_length (int, optional): The length of each block for generation. Defaults to 128.
        temerature (int, optional): The temperature for sampling. Defaults to 0.
        remasking (Literal["low_confidence", "random"], optional): The remasking strategy. Defaults to "low_confidence".
        mask_id (int, optional): The ID of the mask token. Defaults to 126336.

    Returns:
        torch.Tensor: The generated text.
    """
    # 初期化：プロンプトの後ろに生成長文のマスクトークンを連結したsequenceを作成
    # 最初はすべての生成部分がマスクされている状態（mask_idで埋められている）
    x = torch.full(size=(1, prompt.shape[1] + gen_length), fill_value=mask_id, dtype=torch.long).to(device=device)
    # プロンプト部分はそのままコピーしてmask_idを更新する
    x[:, :prompt.shape[1]] = prompt.clone()

    #　プロンプト部分の位置を記録（CFGや再マスキングで変更が行われないように）
    prompt_index = (x != mask_id)
    ...



