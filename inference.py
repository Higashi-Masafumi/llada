from typing import Literal
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from utils import log_print
from typing import Any


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
    model: Any,  # Automodel.from_pretrainedで読み込んだモデル
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
    x = torch.full(
        size=(1, prompt.shape[1] + gen_length), fill_value=mask_id, dtype=torch.long
    ).to(device=device)
    # プロンプト部分はそのままコピーしてmask_idを更新する
    x[:, : prompt.shape[1]] = prompt.clone()

    # 　プロンプト部分の位置を記録（CFGや再マスキングで変更が行われないように）
    prompt_index = x != mask_id
    assert (
        gen_length % block_length == 0
    ), "生成長はブロック長で割り切れる必要があります。"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "ステップ数はブロック数で割り切れる必要があります。"
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        # ブロック内のマスク位置を取得する
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        # 各ステップで確定させるトークン数を計算する
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index, steps=steps
        )
        # デノイズ（復元）ループ
        for i in range(steps):
            # 現在のシーケンス全体でマスクされている位置を取得
            mask_index = x == mask_id

            # Classifier-Free Guidance(CFG)を使用する場合
            if cfg_scale > 0:
                # プロンプトがマスクされていない「条件付き」入力を作成
                un_x = x.clone()
                # プロンプト部分をマスクした「条件なし」入力を作成
                un_x[prompt_index] = mask_id
                # 「条件あり」と「条件なし」の入力を結合してモデルに一度に入力
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                # 出力を分割して「条件あり」と「条件なし」のlogitsを取得
                logits, un_logits = torch.chunk(input=logits, chunks=2, dim=0)
                # CFGの式に従って、プロンプトの方向性を強めたlogitsを計算する
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # CFGを利用しない場合は通常通りモデルで予測
                logits = model(x).logits
            # サンプリング：次のトークンを予測
            logits_with_noise = add_gumbel_noise(
                logits=logits,
                temperature=temerature,
            )
            # ノイズ付きスコアが最も高いトークンを予測結果とする
            x0 = torch.argmax(input=logits_with_noise, dim=1)

            # 再マスキング戦略
            if remasking == "low_confidence":
                # softmaxでlogitsを確率に変換する
                p = F.softmax(input=logits, dim=1)
                # 予測されたトークン(x0)が持つ確率（自信度）を取得する
                x0_p = torch.squeeze(
                    input=torch.gather(
                        input=p,
                        dim=-1,
                        index=torch.unsqueeze(input=x0, dim=-1),
                    ),
                    dim=-1,
                )
            elif remasking == "random":
                # ランダムな値が自信度として使用される
                x0_p = torch.rand(size=(x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 現在のブロック以降の自信度は計算対象外とする
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            # sequenceの更新
            # マスクされている位置は予測トークン(x0)で、それ以外は元のトークン(x)で埋める
            x0 = torch.where(condition=mask_index, input=x0, other=x)
            # マスク位置の自信度をconfidenceとして保持する
            confidence = torch.where(condition=mask_index, input=x0_p, other=-np.inf)

            # 確定させるトークンを選択するためのインデックスを作成
            transfer_index = torch.zeros_like(
                input=x0, dtype=torch.bool, device=x0.device
            )
            # バッチ内の各sequenceで処理（バッチごとの処理）
            for j in range(confidence.shape[0]):
                # 自信度が上位k個のトークンを選択する（k=このステップで確定させるトークン数）
                _, select_index = torch.topk(
                    input=confidence[j],
                    k=int(
                        num_transfer_tokens[j, i]
                    ),  # 各ステップで確定させるトークン数
                )
                # 選択された位置をTrueに設定する
                transfer_index[j, select_index] = True
            # 選択された位置のトークンを予測されたトークンで更新する
            x[transfer_index] = x0[transfer_index]

    return x
