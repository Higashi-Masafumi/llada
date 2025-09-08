import torch
from torch.nn import functional as F
from transformers import GenerationConfig, LogitsProcessorList
from transformers.generation.utils import (
    CausalLMOutputWithPast,
    GenerateOutput,
    GenerationMixin,
)
from dream.sample_helper import sample_tokens


class DreamGenerationConfig(GenerationConfig):
    def __init__(
        self,
        steps: int = 8,
        guidance: float = 3.0,
        number_of_masks: int = 1,
        number_of_transfer_tokens: int = 1,
        max_step_token: int = 32,
        algorithm: str = "lp",
        algorithm_temperature: float = 0.0,
        output_history: bool = False,
        mask_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # ------- 拡散モデル用のカスタムパラメーター ---------
        self.steps = steps
        self.guidance = guidance
        self.number_of_masks = number_of_masks
        self.number_of_transfer_tokens = number_of_transfer_tokens
        self.max_step_token = max_step_token
        self.algorithm = algorithm
        self.algorithm_temperature = algorithm_temperature
        self.output_history = output_history
        self.mask_token_id = mask_token_id


class DreamGenerationMixin(GenerationMixin):
    @torch.no_grad()
    def diffusion_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        generation_config: DreamGenerationConfig | None = None,
        **kwargs,
    ) -> GenerateOutput:
        """拡散モデルでテキストを生成する。

        Args:
            input_ids (torch.LongTensor): 入力トークンID
            attention_mask (torch.LongTensor | None, optional): 注意マスク. Defaults to None.
            generation_config (DreamGenerationConfig | None, optional): 生成設定. Defaults to None.
            **kwargs: その他のキーワード引数

        Returns:
            GenerateOutput: 生成結果
        """
        # モデルに紐づく生成設定を取得（なければデフォルトを使用）
        config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )

        steps = config.steps
        guidance = config.guidance
        number_of_masks = config.number_of_masks
        number_of_transfer_tokens = config.number_of_transfer_tokens
        mas_step_token = config.max_step_token
        algorithm = config.algorithm
        algorithm_temperature = config.algorithm_temperature
        output_history = config.output_history

        # 特殊トークンのIDをモデル設定から取得、設定不備によるエラーを防ぐために存在チェックを行う
        mask_token_id = (
            config.mask_token_id
            if config.mask_token_id is not None
            else self.config.mask_token_id
        )
        pad_token_id = (
            config.pad_token_id
            if config.pad_token_id is not None
            else self.config.pad_token_id
        )
        eos_token_id = (
            config.eos_token_id
            if config.eos_token_id is not None
            else self.config.eos_token_id
        )

        if mask_token_id is None or pad_token_id is None or eos_token_id is None:
            raise ValueError(
                "mask_token_id, pad_token_id, eos_token_id are not set in the model configuration"
            )

        # 入力テンソルの形状を確認
        batch_size = input_ids.shape[0]
        device = self.device

        # 生成履歴を保存するリスト
        history: list[torch.Tensor] = []

        # 2. 最初のMaskトークンを付与
        masks = torch.full(
            size=(batch_size, number_of_masks),
            fill_value=mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_attention = torch.ones_like(input=masks)
        input_ids = torch.cat([input_ids, masks], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, mask_attention], dim=1)
        if output_history:
            history.append(input_ids.clone())

        # 3. デノイズ（復元）生成ループ
        for step in range(steps):
            # transformerで推論する
            outputs: CausalLMOutputWithPast = self(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits.detach()
            masked_logits = logits[:, -number_of_masks:, :]

            # ----4. 自信度の計算 ---------
            probabilities = F.softmax(input=masked_logits, dim=-1)
            confidences, _ = probabilities.max(dim=-1)

            # ガンベル分布のトリックを用いて、確率にノイズを加えることで探索的な性質を持たせる
            # log(-log(U))は標準ガンベル分布に従う。guidanceはこのノイズの強度を調整する
            noise = torch.rand_like(input=confidences)  # 0~1の一様分布からノイズを生成
            confidences = (
                confidences - guidance * (noise + 1e-5).log().neg().log()
            )  # ガンベル分布に従うノイズを加える

            end_tokens_mask = (
                input_ids[:, -number_of_masks:] == eos_token_id
            )  # 終了トークンのマスクを作成
            confidences[end_tokens_mask] = (
                torch.inf
            )  # 終了トークンの自信度を無限大にする

            if algorithm_temperature > 0:
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(input=confidences) + 1e-9) + 1e-9
                )
                confidences += gumbel_noise * algorithm_temperature

            # ----5. MASKの更新 ---------
            if algorithm == "lp":
                # 自信度が
                indices = confidences.sort(dim=-1).indices
                transfer_indices = indices[:, :number_of_transfer_tokens]
            else:
                raise NotImplementedError(algorithm)

            logits_to_transfer = torch.gather(
                input=masked_logits,
                dim=1,
                index=transfer_indices.unsqueeze(dim=-1).expand(
                    size=(-1, -1, logits.shape[-1])
                ),
            )

            # ----6. 次のトークンをサンプリング ---------
            logits_processor = self._get_logits_processor(
                generation_config=config,
                input_ids_seq_length=input_ids.shape[-1],
                encoder_input_ids=input_ids,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
            )

            processed_logits = logits_processor(input_ids=torch.empty(size=logits_to_transfer.shape[0], seq_length=1, device=device), scores=logits_to_transfer)
