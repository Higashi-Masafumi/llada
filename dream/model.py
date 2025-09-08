import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from dream.components.decoder import DreamDecoder
from dream.components.rms_norm import DreamRMSNorm

from .config import DreamConfig


class DreamPretrainedModel(PreTrainedModel):
    # 設定クラスを定義
    config_class = DreamConfig
    # モデルの接頭辞を定義
    base_model_prefix = "model"
    # 勾配チェックポイントをサポート
    supports_gradient_checkpointing = True
    # 分割されないモジュールを定義
    _no_split_modules = ["DreamDecoderLayer"]
    # デバイス配置をスキップするキーを定義
    _skip_keys_device_placement = "past_key_values"
    # Flash Attention 2をサポート
    _supports_flash_attn_2 = True
    # SDPAをサポート
    _supports_sdpa = True
    # キャッシュクラスをサポート
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        # 重みの初期化手法を定義
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # 線形層の重みを正規分布で初期化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # バイアスを0に初期化
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 埋め込み層の重みを正規分布で初期化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # パディングインデックスの重みを0に初期化
                # Note: パディングインデックスとは、バッチ処理の際に使用されるパディングのインデックス
                module.weight.data[module.padding_idx].zero_()


class DreamModel(DreamPretrainedModel):
    def __init__(self, config: DreamConfig):
        super().__init__(config=config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # トークンIDをベクトルに変換する埋め込み層
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        # 複数のデコーダー層をリストとして保持する
        self.layers = nn.ModuleList(
            [DreamDecoder(config=config) for _ in range(config.num_hidden_layers)]
        )
        # 最終出力前の正規化層
        self.norm = DreamRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutputWithPast:
        """DreamModelを実行する。

        Args:
            input_ids (torch.Tensor): 入力トークンID
            attention_mask (torch.Tensor | None, optional): 注意マスク. Defaults to None.
            position_ids (torch.LongTensor | None, optional): 位置ID. Defaults to None.
            past_key_values (list[torch.Tensor] | None, optional): 過去のキーと値. Defaults to None.
            use_cache (bool | None, optional): キャッシュを使用するかどうか. Defaults to None.
            output_attentions (bool | None, optional): 出力注意力. Defaults to None.
            output_hidden_states (bool | None, optional): 出力隠れ状態. Defaults to None.
            return_dict (bool | None, optional): 出力辞書. Defaults to None.

        Returns:
            BaseModelOutputWithPast: 出力テンソル
        """
        # 1. 入力をベクトルに変換
        hidden_states = self.embed_tokens(input_ids)

        # 2. デコーダー層を順番に適用
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # 一層分の処理を実行
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # 出力を更新
            hidden_states = layer_outputs[0]

            # キャッシュを使用している場合は、過去のkeyとvalueを更新
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        # 3. 最終正規化
        hidden_states = self.norm(hidden_states)

        # 4. 出力を返す
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=None,
            attentions=None,
        )


class DreamForCausalLM(DreamPretrainedModel):
    """CasualLM用のモデル"""

    # 重みを共有するキーを定義
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DreamConfig):
        super().__init__(config=config)
        self.model = DreamModel(config=config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        """DreamForCausalLMを実行する。

        Args:
            input_ids (torch.Tensor): 入力トークンID
            attention_mask (torch.Tensor | None, optional): 注意マスク. Defaults to None.
            position_ids (torch.LongTensor | None, optional): 位置ID. Defaults to None.
            past_key_values (list[torch.Tensor] | None, optional): 過去のキーと値. Defaults to None.
            labels (torch.LongTensor | None, optional): ラベル. Defaults to None.
            use_cache (bool | None, optional): キャッシュを使用するかどうか. Defaults to None.
            return_dict (bool | None, optional): 出力辞書. Defaults to None.

        Returns:
            CausalLMOutputWithPast: 出力テンソル
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # 2. LMヘッドで各トークンの次の単語の確率分布（logits）を計算
        # 別にここで予測をしているわけではないので、データのラベルと比較するためにfloatに変換する
        logits: torch.Tensor = self.lm_head(hidden_states)
        logits = logits.float()

        # 3. 損失を計算
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits_for_loss = logits.view(-1, self.config.vocab_size)
            labels_for_loss = labels.view(-1)
            loss = loss_fct(logits_for_loss, labels_for_loss)

        # 4. 出力を返す
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
