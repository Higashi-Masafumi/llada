"""Dreamモデルの設定用クラス"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DreamConfig(PretrainedConfig):
    """
    これはDreamモデルの設定クラスです。
    `~transformers.PretrainedConfig`を継承しており、Hugging Face Transformersライブラリの
    すべての設定クラスで共通のメソッドを持っています。

    このクラスは、特定のチェックポイント（例: `Dream-org/Dream-v0-Instruct-7B`）と
    一緒に保存されている`config.json`ファイルを読み込み、モデルのアーキテクチャを
    定義するために使用されます。
    """

    # モデルの識別に使われる名前
    model_type = "Dream"

    # テキスト生成時に推論（inference）で無視されるべきキーのリスト
    # KVキャッシュは生成ステップごとに更新されるため、保存時には不要
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # --- モデルのサイズと語彙に関するパラメータ ---
        vocab_size: int = 151936,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        # --- 活性化関数と位置埋め込みに関するパラメータ ---
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        # --- 初期化と正規化に関するパラメータ ---
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        # --- KVキャッシュとトークンIDに関するパラメータ ---
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        mask_token_id: int = 151666,
        # --- AttentionとRoPEに関するパラメータ ---
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        max_window_layers: int = 28,
        # --- 親クラスのコンストラクタに渡すためのキーワード引数 ---
        **kwargs,
    ):
        """
        DreamConfigのコンストラクタ

        Args:
            vocab_size (int, optional, defaults to 151936):
                モデルが認識できる単語（トークン）の総数。
            hidden_size (int, optional, defaults to 3584):
                埋め込み層や各Transformerブロックの出力ベクトルの次元数。
            intermediate_size (int, optional, defaults to 18944):
                Feed-Forwardネットワークの中間層の次元数。通常はhidden_sizeの数倍。
            num_hidden_layers (int, optional, defaults to 28):
                Transformerブロック（デコーダー層）を何層積み重ねるか。
            num_attention_heads (int, optional, defaults to 28):
                Multi-Head Attentionにおけるヘッドの数。
            num_key_value_heads (int, optional, defaults to 4):
                Grouped-Query AttentionのためのKey/Valueヘッドの数。num_attention_headsより小さい値。
            hidden_act (str, optional, defaults to "silu"):
                Feed-Forwardネットワークで使用する活性化関数。SiLU（Swish）を使用。
            max_position_embeddings (int, optional, defaults to 131072):
                モデルが処理できる最大のシーケンス長（トークン数）。
            initializer_range (float, optional, defaults to 0.02):
                重み行列を初期化する際の正規分布の標準偏差。
            rms_norm_eps (float, optional, defaults to 1e-6):
                RMSNormでゼロ除算を防ぐための小さな値（イプシロン）。
            use_cache (bool, optional, defaults to False):
                高速なテキスト生成のために、計算済みのKey/Valueの状態（KVキャッシュ）を保持するかどうか。
            pad_token_id (int | None, optional, defaults to None):
                パディングに使われるトークンのID。
            bos_token_id (int, optional, defaults to 151643):
                文章の開始を示す（BOS）トークンのID。
            eos_token_id (int, optional, defaults to 151643):
                文章の終了を示す（EOS）トークンのID。
            mask_token_id (int, optional, defaults to 151666):
                マスクされたトークンのID。
            attention_dropout (float, optional, defaults to 0.0):
                Attentionの確率に対するドロップアウト率。
            rope_theta (float, optional, defaults to 10000.0):
                回転位置埋め込み（RoPE）の計算で使われるベース値。
            rope_scaling (dict | None, optional, defaults to None):
                長いシーケンスに対応するためのRoPEのスケーリング設定。
            max_window_layers (int, optional, defaults to 28):
                Sliding Window Attentionを適用するレイヤーの最大数。
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.mask_token_id = mask_token_id
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout

        # 親クラスのコンストラクタを呼び出し、共通のパラメータを設定
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
