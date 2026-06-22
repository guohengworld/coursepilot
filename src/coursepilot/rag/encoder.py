"""
BGE-M3 编码器 —— 一次 forward 同时输出 dense + learned sparse 向量

用法：
    encoder = Encoder()
    vecs = encoder.encode(["文本一", "文本二"])
    # vecs[0] → {"dense": list[float], "sparse": dict[int, float]}
"""

from __future__ import annotations

import logging

from coursepilot.config import settings
from coursepilot.rag.config import config

logger = logging.getLogger(__name__)

_encoder_instance = None


def _load_model():
    """惰性加载 BGE-M3 模型（CPU 推理）"""
    try:
        from FlagEmbedding import BGEM3FlagModel

        logger.info("加载 BGE-M3 模型：%s", settings.embedding_model_path)
        return BGEM3FlagModel(
            settings.embedding_model_path,
            use_fp16=False,
            device="cpu",
        )
    except Exception:
        logger.warning("加载 BGE-M3 模型失败", exc_info=True)
        return None


class Encoder:
    """
    BGE-M3 统一编码器，返回 dense + sparse 向量

    全局单例（惰性加载），首次实例化时加载模型到内存
    """

    def __init__(self) -> None:
        global _encoder_instance
        if _encoder_instance is None:
            _encoder_instance = _load_model()
        self._model = _encoder_instance

    def encode(self, texts: list[str]) -> list[dict]:
        """
        编码文本列表

        返回：[{"dense": [1024 floats], "sparse": {token_id: weight}}, ...]
        """
        if not texts:
            return []

        output = self._model.encode(
            texts,
            batch_size=config.batch_size,
            return_dense=True,
            return_sparse=True,
        )

        results = []
        for i in range(len(texts)):
            dense = output["dense_vecs"][i]
            if hasattr(dense, "tolist"):
                dense = dense.tolist()

            sparse_weights = output.get("lexical_weights", [{}] * len(texts))[i]
            sparse = {int(k): float(v) for k, v in sparse_weights.items()} if sparse_weights else {}

            results.append({"dense": dense, "sparse": sparse})

        return results

    def encode_queries(self, queries: list[str]) -> list[dict]:
        """查询编码（与文档编码语义一致，复用同一方法）"""
        return self.encode(queries)

    def encode_query(self, query: str) -> dict:
        """单条查询编码快捷方法"""
        return self.encode_queries([query])[0]

    @property
    def dim(self) -> int:
        return config.dim
