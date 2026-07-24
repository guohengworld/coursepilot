"""
BGE-M3 编码器 —— 一次 forward 同时输出 dense + learned sparse 向量

用法：
    encoder = Encoder()
    vecs = encoder.encode(["文本一", "文本二"])
    # vecs[0] → {"dense": [1024 floats], "sparse": {token_id: weight}}
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from coursepilot.config import settings
from coursepilot.rag.config import config

logger = logging.getLogger(__name__)

_encoder_instance = None


def _select_device() -> str:
    """选择编码设备：优先 CUDA，否则 CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    """惰性加载 BGE-M3 模型，优先使用 GPU。"""
    print(f"[DEBUG] embedding_model_path = '{settings.embedding_model_path}'")
    print(f"[DEBUG] 目录是否存在: {Path(settings.embedding_model_path).exists()}")

    from FlagEmbedding import BGEM3FlagModel

    device = _select_device()
    use_fp16 = device.startswith("cuda")

    try:
        logger.info(
            "加载 BGE-M3 模型：%s，device=%s，fp16=%s",
            settings.embedding_model_path,
            device,
            use_fp16,
        )
        return BGEM3FlagModel(
            settings.embedding_model_path,
            use_fp16=use_fp16,
            devices=device,
        )
    except Exception as exc:
        logger.warning("BGE-M3 加载到 %s 失败 (%s)，尝试降级到 CPU", device, exc)
        if device != "cpu":
            try:
                logger.info("降级加载 BGE-M3 到 CPU")
                return BGEM3FlagModel(
                    settings.embedding_model_path,
                    use_fp16=False,
                    devices="cpu",
                )
            except Exception as exc2:
                logger.warning("BGE-M3 CPU 加载也失败 (%s)", exc2)
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

    def encode_qa_records(self, records: list[dict[str, str]]) -> list[dict]:
        """批量编码 QARecord，用于记忆召回（P4）。

        :param records: [{"query": str, "answer": str}, ...]
        :return: 与输入等长的向量列表，每个元素含 dense/sparse
        """
        texts = [f"Q: {r.get('query', '')}\nA: {r.get('answer', '')}" for r in records]
        return self.encode(texts)

    @property
    def dim(self) -> int:
        return config.dim
