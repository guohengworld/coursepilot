"""
Milvus Lite 向量存储 —— CRUD + 混合检索 + 内置 RRF

用法：
    store = VectorStore()
    store.create_collection()       # 幂等
    ids = store.insert(vectors)     # 批量插入
    results = store.hybrid_search(  # dense + sparse 混合检索
        dense_vec, sparse_vec, course_id, top_k=20
    )
"""

from __future__ import annotations

import logging
import os as _os
from pathlib import Path

from coursepilot.config import settings
from coursepilot.rag.config import config

logger = logging.getLogger(__name__)

# ── Windows 下 Milvus Lite 3.0 的 manifest.save() 使用 os.rename，
#    目标已存在时 Windows 抛出 FileExistsError。在 milvus_lite 内部用 os.replace 替换。 ──
if _os.name == "nt":
    try:
        import milvus_lite.storage.manifest as _manifest_mod
        _manifest_mod.os.rename = _os.replace
    except Exception:
        pass

COLLECTION_NAME = "knowledge_units"
DIM = config.dim  # 1024


class VectorStore:
    """
    Milvus Lite 向量存储

    封装 Collection 生命周期、批量插入、混合检索、按条件删除
    """

    def __init__(self, db_path: str | None = None):
        from pymilvus import MilvusClient

        db_path = db_path or str(Path(settings.milvus_uri))
        # 确保目录存在
        db_dir = Path(db_path).parent
        if db_dir != Path("."):
            db_dir.mkdir(parents=True, exist_ok=True)

        logger.info("连接 Milvus Lite: %s", db_path)
        self.client = MilvusClient(db_path)

    # == Collection 管理

    def create_collection(self) -> None:
        """创建 collection + 双索引，幂等（已存在则跳过）"""
        if self.client.has_collection(COLLECTION_NAME):
            logger.info("已存在 collection %s，跳过创建", COLLECTION_NAME)
            return

        from pymilvus import DataType

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("uuid", DataType.VARCHAR, max_length=36)
        schema.add_field("dense_vec", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("sparse_vec", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("kp_id", DataType.VARCHAR, max_length=36)
        schema.add_field("course_id", DataType.VARCHAR, max_length=36)
        schema.add_field("kp_path", DataType.VARCHAR, max_length=512)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)

        # 先建 collection + dense 索引（不包含 sparse，避免 Windows 下 manifest rename 竞态）
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vec",
            index_type="FLAT",  # 修改: 使用标准的 FLAT 索引
            metric_type="IP",
            # params={"nlist": 128},    # 移除: FLAT 索引不需要 nlist 参数
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

        # 单独建 sparse 索引
        sparse_index = self.client.prepare_index_params()
        sparse_index.add_index(
            field_name="sparse_vec",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        self.client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=sparse_index,
        )
        logger.info("创建 collection %s 成功", COLLECTION_NAME)

    # == CRUD

    def insert(self, vectors: list[dict]) -> list[int]:
        """
        批量插入向量

        :param vectors: [{
            "uuid": str, "dense_vec": list[float], "sparse_vec": dict[int, float],
            "kp_id": str, "course_id": str, "kp_path": str, "content": str,
        }, ...]
        :return: insert_ids (list[int])
        """
        if not vectors:
            return []

        data = [
            {
                "uuid": v["uuid"],
                "dense_vec": v["dense_vec"],
                "sparse_vec": v["sparse_vec"],
                "kp_id": v["kp_id"],
                "course_id": v["course_id"],
                "kp_path": v.get("kp_path", ""),
                "content": v["content"],
            }
            for v in vectors
        ]

        result = self.client.insert(collection_name=COLLECTION_NAME, data=data)
        logger.info("Milvus 插入 %d 条向量", len(data))
        return result["ids"]

    def hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        course_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        混合检索 + 内置 RRF

        :param dense_vec: 稠密向量
        :param sparse_vec: 稀疏向量
        :param course_id: 课程 id
        :param top_k: top K
        :return: [{
            "id": int, "uuid": str, "kp_id": str,
            "kp_path": str, "content": str, "score": float
        }, ...]
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        # Dense 检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vec",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=config.dense_top_k,
        )

        # Sparse 检索请求（仅在启用时参与）
        search_requests = [dense_req]
        if config.enable_sparse and sparse_vec:
            sparse_req = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vec",
                param={"metric_type": "IP"},
                limit=config.sparse_top_k,
            )
            search_requests.append(sparse_req)

        filter_expr = f'course_id == "{course_id}"'

        results = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=search_requests,
            ranker=RRFRanker(k=config.rrf_k),
            filter=filter_expr,
            output_fields=["uuid", "kp_id", "kp_path", "content"],
            limit=top_k,
        )

        # pymilvus 3.0 返回 [{'id': ..., 'distance': ..., 'entity': {...}}, ...]
        # 标准化为扁平 dict 格式
        flat = []
        for row in results[0]:
            entity = row.get("entity", row)  # 兼容两种格式
            flat.append({
                "id": entity.get("id", row.get("id")),
                "uuid": entity.get("uuid", ""),
                "kp_id": entity.get("kp_id", ""),
                "kp_path": entity.get("kp_path", ""),
                "content": entity.get("content", ""),
                "score": row.get("distance", row.get("score", 0.0)),
            })
        return flat

    def delete_by_uuids(self, uuids: list[str]) -> None:
        """按 uuid 批量删除（直接使用 filter 表达式删除）"""
        if not uuids:
            return

        # 将 list 转换为 Milvus 支持的 filter 格式: uuid in ["u1", "u2"]
        # 注意字符串需要用双引号包裹
        uuid_str = ",".join([f'"{uid}"' for uid in uuids])
        filter_expr = f"uuid in [{uuid_str}]"

        self.client.delete(collection_name=COLLECTION_NAME, filter=filter_expr)
        logger.info("Milvus 删除 %d 条向量", len(uuids))

    def delete_by_course(self, course_id: str) -> None:
        """删除某课程的全部向量（直接使用 filter 表达式删除）"""
        filter_expr = f'course_id == "{course_id}"'
        self.client.delete(collection_name=COLLECTION_NAME, filter=filter_expr)
        logger.info("Milvus 删除课程 %s 的所有向量", course_id)

    def count(self) -> int:
        """collection 中向量总数"""
        # 确保之前插入的数据已经落盘，避免统计延误
        self.client.flush(COLLECTION_NAME)
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        return stats.get("row_count", 0)

    def drop_collection(self) -> None:
        """删除 collection（用于全量重建）"""
        if self.client.has_collection(COLLECTION_NAME):
            self.client.drop_collection(COLLECTION_NAME)
            logger.info("删除 collection %s 成功", COLLECTION_NAME)

    def query_by_course(self, course_id: str) -> list[dict]:
        """查询某课程的全部向量元数据（不含向量本身）"""
        filter_expr = f'course_id == "{course_id}"'
        return self.client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr,
            output_fields=["uuid", "kp_id", "kp_path", "content"],
        )
