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

        if db_path is None:
            db_path = settings.milvus_uri
            # 如果是相对路径，基于项目根目录转成绝对路径
            if not Path(db_path).is_absolute():
                from coursepilot.config import _PROJECT_ROOT
                db_path = str(_PROJECT_ROOT / db_path)

        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("连接 Milvus Lite: %s", db_path)
        self.client = MilvusClient(
            db_path,
            # gRPC keepalive 配置：降低 ping 频率，避免 Milvus Lite 触发
            # "too_many_pings" GOAWAY 导致连接断开/挂起
            grpc_options=[
                ("grpc.keepalive_time_ms", 60000),       # 60s ping 一次（默认 10s 太频繁）
                ("grpc.keepalive_timeout_ms", 20000),    # ping 超时 20s
                ("grpc.keepalive_permit_without_calls", 1),  # 无活跃调用时也允许 keepalive
                ("grpc.http2.max_pings_without_data", 0),    # 不限次数的空数据 ping
            ],
        )

    # == Collection 管理

    def create_collection(self) -> None:
        """创建 collection + 双索引，幂等（已存在则跳过）"""
        if self.client.has_collection(COLLECTION_NAME):
            logger.info("已存在 collection %s，跳过创建", COLLECTION_NAME)
            self.client.load_collection(COLLECTION_NAME)
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
        self.client.load_collection(COLLECTION_NAME)
        logger.info("创建 collection %s 成功", COLLECTION_NAME)

    def _ensure_loaded(self) -> None:
        """确保 collection 存在并已加载（幂等）"""
        self.create_collection()

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

        self._ensure_loaded()

        # Dense 检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vec",
            param={"metric_type": "IP", "params": {}},
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

        filter_expr = {"course_id": course_id}

        print(f"[vector_store] hybrid_search 开始 (course={course_id}, dense_top_k={config.dense_top_k}, sparse={config.enable_sparse})")
        t0 = __import__("time").monotonic()
        try:
            results = self.client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=search_requests,
                ranker=RRFRanker(k=config.rrf_k),
                filter=filter_expr,
                output_fields=["uuid", "kp_id", "kp_path", "content"],
                limit=top_k,
            )
            elapsed = (__import__("time").monotonic() - t0) * 1000
            print(f"[vector_store] hybrid_search 完成, 耗时={elapsed:.0f}ms, 结果数={len(results[0]) if results else 0}")
        except Exception as e:
            elapsed = (__import__("time").monotonic() - t0) * 1000
            print(f"[vector_store] hybrid_search 失败, 耗时={elapsed:.0f}ms, 错误={e}")
            raise

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
        self._ensure_loaded()
        filter_expr = f'course_id == "{course_id}"'
        print(f"查询条件: {filter_expr}")  # 确认条件正确
        return self.client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr,
            output_fields=["uuid", "kp_id", "kp_path", "content"],
        )
