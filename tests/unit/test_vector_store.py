import logging
import time
import os
import shutil
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from pymilvus.exceptions import MilvusException

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置常量 ---
COLLECTION_NAME = "test_dual_index_collection"
DIM = 768  # 向量维度
MILVUS_FILE = "milvus_demo.db"  # Milvus Lite 本地文件


class MilvusTester:
    def __init__(self):
        # 初始化 Milvus Lite 客户端
        self.client = MilvusClient(uri=MILVUS_FILE)

    def create_collection(self) -> None:
        """创建 collection + 双索引 - Windows 兼容版本"""
        # 为了测试方便，如果已存在则先删除，确保测试纯净
        if self.client.has_collection(COLLECTION_NAME):
            logger.warning("Collection 已存在，为测试纯净性将其删除...")
            self.client.drop_collection(COLLECTION_NAME)
            time.sleep(1.0)  # 等待删除完成

        logger.info("开始创建 Collection...")

        # 1. 定义 Schema
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

        # 2. 创建 Collection (包含 Dense 索引)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vec",
            index_type="FLAT",  # Milvus Lite 推荐 FLAT
            metric_type="IP",
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection 创建成功，Dense 索引已加载。")

        # 3. 创建 Sparse 索引 - 改进的重试逻辑
        time.sleep(1.5)  # 给文件系统足够的时间

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 先检查索引是否已存在
                existing_indexes = self.client.list_indexes(collection_name=COLLECTION_NAME)
                if any('sparse_vec' in str(idx) for idx in existing_indexes):
                    logger.info("Sparse 索引已存在，跳过创建。")
                    return

                logger.info(f"尝试创建 Sparse 索引 (第 {attempt + 1} 次)...")

                # Windows 特殊处理：尝试清理可能存在的临时文件
                manifest_dir = os.path.join(MILVUS_FILE, "collections", COLLECTION_NAME)
                tmp_file = os.path.join(manifest_dir, "manifest.json.tmp")
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                        logger.info(f"已清理临时文件: {tmp_file}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")

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
                logger.info("Sparse 索引创建成功！")
                return

            except MilvusException as e:
                # 如果错误信息包含"索引已存在"，说明实际已创建成功
                if "already exists" in str(e):
                    logger.info("Sparse 索引已存在（可能在前一次尝试中已创建）。")
                    return

                if attempt < max_retries - 1:
                    logger.warning(f"创建 Sparse 索引失败: {e}")
                    logger.info("等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    logger.error(f"创建 Sparse 索引最终失败: {e}")
                    logger.error("提示：继续使用 Dense 索引进行测试...")

    def insert_test_data(self):
        """插入一些测试数据"""
        logger.info("插入测试数据...")
        data = []
        for i in range(10):
            # 构造稀疏向量: {index: value, ...}
            sparse_vec = {i: 0.5, i + 10: 0.8}

            row = {
                "uuid": f"uuid-{i}",
                "dense_vec": [0.1] * DIM,  # 模拟向量
                "sparse_vec": sparse_vec,
                "kp_id": f"kp-{i}",
                "course_id": f"course-{i}",
                "kp_path": f"/path/{i}",
                "content": f"这是第 {i} 条测试内容"
            }
            data.append(row)

        self.client.insert(COLLECTION_NAME, data)
        logger.info(f"成功插入 {len(data)} 条数据。")

    def test_dense_search(self):
        """测试 Dense 向量检索"""
        logger.info("执行 Dense 向量检索...")

        search_dense_vec = [0.1] * DIM

        # 修正：使用 search_params 而不是 param
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[search_dense_vec],
            anns_field="dense_vec",
            search_params={"metric_type": "IP"},
            limit=5,
            output_fields=["uuid", "content"]
        )

        logger.info("Dense 检索结果：")
        for hits in results:
            for hit in hits:
                logger.info(f"  ID: {hit.id}, Score: {hit.distance:.4f}, Content: {hit.entity.get('content')}")

    def test_sparse_search(self):
        """测试 Sparse 向量检索"""
        logger.info("执行 Sparse 向量检索...")

        # 模拟稀疏检索向量：我们想找包含 token 0 的文档
        search_sparse_vec = [{0: 1.0}]

        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=search_sparse_vec,
                anns_field="sparse_vec",
                search_params={"metric_type": "IP"},
                limit=5,
                output_fields=["uuid", "content"]
            )

            logger.info("Sparse 检索结果：")
            for hits in results:
                for hit in hits:
                    logger.info(f"  ID: {hit.id}, Score: {hit.distance:.4f}, Content: {hit.entity.get('content')}")
        except Exception as e:
            logger.warning(f"Sparse 检索失败（索引可能未创建成功）: {e}")

    def test_hybrid_search(self):
        """测试混合检索"""
        logger.info("执行混合检索...")

        # 准备检索向量
        search_dense_vec = [0.1] * DIM
        search_sparse_vec = [{0: 1.0}]

        try:
            # 1. Dense 检索请求
            dense_req = AnnSearchRequest(
                data=[search_dense_vec],
                anns_field="dense_vec",
                param={"metric_type": "IP", "params": {}},
                limit=5
            )

            # 2. Sparse 检索请求
            sparse_req = AnnSearchRequest(
                data=search_sparse_vec,
                anns_field="sparse_vec",
                param={"metric_type": "IP"},
                limit=5
            )

            # 3. 混合检索 - 使用 RRFRanker
            results = self.client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(),
                limit=5,
                output_fields=["uuid", "content"]
            )

            logger.info("混合检索结果：")
            for hits in results:
                for hit in hits:
                    logger.info(f"  ID: {hit.id}, Score: {hit.distance:.4f}, Content: {hit.entity.get('content')}")
        except Exception as e:
            logger.warning(f"混合检索失败（Sparse 索引可能未创建成功）: {e}")


if __name__ == "__main__":
    # Windows 用户可选：删除旧数据库文件，彻底避免文件冲突问题
    # 取消下面的注释可以每次运行前删除旧数据
    # if os.path.exists(MILVUS_FILE):
    #     logger.info(f"删除旧数据库文件: {MILVUS_FILE}")
    #     shutil.rmtree(MILVUS_FILE)

    tester = MilvusTester()

    # 1. 创建 Collection 和索引
    tester.create_collection()

    # 2. 插入数据
    tester.insert_test_data()

    # 3. 测试检索
    print("\n" + "=" * 60)
    tester.test_dense_search()
    print("\n" + "=" * 60)
    tester.test_sparse_search()
    print("\n" + "=" * 60)
    tester.test_hybrid_search()

