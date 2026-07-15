from pymilvus import MilvusClient
import logging
from coursepilot.config import settings

# 配置日志输出（方便查看结果）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def test_milvus_collection_data():
    # 1. Milvus 数据库路径（与你的配置一致）
    milvus_uri = settings.milvus_uri
    collection_name = "knowledge_units"  # 你的集合名称

    # 2. 创建 Milvus 客户端连接
    try:
        client = MilvusClient(milvus_uri)
        logging.info(f"成功连接到 Milvus Lite: {milvus_uri}")
    except Exception as e:
        logging.error(f"连接 Milvus 失败: {e}")
        return

    # 3. 检查集合是否存在
    if not client.has_collection(collection_name):
        logging.error(f"集合 '{collection_name}' 不存在！")
        return

    # 4. 加载集合（确保数据在内存中）
    try:
        client.load_collection(collection_name)
        logging.info(f"集合 '{collection_name}' 加载成功")
    except Exception as e:
        logging.error(f"加载集合失败: {e}")
        return

    # 5. 获取集合统计信息（数据量）
    try:
        stats = client.get_collection_stats(collection_name)
        row_count = stats.get("row_count", 0)
        logging.info(f"集合 '{collection_name}' 的数据量: {row_count}")
    except Exception as e:
        logging.error(f"获取集合统计失败: {e}")
        return

    # 6. 如果有数据，查询前 3 条数据验证
    if row_count > 0:
        logging.info("查询前 3 条数据（验证内容）：")
        try:
            results = client.query(
                collection_name=collection_name,
                output_fields=["uuid", "course_id", "kp_path", "content"],  # 查询需要的字段
                limit=3  # 限制返回 3 条
            )
            for idx, result in enumerate(results, 1):
                logging.info(f"\n--- 数据 {idx} ---")
                logging.info(f"  uuid: {result.get('uuid', 'N/A')}")
                logging.info(f"  course_id: {result.get('course_id', 'N/A')}")
                logging.info(f"  kp_path: {result.get('kp_path', 'N/A')}")
                logging.info(f"  content: {result.get('content', 'N/A')[:100]}...")  # 内容取前 100 字符
        except Exception as e:
            logging.error(f"查询数据失败: {e}")
    else:
        logging.warning(f"集合 '{collection_name}' 中没有数据")

if __name__ == "__main__":
    test_milvus_collection_data()

