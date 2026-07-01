"""查询 Milvus 数据库：查看集合 + 向量数量 + 显示前几条"""
import sys
sys.path.insert(0, "src")

from coursepilot.rag.vector_store import VectorStore

store = VectorStore()

if store.client.has_collection("knowledge_units"):
    store.client.load_collection("knowledge_units")
    stats = store.client.get_collection_stats("knowledge_units")
    print(f"行数: {stats.get('row_count', 0)}")

    # 查前 5 条
    rows = store.client.query(
        collection_name="knowledge_units",
        filter="",
        output_fields=["id", "uuid", "kp_id", "course_id", "kp_path", "content"],
        limit=5,
    )
    for r in rows:
        print(f"\n--- id={r['id']} ---")
        print(f"  kp_path: {r.get('kp_path', '')}")
        print(f"  content: {r.get('content', '')[:80]}...")
else:
    print("集合 'knowledge_units' 不存在")
