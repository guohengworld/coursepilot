"""查看 Milvus 向量数据库中的数据

用法：
    PYTHONPATH=src .venv/Scripts/python scripts/inspect_milvus.py            # 概览
    PYTHONPATH=src .venv/Scripts/python scripts/inspect_milvus.py --sample 5 # 每条采样
    PYTHONPATH=src .venv/Scripts/python scripts/inspect_milvus.py --course-id <uuid>  # 按课程过滤
"""

import argparse
import sys
from pathlib import Path

# Windows GBK 终端强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coursepilot.rag.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="查看 Milvus 向量数据")
    parser.add_argument("--sample", type=int, default=0, help="采样展示 N 条记录详情")
    parser.add_argument("--course-id", type=str, default=None, help="按课程 ID 过滤")
    args = parser.parse_args()

    store = VectorStore()

    # Milvus Lite 重启后 collection 处于 released 状态，需要先 load
    store.client.load_collection("knowledge_units")

    total = store.count()
    print(f"Milvus collection: knowledge_units")
    print(f"Total rows: {total}")
    print()

    if total == 0:
        print("(empty)")
        return

    # 查询数据
    if args.course_id:
        rows = store.query_by_course(args.course_id)
        print(f"Course {args.course_id}: {len(rows)} rows")
    else:
        rows = store.client.query(
            collection_name="knowledge_units",
            filter="id >= 0",
            output_fields=["uuid", "kp_id", "course_id", "kp_path", "content"],
            limit=10000,
        )
        print(f"Total queried: {len(rows)} rows")

    if not rows:
        print("(no rows returned)")
        return

    # 按课程分组统计
    course_counts: dict[str, int] = {}
    for r in rows:
        cid = r.get("course_id", "?")
        course_counts[cid] = course_counts.get(cid, 0) + 1
    print(f"\nBy course:")
    for cid, cnt in sorted(course_counts.items()):
        print(f"  {cid}: {cnt}")

    # 按 kp_path 前两级分组
    kp_counts: dict[str, int] = {}
    for r in rows:
        kp = r.get("kp_path", "") or "(empty)"
        parts = kp.split("/")
        prefix = "/".join(parts[:2]) if len(parts) >= 2 else kp
        kp_counts[prefix] = kp_counts.get(prefix, 0) + 1
    print(f"\nBy kp_path (top-2 levels):")
    for kp, cnt in sorted(kp_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kp}: {cnt}")

    if args.sample > 0:
        print(f"\n{'='*70}")
        print(f"Sample {min(args.sample, len(rows))} rows:")
        print(f"{'='*70}")
        for i, r in enumerate(rows[: args.sample]):
            print(f"\n--- [{i+1}] ---")
            print(f"  uuid:      {r.get('uuid', '')}")
            print(f"  course_id: {r.get('course_id', '')}")
            print(f"  kp_path:   {r.get('kp_path', '')}")
            content = r.get("content", "")
            print(f"  content:   {content[:200]}{'...' if len(content) > 200 else ''}")


if __name__ == "__main__":
    main()
