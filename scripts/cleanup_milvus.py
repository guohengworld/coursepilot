"""Milvus 数据清理工具

删除 Milvus 中的向量数据，支持三种粒度：
  1. --all            清空整个 collection（重建 schema）
  2. --course <uuid>  按课程删除
  3. --document <uuid> 按文档删除

用法：
  PYTHONPATH=src .venv/Scripts/python -m scripts.cleanup_milvus --all
  PYTHONPATH=src .venv/Scripts/python -m scripts.cleanup_milvus --course <course_uuid>
  PYTHONPATH=src .venv/Scripts/python -m scripts.cleanup_milvus --document <doc_uuid>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def delete_all() -> int:
    """清空整个 Milvus collection（删除重建，相当于重置 schema）。"""
    from coursepilot.rag.vector_store import VectorStore

    store = VectorStore()
    count_before = store.count()
    store.drop_collection()
    store.create_collection()
    print(f"  ✅ 已清空 Milvus collection（删除 {count_before} 条向量并重建 schema）")
    return count_before


def delete_by_course(course_id: str) -> int:
    """删除某课程的全部向量。"""
    from coursepilot.rag.vector_store import VectorStore

    store = VectorStore()
    store.create_collection()

    # 先查数量
    before = store.query_by_course(course_id)
    count = len(before)

    store.delete_by_course(course_id)
    print(f"  ✅ 已删除课程 {course_id} 的 {count} 条向量")
    return count


def delete_by_document(document_id: str) -> int:
    """删除某文档的全部向量（利用 Milvus document_id 字段直接过滤）。"""
    from coursepilot.rag.vector_store import VectorStore

    store = VectorStore()
    store.create_collection()
    store.delete_by_document(document_id)
    print(f"  ✅ 已删除文档 {document_id} 的全部向量")
    return 0  # Milvus 不返回删除行数


def main():
    # Windows 终端编码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Milvus 数据清理工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="清空整个 Milvus（重建 schema）")
    group.add_argument("--course", type=str, help="按课程 ID 删除")
    group.add_argument("--document", type=str, help="按文档 ID 删除")
    args = parser.parse_args()

    if args.all:
        count = delete_all()
    elif args.course:
        count = delete_by_course(args.course)
    elif args.document:
        count = delete_by_document(args.document)

    print(f"\n  📊 共处理 {count} 条向量")

    # 验证
    from coursepilot.rag.vector_store import VectorStore
    remaining = VectorStore().count()
    print(f"  📊 Milvus 剩余向量数: {remaining}")


if __name__ == "__main__":
    main()
