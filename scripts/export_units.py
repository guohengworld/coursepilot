"""从数据库导出 knowledge unit，供 RAG 评估出题使用。

用法：
    PYTHONPATH=src .venv/Scripts/python -m scripts.export_units
    PYTHONPATH=src .venv/Scripts/python -m scripts.export_units --document-id <uuid>
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from coursepilot.db import get_session_etx
from coursepilot.models.knowledge_point import KnowledgePoint
from coursepilot.models.knowledge_unit import KnowledgeUnit

OUTPUT = Path("eval/questions/exported_units.json")

# 非教学内容前缀，评估时排除
EXCLUDE_PREFIXES = [
    "微积分/习题参考答案",
    "微积分/大学数学微积分",
]


def _should_exclude(kp_path: str) -> bool:
    """排除前言、目录、内容提要、习题答案等非教学 KP。"""
    return any(kp_path.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


async def main():
    parser = argparse.ArgumentParser(description="导出 KnowledgeUnit 素材")
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="仅导出指定文档下的 KP/Unit（UUID）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT),
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_session_etx() as sess:
        stmt = select(KnowledgePoint).order_by(KnowledgePoint.kp_path)
        if args.document_id:
            from uuid import UUID
            stmt = stmt.where(KnowledgePoint.document_id == UUID(args.document_id))
        kp_rows = (await sess.execute(stmt)).scalars().all()

        result = []
        excluded_kps = 0
        for kp in kp_rows:
            if _should_exclude(kp.kp_path):
                excluded_kps += 1
                continue

            # 查该 KP 下的所有 unit
            units = (await sess.execute(
                select(KnowledgeUnit)
                .where(KnowledgeUnit.kp_id == kp.id)
                .order_by(KnowledgeUnit.seq_order)
            )).scalars().all()

            result.append({
                "kp_path": kp.kp_path,
                "kp_id": str(kp.id),
                "course_id": str(kp.course_id),
                "units": [
                    {
                        "uuid": str(u.id),
                        "content": u.content,
                        "summary": u.summary or "",
                        "page_ref": u.page_ref or "",
                    }
                    for u in units
                ],
            })

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_units = sum(len(kp["units"]) for kp in result)
    with_summary = sum(
        1 for kp in result for u in kp["units"] if len(u["summary"]) > 10
    )
    print(f"Exported {len(result)} KPs, {total_units} units")
    print(f"Excluded {excluded_kps} non-teaching KPs")
    print(f"Units with summary: {with_summary}/{total_units} ({100*with_summary//max(1,total_units)}%)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
