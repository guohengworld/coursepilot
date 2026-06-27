"""从数据库导出全量 knowledge unit，供 RAG 评估出题使用。"""
import asyncio, json
from pathlib import Path

from sqlalchemy import select

from coursepilot.db import get_session_etx
from coursepilot.models.knowledge_point import KnowledgePoint
from coursepilot.models.knowledge_unit import KnowledgeUnit

OUTPUT = Path("tests/fixtures/exported_units.json")


async def main():
    async with get_session_etx() as sess:
        # 查询所有 KP，按路径排序
        kp_rows = (await sess.execute(
            select(KnowledgePoint).order_by(KnowledgePoint.kp_path)
        )).scalars().all()

        result = []
        for kp in kp_rows:
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_units = sum(len(kp["units"]) for kp in result)
    with_summary = sum(
        1 for kp in result for u in kp["units"] if len(u["summary"]) > 10
    )
    print(f"Exported {len(result)} KPs, {total_units} units")
    print(f"Units with summary: {with_summary}/{total_units} ({100*with_summary//max(1,total_units)}%)")


if __name__ == "__main__":
    asyncio.run(main())
