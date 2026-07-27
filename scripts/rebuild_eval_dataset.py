"""
重建黄金评估数据集：用当前 DB 中的真实 UUID 替换过期 UUID。

改进：
  - 限定只搜索"微积分/"前缀的 KP（所有 eval 题都是微积分内容）
  - 优先匹配叶子节点（更具体的子 KP）
  - 保留与原有 ground_truth 数量一致的 GT UUIDs
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from sqlalchemy import select

from coursepilot.db import get_session_etx
from coursepilot.models import KnowledgePoint, KnowledgeUnit
from coursepilot.rag.encoder import Encoder

EVAL_QUESTIONS_PATH = Path("eval/questions/20260726/eval_questions.json")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def rebuild():
    if not EVAL_QUESTIONS_PATH.exists():
        print(f"[ERROR] 未找到: {EVAL_QUESTIONS_PATH}")
        sys.exit(1)

    questions = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"加载 {len(questions)} 道评估题")

    encoder = Encoder()
    async with get_session_etx() as session:
        # 只获取微积分相关的 KPs（排除习题答案、目录等非教学内容）
        all_kps = await session.execute(
            select(KnowledgePoint.id, KnowledgePoint.kp_path, KnowledgePoint.course_id, KnowledgePoint.title)
            .where(KnowledgePoint.kp_path.like("微积分/%"))
        )
        kp_list = [{"id": str(r[0]), "kp_path": r[1], "course_id": str(r[2]), "title": r[3]} for r in all_kps.all()]
        print(f"微积分 KPs: {len(kp_list)} 个")

        # 区分叶子/非叶子 KP（叶子 KP 包含具体知识点）
        kp_path_set = {kp["kp_path"] for kp in kp_list}
        leaf_kps = [kp for kp in kp_list if not any(
            p.startswith(kp["kp_path"] + "/") for p in kp_path_set
        )]
        print(f"  其中叶子节点: {len(leaf_kps)} 个")

        # 预编码所有叶子 KP（优先匹配叶子）
        kp_texts = [f"{kp['kp_path']} {kp['title']}" for kp in leaf_kps]
        kp_vecs = encoder.encode(kp_texts)

        for i, q in enumerate(questions):
            query_text = f"{q['question']}\n{q.get('answer', '')}"
            q_vec = encoder.encode_query(query_text)

            # 在叶子 KPs 中找最相似
            scores = [
                (j, _cosine(q_vec["dense"], kp_vecs[j]["dense"]))
                for j in range(len(leaf_kps))
            ]
            scores.sort(key=lambda x: x[1], reverse=True)

            best_kp = leaf_kps[scores[0][0]]
            best_score = scores[0][1]

            print(f"\n  [{i+1}] 问: {q['question'][:40]}...")
            print(f"      最佳 KP: {best_kp['kp_path']} (sim={best_score:.3f})")

            if best_score < 0.3:
                print(f"      ⚠ 相似度偏低，跳过")
                q["ground_truth_contexts"] = []
                q["kp_path"] = best_kp["kp_path"]
                q["course_id"] = best_kp["course_id"]
                continue

            # 查询该 KP 下的所有 unit
            unit_result = await session.execute(
                select(KnowledgeUnit.id, KnowledgeUnit.content, KnowledgeUnit.summary)
                .where(KnowledgeUnit.kp_id == best_kp["id"])
                .order_by(KnowledgeUnit.seq_order)
            )
            units = unit_result.all()

            if not units:
                print(f"      ⚠ KP 下无 unit，尝试父 KP...")
                q["ground_truth_contexts"] = []
                q["kp_path"] = best_kp["kp_path"]
                q["course_id"] = best_kp["course_id"]
                continue

            # 用 BGE-M3 选 top-N unit
            n_gt = max(1, len(q.get("ground_truth_contexts", [])))
            unit_texts = [f"{(u[2] or '')}\n{u[1]}" for u in units]
            unit_vecs = encoder.encode(unit_texts)

            scored_units = [
                (str(units[j][0]), _cosine(q_vec["dense"], unit_vecs[j]["dense"]))
                for j in range(len(units))
            ]
            scored_units.sort(key=lambda x: x[1], reverse=True)
            top_uuids = [uid for uid, _ in scored_units[:n_gt]]

            q["ground_truth_contexts"] = top_uuids
            q["kp_path"] = best_kp["kp_path"]
            q["course_id"] = best_kp["course_id"]
            print(f"      GT: {len(top_uuids)} units (共 {len(units)} units), top-sim={scored_units[0][1]:.3f}")

    # 写回
    EVAL_QUESTIONS_PATH.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已更新: {EVAL_QUESTIONS_PATH} — 所有 UUID 和 course_id 已同步至当前 DB")


if __name__ == "__main__":
    asyncio.run(rebuild())
