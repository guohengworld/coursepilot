"""生成 RAG 黄金评估数据集。

用法：
    PYTHONPATH=src .venv/Scripts/python -m scripts.generate_eval_dataset \
        --course-id e7a20f2f-c98e-4ff3-9938-04351616e66d \
        --document-id 92d20a3b-1bc4-4b2d-b582-e257606b52c9 \
        --output eval/questions/eval_questions_candidate.json

流程：
    1. 读取 eval/questions/exported_units.json（由 scripts.export_units 生成）
    2. 按章节均匀采样 unit，调用 LLM 生成问题、答案、ground_truth_contexts
    3. 校验 UUID、kp_path、字段完整性
    4. 输出候选数据集，供人工校验后改名为 eval_questions.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from coursepilot.config import settings
from coursepilot.evaluation.dataset_generator import (
    DEFAULT_TYPE_QUOTAS,
    EvalDatasetGenerator,
    print_distribution,
)


DEFAULT_EXPORTED_UNITS = Path("eval/questions/exported_units.json")
DEFAULT_OUTPUT = Path("eval/questions/eval_questions_candidate.json")


async def main():
    parser = argparse.ArgumentParser(description="生成 RAG 黄金评估数据集")
    parser.add_argument(
        "--course-id",
        type=str,
        required=True,
        help="课程 UUID",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        required=True,
        help="文档 UUID",
    )
    parser.add_argument(
        "--exported-units",
        type=str,
        default=str(DEFAULT_EXPORTED_UNITS),
        help="导出的 KnowledgeUnit 素材 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="候选数据集输出路径",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        default=True,
        help="按目标配额重平衡题型数量",
    )
    parser.add_argument(
        "--no-rebalance",
        action="store_true",
        help="禁用重平衡",
    )
    parser.add_argument(
        "--max-units-per-chapter",
        type=int,
        default=10,
        help="每章输入 LLM 的最大 unit 数",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="LLM 并发调用数上限",
    )
    parser.add_argument(
        "--questions-per-chapter-min",
        type=int,
        default=5,
        help="每章最少生成题数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM 温度",
    )

    args = parser.parse_args()

    if not settings.llm_api_key:
        print("[ERROR] 未配置 LLM_API_KEY，请检查 .env")
        sys.exit(1)

    exported_path = Path(args.exported_units)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = EvalDatasetGenerator(
        max_units_per_chapter=args.max_units_per_chapter,
        questions_per_chapter_min=args.questions_per_chapter_min,
        temperature=args.temperature,
    )

    print(f"加载素材: {exported_path}")
    print(f"课程 ID: {args.course_id}")
    print(f"文档 ID: {args.document_id}")
    print(f"目标题型配额: {DEFAULT_TYPE_QUOTAS}")
    print()

    questions = await generator.generate(
        exported_path,
        course_id=args.course_id,
        document_id=args.document_id,
        max_concurrency=args.max_concurrency,
    )

    if not args.no_rebalance:
        questions = generator.rebalance(questions)

    output_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"生成完成: {len(questions)} 道候选题目")
    print(f"输出文件: {output_path}")
    print_distribution(questions)
    print("=" * 60)
    print("\n下一步: 人工校验后改名为 eval/questions/eval_questions.json")


if __name__ == "__main__":
    asyncio.run(main())
