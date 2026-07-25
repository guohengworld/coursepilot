"""RAG 黄金评估数据集生成器。

设计目标：
- 严格基于当前数据库中的 KnowledgeUnit 内容出题；
- 覆盖全部教学章节，题型分布符合 RAG 评估体系构建指南；
- 生成的 ground_truth_contexts 必须是真实存在的 unit UUID；
- 输出候选数据集，供人工校验后转为正式 eval_questions.json。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import openai

from coursepilot.config import settings

logger = logging.getLogger(__name__)

# 非教学内容前缀，不参与出题
EXCLUDE_PREFIXES = [
    "微积分/习题参考答案",
    "微积分/大学数学微积分",
]

# 默认题型配额（总计约 40 题）
DEFAULT_TYPE_QUOTAS = {
    "concept": 10,
    "calculation": 8,
    "theorem": 8,
    "comparison": 6,
    "application": 4,
    "unanswerable": 4,
}

VALID_TYPES = set(DEFAULT_TYPE_QUOTAS.keys())


@dataclass
class ChapterPack:
    """一个章节的素材包。"""

    chapter: str
    kp_items: list[dict] = field(default_factory=list)
    all_kp_paths: set[str] = field(default_factory=set)


@dataclass
class GenerationPlan:
    """某章节需要生成的题型与数量。"""

    chapter: str
    type_plan: dict[str, int] = field(default_factory=dict)


# ── Prompts ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是大学数学助教，负责为 RAG 检索系统构建评估用问答对。

你会收到教材某一章的若干知识片段，每个片段有唯一的 uuid。

请严格按照要求的题型和数量为该章节生成评估问答对。所有问题、答案必须完全基于提供的教材片段，不得引入外部知识。必须生成足够数量的题目，且每种题型的数量不得少于要求。

题型说明：
- concept: 概念解释（"什么是..."、"说明...的定义/含义/几何意义"）
- calculation: 计算题（"求..."、"计算..."、"解..."）
- theorem: 定理/公式/推导（"叙述...定理"、"推导...公式"、"证明..."）
- comparison: 辨析比较（"...和...有什么区别/联系"）
- application: 应用题（"用...解决...实际问题"）
- unanswerable: 不可回答题（问题看起来与教材相关，但答案不在提供的片段中；标注 unanswerable=true，answer 写"根据当前教材内容无法回答"）

输出严格的 JSON 数组，不要加任何解释文字：

[
  {
    "question": "问题文字",
    "answer": "标准答案（2-5句话，基于教材内容；不可答题写\"根据当前教材内容无法回答\"）",
    "ground_truth_contexts": ["uuid-1", "uuid-2"],
    "question_type": "concept|calculation|theorem|comparison|application|unanswerable",
    "kp_path": "知识点路径（必须是提供的片段所在路径）",
    "unanswerable": false
  }
]

要求：
1. 问题必须能用提供的教材片段回答（unanswerable 除外），不要编造教材中没有的内容。
2. 标准答案必须基于教材原意，不要引入外部知识。
3. ground_truth_contexts 必须是提供的 unit uuid 中真实存在的；可以有 1-3 个。
4. 每道题的 kp_path 必须是该题知识点所在的、教材片段中真实存在的路径。
5. 问题用中文，涉及数学公式用 LaTeX $...$ 或 $$...$$ 表达。
6. 不可答题要看起来像 legitimate 问题，但提供的片段确实无法给出完整答案。"""


# ── 工具函数 ───────────────────────────────────────────────────────


def _fix_latex_backslashes(text: str) -> str:
    """修复 LLM 输出 JSON 中未正确转义的 LaTeX 反斜杠。"""
    text = re.sub(r"\\{3,}", r"\\\\", text)
    text = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
    return text


def _extract_json_array(text: str) -> list[dict]:
    """从 LLM 回复中提取 JSON 数组，容忍截断。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text_fixed = _fix_latex_backslashes(text)
    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", text_fixed, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 逐个提取 JSON 对象
    objs = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text_fixed):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text_fixed[start : i + 1]
                try:
                    objs.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start = -1
    return objs


def _get_chapter_key(kp_path: str) -> str | None:
    """从 kp_path 提取章节名，如 '第3章 导数与微分'。"""
    parts = kp_path.split("/")
    if len(parts) < 2:
        return None
    return parts[1]


def _should_exclude(kp_path: str) -> bool:
    """排除非教学内容。"""
    return any(kp_path.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def _pick_representative_units(kp_items: list[dict], max_total: int = 18) -> list[dict]:
    """从章节的所有 KP 中均匀选取代表性 unit。

    优先保留叶子节点（有 unit 的 KP），并在章节内均匀采样，避免只取前几题。
    """
    all_units: list[dict] = []
    for kp in kp_items:
        for u in kp["units"]:
            all_units.append({
                **u,
                "kp_path": kp["kp_path"],
                "kp_id": kp["kp_id"],
            })

    if len(all_units) <= max_total:
        return all_units

    # 均匀采样
    step = len(all_units) / max_total
    picked: list[dict] = []
    for i in range(max_total):
        idx = min(int(i * step), len(all_units) - 1)
        picked.append(all_units[idx])
    return picked


def _build_unit_reference(chapter: str, units: list[dict]) -> str:
    """为某章构建 unit 参考文本。"""
    lines = [f"## {chapter}"]
    for u in units:
        uuid = u["uuid"]
        kp_path = u["kp_path"]
        summary = (u.get("summary") or "")[:120]
        content_preview = u["content"][:600].replace("\n", " ")
        lines.append(
            f'<unit uuid="{uuid}" path="{kp_path}" page="{u.get("page_ref", "")}">\n'
            f"  summary: {summary}\n"
            f"  content: {content_preview}...\n"
            f"</unit>"
        )
    return "\n".join(lines)


def _distribute_types(
    total_quota: dict[str, int], chapter_count: int, buffer: float = 1.2
) -> list[dict[str, int]]:
    """将全局题型配额分配到各章节，每章生成略多于最终目标的数量。

    策略：
    - 先按题型循环分配到各章，保证章节覆盖；
    - 每章总题数控制在 6-9 道，便于 LLM 一次性输出。
    """
    import math

    plans: list[Counter] = [Counter() for _ in range(chapter_count)]
    total = sum(total_quota.values())
    per_chapter = max(6, math.ceil(total / chapter_count * buffer))

    type_cycle = list(total_quota.keys())
    chapter_idx = 0
    # 第一轮：确保每种题型都覆盖到每章
    for _ in range(per_chapter * chapter_count):
        qtype = type_cycle[chapter_idx % len(type_cycle)]
        plans[chapter_idx % chapter_count][qtype] += 1
        chapter_idx += 1

    # 第二轮：补足总配额中数量较多的题型
    remaining = Counter()
    for qtype, count in total_quota.items():
        current = sum(p[qtype] for p in plans)
        if count > current:
            remaining[qtype] = count - current

    extra_idx = 0
    for qtype, count in remaining.items():
        for _ in range(count):
            plans[extra_idx % chapter_count][qtype] += 1
            extra_idx += 1

    return [dict(p) for p in plans]


# ── 核心生成器 ─────────────────────────────────────────────────────


class EvalDatasetGenerator:
    """基于当前 DB 的 KnowledgeUnit 生成 RAG 黄金评估数据集。"""

    def __init__(
        self,
        llm_client: openai.AsyncOpenAI | None = None,
        model: str | None = None,
        type_quotas: dict[str, int] | None = None,
        max_units_per_chapter: int = 10,
        questions_per_chapter_min: int = 5,
        temperature: float = 0.7,
    ):
        self.llm_client = llm_client or openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = model or settings.llm_model
        self.type_quotas = type_quotas or DEFAULT_TYPE_QUOTAS.copy()
        self.max_units_per_chapter = max_units_per_chapter
        self.questions_per_chapter_min = questions_per_chapter_min
        self.temperature = temperature

    # ── 素材加载 ──────────────────────────────────────────────

    def load_exported_units(self, path: str | Path) -> list[ChapterPack]:
        """加载由 export_units 导出的 JSON，按章节分组。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"素材文件不存在: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))

        chapters: dict[str, list[dict]] = {}
        for kp in raw:
            kp_path = kp["kp_path"]
            if _should_exclude(kp_path):
                continue
            chapter = _get_chapter_key(kp_path)
            if chapter is None:
                continue
            # 跳过没有 unit 的 KP
            if not kp.get("units"):
                continue
            chapters.setdefault(chapter, []).append(kp)

        packs = [
            ChapterPack(
                chapter=ch,
                kp_items=items,
                all_kp_paths={kp["kp_path"] for kp in items},
            )
            for ch, items in sorted(chapters.items())
        ]
        logger.info("加载 %d 个章节", len(packs))
        return packs

    # ── LLM 生成 ──────────────────────────────────────────────

    async def generate_for_type(
        self,
        chapter_pack: ChapterPack,
        qtype: str,
        count: int,
    ) -> list[dict]:
        """为单个章节的指定题型生成评估问答对。

        按题型单独调用 LLM，能更精确地控制输出数量和类型。
        """
        units = _pick_representative_units(
            chapter_pack.kp_items, self.max_units_per_chapter
        )
        if not units:
            logger.warning("章节 %s 没有可用 unit", chapter_pack.chapter)
            return []

        unit_ref = _build_unit_reference(chapter_pack.chapter, units)
        valid_uuids = {u["uuid"] for u in units}

        prompt = self._build_type_prompt(
            chapter_pack.chapter, qtype, count, unit_ref
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=8192,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            logger.error("[%s][%s] LLM 调用失败: %s", chapter_pack.chapter, qtype, e)
            return []

        text = response.choices[0].message.content or ""
        questions = _extract_json_array(text)
        valid, warnings = self._validate_questions(
            questions,
            valid_uuids,
            chapter_pack.all_kp_paths,
            chapter_pack.chapter,
            expected_type=qtype,
        )
        for w in warnings:
            logger.warning("[%s][%s] %s", chapter_pack.chapter, qtype, w)
        return valid

    def _build_type_prompt(
        self, chapter: str, qtype: str, count: int, unit_ref: str
    ) -> str:
        """构建针对单一题型的用户 prompt。"""
        type_desc = {
            "concept": "概念解释题（\"什么是...\"、\"说明...的定义/含义/几何意义\"）",
            "calculation": "计算题（\"求...\"、\"计算...\"、\"解...\"）",
            "theorem": "定理/公式/推导题（\"叙述...定理\"、\"推导...公式\"、\"证明...\"）",
            "comparison": "辨析比较题（\"...和...有什么区别/联系\"）",
            "application": "应用题（\"用...解决...实际问题\"）",
            "unanswerable": "不可回答题（问题与教材相关，但提供的片段无法回答）",
        }.get(qtype, qtype)

        lines = [
            f"请为以下教材章节生成 {count} 道 {type_desc}。",
            "",
            f"要求：必须生成恰好 {count} 道题，输出 JSON 数组长度必须为 {count}。",
            "",
            "教材片段如下：",
            "",
            unit_ref,
        ]
        return "\n".join(lines)

    def _validate_questions(
        self,
        questions: list[Any],
        valid_uuids: set[str],
        valid_kp_paths: set[str],
        chapter: str,
        expected_type: str | None = None,
    ) -> tuple[list[dict], list[str]]:
        """校验并清理 LLM 输出。"""
        valid: list[dict] = []
        warnings: list[str] = []
        required_fields = [
            "question",
            "answer",
            "ground_truth_contexts",
            "question_type",
            "kp_path",
        ]

        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                warnings.append(f"Q{i}: 非对象，跳过")
                continue

            missing = [f for f in required_fields if f not in q or not q[f]]
            if missing:
                warnings.append(f"Q{i}: 缺少字段 {missing}，跳过")
                continue

            qtype = q["question_type"]
            if qtype not in VALID_TYPES:
                warnings.append(f"Q{i}: 未知题型 {qtype}，跳过")
                continue

            if expected_type and qtype != expected_type:
                warnings.append(f"Q{i}: 题型不匹配（期望 {expected_type}，实际 {qtype}），跳过")
                continue

            invalid_uuids = [
                uid for uid in q["ground_truth_contexts"] if uid not in valid_uuids
            ]
            if invalid_uuids:
                warnings.append(f"Q{i}: UUID 不存在 {invalid_uuids}，已移除")
                q["ground_truth_contexts"] = [
                    uid for uid in q["ground_truth_contexts"] if uid in valid_uuids
                ]
            if not q["ground_truth_contexts"] and qtype != "unanswerable":
                warnings.append(f"Q{i}: 无可引用 UUID 且非不可答题，跳过")
                continue

            kp_path = q["kp_path"]
            if kp_path not in valid_kp_paths:
                warnings.append(f"Q{i}: kp_path {kp_path} 不在本章节，保留但需人工校验")

            q.setdefault("unanswerable", qtype == "unanswerable")
            valid.append(q)

        return valid, warnings

    # ── 批量生成 ──────────────────────────────────────────────

    async def generate(
        self,
        exported_units_path: str | Path,
        course_id: str,
        document_id: str,
        max_concurrency: int = 5,
    ) -> list[dict]:
        """生成完整候选数据集。

        按章节+题型并发调用 LLM，精确控制题型分布。
        """
        packs = self.load_exported_units(exported_units_path)
        if not packs:
            raise ValueError("没有可用的章节素材")

        type_plans = _distribute_types(self.type_quotas, len(packs))
        all_questions: list[dict] = []

        # 构造所有 (chapter, type, count) 任务
        task_items = []
        for idx, pack in enumerate(packs):
            plan = type_plans[idx]
            for qtype, count in plan.items():
                if count <= 0:
                    continue
                task_items.append((pack, qtype, count))

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_with_limit(pack, qtype, count):
            async with semaphore:
                return await self.generate_for_type(pack, qtype, count)

        tasks = [_run_with_limit(p, t, c) for p, t, c in task_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (pack, qtype, count), result in zip(task_items, results):
            chapter = pack.chapter
            if isinstance(result, Exception):
                logger.error("[%s][%s] 生成失败: %s", chapter, qtype, result)
                continue

            # 注入元数据
            for q in result:
                q["course_id"] = course_id
                q["document_id"] = document_id
                q["unanswerable"] = q.get("question_type") == "unanswerable"

            all_questions.extend(result)
            logger.info(
                "[%s][%s] 生成 %d/%d 道，累计 %d 道",
                chapter,
                qtype,
                len(result),
                count,
                len(all_questions),
            )

        return all_questions

    # ── 后处理：按题型重平衡 ──────────────────────────────────

    def rebalance(
        self,
        questions: list[dict],
        target_quota: dict[str, int] | None = None,
    ) -> list[dict]:
        """按目标配额截断/提示各题型数量（不改变内容，仅排序和截断）。"""
        target = target_quota or self.type_quotas
        by_type: dict[str, list[dict]] = {t: [] for t in VALID_TYPES}
        for q in questions:
            by_type.setdefault(q.get("question_type", "concept"), []).append(q)

        result: list[dict] = []
        for qtype in VALID_TYPES:
            limit = target.get(qtype, 0)
            result.extend(by_type.get(qtype, [])[:limit])

        logger.info(
            "重平衡后: %s",
            {t: sum(1 for q in result if q["question_type"] == t) for t in VALID_TYPES},
        )
        return result


# ── CLI 辅助 ───────────────────────────────────────────────────────


def print_distribution(questions: list[dict]) -> None:
    """打印题型分布。"""
    counts = Counter(q.get("question_type", "unknown") for q in questions)
    print("\n问题类型分布:")
    for t in VALID_TYPES:
        print(f"  {t}: {counts.get(t, 0)} 道")
    print(f"  总计: {len(questions)} 道")
