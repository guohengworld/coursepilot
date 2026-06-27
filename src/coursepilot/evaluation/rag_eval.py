"""
RAGAS 风格评估 —— 四大指标 + LLM-as-Judge + 参数网格搜索

指标：
- Context Recall: 多少 ground-truth UUID 被检索到（确定性，评估门禁指标）
- Context Precision: 检索到的上下文中多少真正相关（LLM-as-Judge）
- Faithfulness: 生成的回答是否忠实于上下文（LLM-as-Judge）
- Answer Relevancy: 回答是否切题（LLM-as-Judge）

用法：
    evaluator = RAGEvaluator()
    async with get_session_etx() as session:
        report = await evaluator.evaluate_dataset(session, "eval/questions/eval_questions.json")
    print(report.summary())
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import KnowledgeUnit
from coursepilot.rag.config import config as rag_config
from coursepilot.rag.generator import Generator, build_course_context
from coursepilot.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ── LLM-as-Judge prompts ──────────────────────────────────────────

JUDGE_CONTEXT_PRECISION_PROMPT = """你是一个严格的评估者。给定一个问题和一段检索到的教材内容，判断这段内容是否对回答该问题有帮助。

问题：{question}

检索到的教材内容：
{context}

这段内容是否包含可用于回答该问题的信息？只回答 "yes" 或 "no"。"""

JUDGE_FAITHFULNESS_PROMPT = """你是一个严格的评估者。给定一段检索到的教材上下文和一个基于该上下文生成的回答，判断回答中的每个陈述是否都能从上下文中找到依据。

上下文：
{context}

回答：
{answer}

请将回答分解为原子性陈述（atomic claims），然后逐一判断每个陈述是否能在上下文中找到支持依据。
以 JSON 数组格式输出：
```json
[
  {{"claim": "陈述1", "supported": true/false}},
  {{"claim": "陈述2", "supported": true/false}}
]
```

只输出 JSON 数组，不要添加其他文字。"""

JUDGE_ANSWER_RELEVANCY_PROMPT = """你是一个严格的评估者。判断以下回答是否直接且完整地回应了给定的问题。

问题：{question}

回答：{answer}

请从以下维度评分（0-1）：
- 是否直接回应了问题（而非答非所问）
- 是否覆盖了问题的核心要点
- 是否包含不相关的内容

给出一个 0 到 1 之间的分数，只输出数字。"""


# ── Data structures ────────────────────────────────────────────────

@dataclass
class EvalResult:
    """单道题的评估结果"""
    question: str
    question_type: str
    kp_path: str

    retrieved_uuids: list[str] = field(default_factory=list)
    ground_truth_uuids: list[str] = field(default_factory=list)

    answer: str = ""

    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0

    query_rewritten: str = ""
    top_kp_paths: list[str] = field(default_factory=list)
    candidate_count: int = 0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class EvalReport:
    """批量评估报告"""
    results: list[EvalResult]
    config: dict
    elapsed_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def avg_context_recall(self) -> float:
        return _safe_mean(r.context_recall for r in self.results)

    @property
    def avg_context_precision(self) -> float:
        return _safe_mean(r.context_precision for r in self.results)

    @property
    def avg_faithfulness(self) -> float:
        return _safe_mean(r.faithfulness for r in self.results)

    @property
    def avg_answer_relevancy(self) -> float:
        return _safe_mean(r.answer_relevancy for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RAGAS 评估报告",
            "=" * 60,
            f"题目数: {self.count}",
            f"错误数: {self.error_count}",
            f"总耗时: {self.elapsed_seconds:.0f}s",
            f"配置:   {json.dumps(self.config, ensure_ascii=False)}",
            "",
            f"  Context Recall:      {self.avg_context_recall:.3f}  {'[PASS]' if self.avg_context_recall >= 0.85 else '[FAIL]'}",
            f"  Context Precision:   {self.avg_context_precision:.3f}",
            f"  Faithfulness:        {self.avg_faithfulness:.3f}",
            f"  Answer Relevancy:    {self.avg_answer_relevancy:.3f}",
            "",
            self._per_question_table(),
            "=" * 60,
        ]
        return "\n".join(lines)

    def _per_question_table(self) -> str:
        header = f"  {'#':<3} {'类型':<10} {'Recall':<8} {'Prec':<8} {'Faith':<8} {'Relev':<8} 问题"
        rows = [header, "  " + "-" * (len(header) - 2)]
        for i, r in enumerate(self.results):
            q = r.question[:50]
            rows.append(
                f"  {i+1:<3} {r.question_type:<10} "
                f"{r.context_recall:<8.3f} {r.context_precision:<8.3f} "
                f"{r.faithfulness:<8.3f} {r.answer_relevancy:<8.3f} "
                f"{q}"
            )
            if r.error:
                rows.append(f"       [ERROR] {r.error[:80]}")
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "elapsed_seconds": self.elapsed_seconds,
            "averages": {
                "context_recall": self.avg_context_recall,
                "context_precision": self.avg_context_precision,
                "faithfulness": self.avg_faithfulness,
                "answer_relevancy": self.avg_answer_relevancy,
            },
            "results": [
                {
                    "question": r.question,
                    "question_type": r.question_type,
                    "kp_path": r.kp_path,
                    "context_recall": r.context_recall,
                    "context_precision": r.context_precision,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def _safe_mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ── Core evaluator ─────────────────────────────────────────────────

class RAGEvaluator:
    """RAGAS 风格评估器

    对黄金数据集逐题执行检索+生成，计算四大指标。
    支持通过 config_overrides 临时覆盖 RAG 参数（用于网格搜索）。

    所有重型对象（LLM client、Retriever、Generator）在构造时创建并复用，
    避免每次调用新建连接导致 Milvus/API 连接爆炸。
    """

    def __init__(self, config_overrides: dict | None = None):
        self.overrides = config_overrides or {}
        self._original_config = {}

        # 复用单例，避免重复创建连接
        self.retriever = Retriever()
        self.generator = Generator()
        self.llm_client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def _apply_overrides(self):
        """临时覆盖 RAG 配置参数，保存原始值用于恢复"""
        self._original_config = {}
        for key, value in self.overrides.items():
            if hasattr(rag_config, key):
                self._original_config[key] = getattr(rag_config, key)
                setattr(rag_config, key, value)

    def _restore_config(self):
        """恢复被覆盖的 RAG 配置参数"""
        for key, value in self._original_config.items():
            setattr(rag_config, key, value)

    # ── 检索 + UUID 收集 ──────────────────────────────────────

    async def _retrieve_with_uuids(
        self,
        session: AsyncSession,
        query: str,
        course_id: str,
    ) -> tuple[str, dict, list[str]]:
        """执行检索并返回 (context, metadata, all_retrieved_uuids)

        all_retrieved_uuids 包含 KP 扩展后最终上下文中的全部 unit UUID，
        用于与 ground_truth_contexts 对比计算 Context Recall。
        """
        context, metadata = await self.retriever.retrieve(
            session, query, course_id
        )

        # 收集 KP 扩展后的全部 UUID
        top_kp_ids = metadata.get("top_kp_ids", [])
        all_uuids: list[str] = []
        if top_kp_ids:
            result = await session.execute(
                select(KnowledgeUnit.id)
                .where(KnowledgeUnit.kp_id.in_([UUID(k) for k in top_kp_ids]))
            )
            all_uuids = [str(r[0]) for r in result.all()]

        return context, metadata, all_uuids

    # ── 单题评估 ──────────────────────────────────────────────

    async def evaluate_single(
        self,
        session: AsyncSession,
        question: dict,
        course_id: str,
        *,
        skip_generation: bool = False,
    ) -> EvalResult:
        """评估单道题，返回 EvalResult"""
        t0 = time.monotonic()

        gt_uuids = question.get("ground_truth_contexts", [])

        result = EvalResult(
            question=question["question"],
            question_type=question.get("question_type", ""),
            kp_path=question.get("kp_path", ""),
            ground_truth_uuids=gt_uuids,
        )

        try:
            self._apply_overrides()

            # 1) 检索
            context, metadata, all_uuids = await self._retrieve_with_uuids(
                session, question["question"], course_id
            )
            result.retrieved_uuids = all_uuids
            result.query_rewritten = metadata.get("query_rewritten", "")
            result.top_kp_paths = metadata.get("source_kp_paths", [])
            result.candidate_count = metadata.get("candidate_count", 0)

            # 2) Context Recall（确定性）
            result.context_recall = self._compute_context_recall(
                gt_uuids, all_uuids
            )

            # 3) 生成回答（可选跳过，仅评估检索质量时）
            if not skip_generation and context:
                print("[eval] 调用 LLM 生成回答...")
                t_gen = time.monotonic()
                course_context = await build_course_context(session, course_id)
                result.answer = await self.generator.generate(
                    question["question"], context, course_context
                )
                print(f"[eval] LLM 生成完成, 耗时={(time.monotonic()-t_gen)*1000:.0f}ms, answer_chars={len(result.answer)}")
            elif not context:
                result.answer = ""
                result.error = "检索上下文为空"

            # 4) LLM-as-Judge 指标（有回答时才计算）
            if result.answer and context:
                print("[eval] LLM-as-Judge 评估开始...")
                t_judge = time.monotonic()
                top_kp_ids = metadata.get("top_kp_ids", [])
                result.context_precision = await self._judge_context_precision(
                    question["question"], top_kp_ids, session
                )
                result.faithfulness = await self._judge_faithfulness(
                    result.answer, context
                )
                result.answer_relevancy = await self._judge_answer_relevancy(
                    question["question"], result.answer
                )
                print(f"[eval] LLM-as-Judge 完成, 耗时={(time.monotonic()-t_judge)*1000:.0f}ms")

        except Exception as e:
            result.error = str(e)
            logger.error(
                "评估失败 [%s...]: %s", question["question"][:50], e
            )

        finally:
            self._restore_config()

        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    # ── 批量评估 ──────────────────────────────────────────────

    async def evaluate_dataset(
        self,
        session: AsyncSession,
        dataset_path: str | Path,
        *,
        skip_generation: bool = False,
    ) -> EvalReport:
        """加载黄金数据集并逐题评估"""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {path}")

        questions = json.loads(path.read_text(encoding="utf-8"))
        if not questions:
            raise ValueError("数据集为空")

        course_id = questions[0].get("course_id", "")
        if not course_id:
            raise ValueError("数据集中缺少 course_id")

        t0 = time.monotonic()
        results: list[EvalResult] = []

        for i, q in enumerate(questions):
            print(f"\n{'='*40}\n[eval] [{i+1}/{len(questions)}] {q['question'][:60]}\n{'='*40}")
            logger.info("[%d/%d] 评估: %s", i + 1, len(questions), q["question"][:60])
            r = await self.evaluate_single(
                session, q, course_id, skip_generation=skip_generation
            )
            print(f"[eval] 第{i+1}题完成: recall={r.context_recall:.3f}, latency={r.latency_ms:.0f}ms, error={r.error!r}")
            results.append(r)
            # 题间短暂延迟，避免打爆 Milvus 和 LLM API
            print(f"[eval] 题间休眠 1s...")
            await asyncio.sleep(1.0)

        elapsed = time.monotonic() - t0
        report = EvalReport(
            results=results,
            config=self._current_config_snapshot(),
            elapsed_seconds=elapsed,
        )
        return report

    # ── 四大指标 ──────────────────────────────────────────────

    @staticmethod
    def _compute_context_recall(
        ground_truth: list[str], retrieved: list[str]
    ) -> float:
        """Context Recall = |ground_truth ∩ retrieved| / |ground_truth|"""
        if not ground_truth:
            return 1.0
        gt_set = set(ground_truth)
        ret_set = set(retrieved)
        return len(gt_set & ret_set) / len(gt_set)

    async def _judge_context_precision(
        self, question: str, top_kp_ids: list[str], session: AsyncSession
    ) -> float:
        """LLM-as-Judge: KP 级判定，每 KP 拼接全部 unit 后截断一次 judge"""
        if not top_kp_ids:
            return 0.0

        kp_ids = top_kp_ids[:5]
        relevant_kps = 0

        for kp_id in kp_ids:
            units = await self._load_units_by_kp(session, kp_id)
            if not units:
                continue
            combined = "\n\n".join(units)[:3000]
            try:
                answer = await self._llm_judge(
                    JUDGE_CONTEXT_PRECISION_PROMPT.format(
                        question=question, context=combined
                    ),
                    max_tokens=5,
                )
                if answer.strip().lower().startswith("yes"):
                    relevant_kps += 1
            except Exception:
                relevant_kps += 1

        return relevant_kps / len(kp_ids)

    async def _judge_faithfulness(
        self, answer: str, context: str
    ) -> float:
        """LLM-as-Judge: 回答中的陈述是否能从上下文中找到依据"""
        prompt = JUDGE_FAITHFULNESS_PROMPT.format(
            context=context[:6000], answer=answer
        )
        try:
            raw = await self._llm_judge(prompt, max_tokens=1024)
            claims = _parse_json_array(raw)
            if not claims:
                return 0.0
            supported = sum(1 for c in claims if c.get("supported", False))
            return supported / len(claims)
        except Exception:
            return 0.0

    async def _judge_answer_relevancy(
        self, question: str, answer: str
    ) -> float:
        """LLM-as-Judge: 回答是否切题"""
        prompt = JUDGE_ANSWER_RELEVANCY_PROMPT.format(
            question=question, answer=answer[:2000]
        )
        try:
            raw = await self._llm_judge(prompt, max_tokens=10)
            score = float(raw.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    # ── 工具方法 ──────────────────────────────────────────────

    async def _llm_judge(self, prompt: str, max_tokens: int = 256) -> str:
        """调用 DeepSeek 执行 LLM-as-Judge（复用共享 client）"""
        response = await self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "你是一个严格的评估者。请严格按要求回答。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or ""

    @staticmethod
    async def _load_units(
        session: AsyncSession, uuids: list[str]
    ) -> list[str]:
        """按 UUID 批量加载 KnowledgeUnit 内容"""
        result = await session.execute(
            select(KnowledgeUnit.content)
            .where(KnowledgeUnit.id.in_([UUID(uid) for uid in uuids]))
        )
        return [row[0] for row in result.all() if row[0]]

    @staticmethod
    async def _load_units_by_kp(
        session: AsyncSession, kp_id: str
    ) -> list[str]:
        """按 KP ID 加载全部 unit 内容"""
        result = await session.execute(
            select(KnowledgeUnit.content)
            .where(KnowledgeUnit.kp_id == UUID(kp_id))
        )
        return [row[0] for row in result.all() if row[0]]

    def _current_config_snapshot(self) -> dict:
        """当前 RAG 配置快照（便于报告记录）"""
        return {
            "rrf_k": rag_config.rrf_k,
            "rerank_top_k": rag_config.rerank_top_k,
            "context_max_chars": rag_config.context_max_chars,
            "dense_top_k": rag_config.dense_top_k,
            "sparse_top_k": rag_config.sparse_top_k,
            "enable_rewrite": rag_config.enable_rewrite,
            "enable_sparse": rag_config.enable_sparse,
            "enable_rerank": rag_config.enable_rerank,
            "enable_kp_expand": rag_config.enable_kp_expand,
        }


def _parse_json_array(raw: str) -> list[dict]:
    """从 LLM 回复中提取 JSON 数组"""
    import re

    raw = raw.strip()
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试提取 markdown code block 中的 JSON
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取 [...]
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    logger.warning("无法解析 LLM judge 输出: %.200s", raw)
    return []
