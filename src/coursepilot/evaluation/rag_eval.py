"""
RAGAS 评估核心 —— 基于 RAGAS 库计算 8 项指标。

指标：
- Context Recall: ground_truth_contexts(UUID) 被检索到的比例（客观指标）
- Context Precision: RAGAS LLMContextPrecisionWithReference
- Context Entity Recall: RAGAS ContextEntityRecall
- Faithfulness: RAGAS Faithfulness
- Answer Relevancy: RAGAS AnswerRelevancy
- Answer Correctness: RAGAS AnswerCorrectness
- Answer Similarity: RAGAS AnswerSimilarity
- Aspect Critique: RAGAS AspectCritic（conciseness）

用法：
    evaluator = RAGEvaluator()
    async with get_session_etx() as session:
        report = await evaluator.evaluate_dataset(session, "eval/questions/20260726/eval_questions.json")
    print(report.summary())
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# 兼容性修复（参考 tests/rag/test_ragas.py）
# ═══════════════════════════════════════════════════════════════
# ragas 0.4.3 在 llms/base.py 中导入了
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# 但 langchain-community >= 0.3.0 已将该模块独立到 langchain-google-vertexai 包中。
# 这里在 ragas 导入前注册一个存根模块，避免 ModuleNotFoundError。
import sys
from types import ModuleType

try:
    import langchain_community.chat_models  # noqa: F401

    _parent_file = getattr(langchain_community.chat_models, "__file__", None)
except ImportError:
    _parent_file = None

_chat_vertexai = ModuleType("langchain_community.chat_models.vertexai")
_chat_vertexai.__path__ = [_parent_file or ""]
_chat_vertexai.__file__ = _parent_file or __file__


class ChatVertexAIStub:
    """存根类，仅用于通过 ragas 的模块加载检查，不会在评估中实际使用。"""

    pass


_chat_vertexai.ChatVertexAI = ChatVertexAIStub
sys.modules["langchain_community.chat_models.vertexai"] = _chat_vertexai
# ═══════════════════════════════════════════════════════════════

import asyncio
import httpx
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import openai
from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate, RunConfig
from ragas.embeddings import _LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    _AnswerCorrectness,
    _AnswerRelevancy,
    _AnswerSimilarity,
    _AspectCritic,
    _ContextEntityRecall,
    _Faithfulness,
    _LLMContextPrecisionWithReference,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.models import KnowledgeUnit
from coursepilot.rag.config import config as rag_config
from coursepilot.rag.generator import Generator, build_course_context
from coursepilot.rag.retriever import Retriever

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """单道题的评估结果（RAGAS 8 项指标）"""

    question: str
    question_type: str
    kp_path: str

    retrieved_uuids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    ground_truth_uuids: list[str] = field(default_factory=list)

    answer: str = ""
    ground_truth: str = ""

    # 检索层
    context_recall: float = 0.0
    context_precision: float = 0.0
    context_entity_recall: float = 0.0

    # 生成层
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0
    answer_similarity: float = 0.0
    aspect_critique: float = 0.0

    context_length: int = 0

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
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)

    def _avg(self, getter) -> float:
        valid = [r for r in self.results if r.question_type != "unanswerable"]
        return _safe_mean(getter(r) for r in valid)

    @property
    def avg_context_recall(self) -> float:
        return self._avg(lambda r: r.context_recall)

    @property
    def avg_context_precision(self) -> float:
        return self._avg(lambda r: r.context_precision)

    @property
    def avg_context_entity_recall(self) -> float:
        return self._avg(lambda r: r.context_entity_recall)

    @property
    def avg_faithfulness(self) -> float:
        return self._avg(lambda r: r.faithfulness)

    @property
    def avg_answer_relevancy(self) -> float:
        return self._avg(lambda r: r.answer_relevancy)

    @property
    def avg_answer_correctness(self) -> float:
        return self._avg(lambda r: r.answer_correctness)

    @property
    def avg_answer_similarity(self) -> float:
        return self._avg(lambda r: r.answer_similarity)

    @property
    def avg_aspect_critique(self) -> float:
        return self._avg(lambda r: r.aspect_critique)

    @property
    def avg_context_length(self) -> float:
        return self._avg(lambda r: r.context_length)

    @property
    def averages(self) -> dict:
        """与 to_dict()['averages'] 一致的字典，便于门禁脚本直接使用。"""
        return {
            "context_recall": self.avg_context_recall,
            "context_precision": self.avg_context_precision,
            "context_entity_recall": self.avg_context_entity_recall,
            "faithfulness": self.avg_faithfulness,
            "answer_relevancy": self.avg_answer_relevancy,
            "answer_correctness": self.avg_answer_correctness,
            "answer_similarity": self.avg_answer_similarity,
            "aspect_critique": self.avg_aspect_critique,
            "context_length": self.avg_context_length,
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "RAGAS 评估报告",
            "=" * 70,
            f"题目数: {self.count}",
            f"错误数: {self.error_count}",
            f"总耗时: {self.elapsed_seconds:.0f}s",
            f"配置:   {json.dumps(self.config, ensure_ascii=False)}",
            "",
            "检索层:",
            f"  Context Recall:           {self.avg_context_recall:.3f}  {'[PASS]' if self.avg_context_recall >= 0.85 else '[FAIL]'}",
            f"  Context Precision:        {self.avg_context_precision:.3f}",
            f"  Context Entity Recall:    {self.avg_context_entity_recall:.3f}",
            "",
            "生成层:",
            f"  Faithfulness:             {self.avg_faithfulness:.3f}",
            f"  Answer Relevancy:         {self.avg_answer_relevancy:.3f}",
            f"  Answer Correctness:       {self.avg_answer_correctness:.3f}",
            f"  Answer Similarity:        {self.avg_answer_similarity:.3f}",
            f"  Aspect Critique(简洁性):  {self.avg_aspect_critique:.3f}",
            "",
            self._per_question_table(),
            "=" * 70,
        ]
        return "\n".join(lines)

    def _per_question_table(self) -> str:
        header = (
            f"  {'#':<3} {'类型':<10} {'Recall':<7} {'Prec':<7} "
            f"{'Ent':<7} {'Faith':<7} {'Relev':<7} {'Corr':<7} {'Sim':<7} {'Crit':<7} 问题"
        )
        rows = [header, "  " + "-" * (len(header) - 2)]
        for i, r in enumerate(self.results):
            q = r.question[:45]
            rows.append(
                f"  {i+1:<3} {r.question_type:<10} "
                f"{r.context_recall:<7.3f} {r.context_precision:<7.3f} "
                f"{r.context_entity_recall:<7.3f} {r.faithfulness:<7.3f} "
                f"{r.answer_relevancy:<7.3f} {r.answer_correctness:<7.3f} "
                f"{r.answer_similarity:<7.3f} {r.aspect_critique:<7.3f} "
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
                "context_entity_recall": self.avg_context_entity_recall,
                "faithfulness": self.avg_faithfulness,
                "answer_relevancy": self.avg_answer_relevancy,
                "answer_correctness": self.avg_answer_correctness,
                "answer_similarity": self.avg_answer_similarity,
                "aspect_critique": self.avg_aspect_critique,
                "context_length": self.avg_context_length,
            },
            "results": [
                {
                    "question": r.question,
                    "question_type": r.question_type,
                    "kp_path": r.kp_path,
                    "context_recall": r.context_recall,
                    "context_precision": r.context_precision,
                    "context_entity_recall": r.context_entity_recall,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "answer_correctness": r.answer_correctness,
                    "answer_similarity": r.answer_similarity,
                    "aspect_critique": r.aspect_critique,
                    "context_length": r.context_length,
                    "retrieved_uuids": r.retrieved_uuids,
                    "ground_truth_uuids": r.ground_truth_uuids,
                    "query_rewritten": r.query_rewritten,
                    "top_kp_paths": r.top_kp_paths,
                    "candidate_count": r.candidate_count,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def _safe_mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ═══════════════════════════════════════════════════════════════
# 核心评估器
# ═══════════════════════════════════════════════════════════════

class RAGEvaluator:
    """RAGAS 评估器。

    对黄金数据集逐题执行检索+生成，使用 RAGAS 库计算 8 项指标。
    支持通过 config_overrides 临时覆盖 RAG 参数（用于网格搜索）。

    所有重型对象（LLM client、Retriever、Generator、RAGAS 客户端）
    在构造时创建并复用，避免每次调用新建连接导致 Milvus/API 连接爆炸。
    """

    def __init__(
        self,
        config_overrides: dict | None = None,
        *,
        use_mimo: bool = True,
        ragas_max_tokens: int = 4096,
        ragas_timeout: float = 120.0,
        ragas_max_workers: int = 4,       # 从 2 提升到 4（HTTP/1.1 + 无 keepalive）
    ):
        self.overrides = config_overrides or {}
        self._original_config = {}
        self.use_mimo = use_mimo
        self.ragas_max_tokens = ragas_max_tokens
        self.ragas_timeout = ragas_timeout
        self.ragas_max_workers = ragas_max_workers

        # 复用单例，避免重复创建连接
        self.retriever = Retriever()
        self.generator = Generator()
        self.llm_client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        # RAGAS 评估客户端（延迟初始化，仅 rag-only=False 时加载）
        self._ragas_llm = None
        self._ragas_embeddings = None
        self._metrics = []
        self._ragas_inited = False

    def _ensure_ragas_clients(self) -> None:
        """确保 RAGAS 客户端已初始化（惰性加载）。"""
        if self._ragas_inited:
            return
        self._init_ragas_clients()
        self._ragas_inited = True

    def _init_ragas_clients(self) -> None:
        """初始化 RAGAS 所需的 LLM 与 Embeddings 客户端。"""
        if not self.use_mimo:
            print("[eval] 使用默认 LLM 作为 RAGAS judge 模型")
            api_key = settings.llm_api_key
            base_url = settings.llm_base_url
            model = settings.llm_model
        else:
            api_key = settings.mimo_api_key
            base_url = settings.mimo_base_url
            # 改用 mimo-v2.5（非 pro）控制成本
            model = "mimo-v2.5"
            print(f"[eval] 使用 MiMO 作为 RAGAS judge 模型: {model}")

        if not api_key or not base_url or not model:
            raise ValueError(
                "RAGAS judge 配置不完整，请检查 .env 中的 "
                "MIMO_API_KEY / MIMO_BASE_URL / MIMO_MODEL（或 LLM 对应配置）"
            )

        # ── 创建 RAGAS LLM client ────────────────────────────
        # 彻底关闭 keepalive（max_keepalive_connections=0），每次请求新建连接，
        # 避免本地代理（127.0.0.1:13330）的 gRPC keepalive 触发 too_many_pings。
        # trust_env=False 跳过 HTTP_PROXY/HTTPS_PROXY 环境变量，尝试直连。
        transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_keepalive_connections=0,
                keepalive_expiry=0,
            ),
            trust_env=False,
        )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(self.ragas_timeout, connect=30.0),
            max_retries=3,
            http_client=httpx.AsyncClient(
                transport=transport,
                http1=True,
                http2=False,         # 强制 HTTP/1.1，无 gRPC
            ),
        )

        # RAGAS 0.4.3 的 llm_factory 接受 openai client，返回 InstructorLLM
        self._ragas_llm = llm_factory(
            model, client=client,
            max_tokens=self.ragas_max_tokens,
        )

        # embeddings：默认使用本地 BGE-M3，避免依赖 MiMO 的 embedding 接口
        # 因为实测 MiMO 对 /embeddings 返回 404。
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            lc_emb = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._ragas_embeddings = _LangchainEmbeddingsWrapper(lc_emb)
            print(
                f"[eval] RAGAS embeddings 使用本地模型: {settings.embedding_model_path}"
            )
        except Exception as e:
            print(f"[eval] [WARN] RAGAS embeddings 初始化失败: {e}")
            self._ragas_embeddings = None

        self._metrics = self._build_metrics()

    def _build_metrics(self) -> list:
        """构造 RAGAS 官方评估指标列表。"""
        metrics = [
            _Faithfulness(llm=self._ragas_llm),
            _LLMContextPrecisionWithReference(llm=self._ragas_llm),
            _ContextEntityRecall(llm=self._ragas_llm),
        ]

        if self._ragas_embeddings:
            metrics.extend(
                [
                    _AnswerRelevancy(
                        llm=self._ragas_llm, embeddings=self._ragas_embeddings
                    ),
                    _AnswerCorrectness(
                        llm=self._ragas_llm, embeddings=self._ragas_embeddings
                    ),
                    _AnswerSimilarity(embeddings=self._ragas_embeddings),
                ]
            )
        else:
            print("[eval] [WARN] 无 embeddings，跳过 AnswerRelevancy/AnswerCorrectness/AnswerSimilarity")

        metrics.append(
            _AspectCritic(
                name="conciseness",
                definition="答案是否简洁，无冗余信息",
                llm=self._ragas_llm,
            )
        )
        return metrics

    def _apply_overrides(self):
        """临时覆盖 RAG 配置参数，保存原始值用于恢复。"""
        self._original_config = {}
        for key, value in self.overrides.items():
            if hasattr(rag_config, key):
                self._original_config[key] = getattr(rag_config, key)
                setattr(rag_config, key, value)

    def _restore_config(self):
        """恢复被覆盖的 RAG 配置参数。"""
        for key, value in self._original_config.items():
            setattr(rag_config, key, value)

    # ═══════════════════════════════════════════════════════════
    # 检索 + UUID 收集
    # ═══════════════════════════════════════════════════════════

    async def _retrieve_with_uuids(
        self,
        session: AsyncSession,
        query: str,
        course_id: str,
    ) -> tuple[str, dict, list[str]]:
        """执行检索并返回 (context, metadata, all_retrieved_uuids)。

        all_retrieved_uuids 取 metadata 中的 context_uuids（精准反映最终上下文内容），
        不再回查 DB 中对应 KP 下的全部 unit，确保 baseline / kp_full / kp_neighbor
        三种策略的 recall 计算准确可比。
        """
        context, metadata = await self.retriever.retrieve(session, query, course_id)
        all_uuids = metadata.get("context_uuids", [])
        return context, metadata, all_uuids

    # ═══════════════════════════════════════════════════════════
    # 单题准备
    # ═══════════════════════════════════════════════════════════

    async def _prepare_result(
        self,
        session: AsyncSession,
        question: dict,
        course_id: str,
        *,
        skip_generation: bool = False,
        _cached_course_context: dict | None = None,
    ) -> EvalResult:
        """准备单道题的 EvalResult（检索 + 生成，不含 RAGAS 打分）。"""
        t0 = time.monotonic()

        gt_uuids = question.get("ground_truth_contexts", [])

        result = EvalResult(
            question=question["question"],
            question_type=question.get("question_type", ""),
            kp_path=question.get("kp_path", ""),
            ground_truth_uuids=gt_uuids,
            ground_truth=question.get("ground_truth", question.get("answer", "")),
        )

        try:
            self._apply_overrides()

            # 检测不可回答问题：不检索、不生成，直接返回标准答案
            if question.get("question_type") == "unanswerable" or question.get("unanswerable"):
                print("[eval] 检测到不可回答问题，跳过检索与生成")
                result.answer = question.get("ground_truth",
                                              question.get("answer", "教材未涉及此内容，无法回答"))
                result.latency_ms = (time.monotonic() - t0) * 1000
                return result

            # 1) 检索
            context, metadata, all_uuids = await self._retrieve_with_uuids(
                session, question["question"], course_id
            )
            result.retrieved_uuids = all_uuids
            result.retrieved_contexts = await self._load_units(session, all_uuids)
            result.query_rewritten = metadata.get("query_rewritten", "")
            result.top_kp_paths = metadata.get("source_kp_paths", [])
            result.candidate_count = metadata.get("candidate_count", 0)

            # 2) Context Recall（基于 UUID 的客观指标）
            result.context_recall = self._compute_context_recall(
                gt_uuids, all_uuids
            )
            result.context_length = len(context)

            # 3) 生成回答（可选跳过，仅评估检索质量时）
            if not skip_generation and context:
                print("[eval] 调用 LLM 生成回答...")
                t_gen = time.monotonic()
                if _cached_course_context is not None:
                    course_context = _cached_course_context
                else:
                    course_context = await build_course_context(session, course_id)
                result.answer, _ = await self.generator.generate(
                    question["question"], context, course_context
                )
                print(
                    f"[eval] LLM 生成完成, 耗时={(time.monotonic()-t_gen)*1000:.0f}ms, "
                    f"answer_chars={len(result.answer)}"
                )
            elif not context:
                result.answer = ""
                result.error = "检索上下文为空"

        except Exception as e:
            result.error = str(e)
            logger.error(
                "准备题目失败 [%s...]: %s", question["question"][:50], e
            )

        finally:
            self._restore_config()

        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    async def evaluate_single(
        self,
        session: AsyncSession,
        question: dict,
        course_id: str,
        *,
        skip_generation: bool = False,
    ) -> EvalResult:
        """评估单道题，返回 EvalResult。"""
        print(f"\n{'='*40}\n[eval] 单题评估: {question['question'][:60]}\n{'='*40}")
        result = await self._prepare_result(
            session, question, course_id, skip_generation=skip_generation
        )
        scores = (await self._run_ragas([result]))[0]
        self._merge_ragas_scores(result, scores)
        print(
            f"[eval] 单题完成: recall={result.context_recall:.3f}, "
            f"faithfulness={result.faithfulness:.3f}, "
            f"latency={result.latency_ms:.0f}ms, error={result.error!r}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 批量评估
    # ═══════════════════════════════════════════════════════════

    async def evaluate_dataset(
        self,
        session: AsyncSession,
        dataset_path: str | Path,
        *,
        skip_generation: bool = False,
        skip_ragas: bool = False,
    ) -> EvalReport:
        """加载黄金数据集并逐题评估。

        Args:
            skip_generation: 跳过 LLM 生成（同时自动跳过 RAGAS 评分）
            skip_ragas: 执行 LLM 生成但跳过 RAGAS 评分（仅保存问答结果）
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {path}")

        questions = json.loads(path.read_text(encoding="utf-8"))
        if not questions:
            raise ValueError("数据集为空")

        course_id = questions[0].get("course_id", "")
        if not course_id:
            raise ValueError("数据集中缺少 course_id")

        # skip_generation 时强制跳过 RAGAS
        if skip_generation:
            skip_ragas = True

        print(f"[eval] 加载数据集: {path}, 共 {len(questions)} 题")
        print(f"[eval] 跳过生成: {skip_generation}, 跳过 RAGAS: {skip_ragas}")

        t0 = time.monotonic()
        results: list[EvalResult] = []

        # 缓存 course_context，所有题共享（同一 course_id）
        course_context_cached = await build_course_context(session, course_id) if not skip_generation else {}
        print(f"[eval] 课程上下文已缓存: {list(course_context_cached.keys()) if course_context_cached else '{}'}")

        for i, q in enumerate(questions):
            print(
                f"\n{'='*40}\n[eval] [{i+1}/{len(questions)}] {q['question'][:60]}\n{'='*40}"
            )
            logger.info("[%d/%d] 评估: %s", i + 1, len(questions), q["question"][:60])
            r = await self._prepare_result(
                session, q, course_id, skip_generation=skip_generation,
                _cached_course_context=course_context_cached,
            )
            results.append(r)
            print(
                f"[eval] 第{i+1}题准备完成: recall={r.context_recall:.3f}, "
                f"latency={r.latency_ms:.0f}ms, error={r.error!r}"
            )

        # 批量 RAGAS 评分（仅非 rag-only 模式）
        if not skip_ragas:
            print("\n[eval] 开始批量 RAGAS 评分...")
            t_ragas = time.monotonic()
            all_scores = await self._run_ragas(results)
            for r, scores in zip(results, all_scores):
                self._merge_ragas_scores(r, scores)
            print(
                f"[eval] RAGAS 评分完成, 耗时={(time.monotonic()-t_ragas)*1000:.0f}ms"
            )
        else:
            print("[eval] 跳过 RAGAS 评分（rag-only 模式）")

        elapsed = time.monotonic() - t0
        report = EvalReport(
            results=results,
            config=self._current_config_snapshot(),
            elapsed_seconds=elapsed,
        )
        return report

    # ═══════════════════════════════════════════════════════════
    # RAGAS 评分
    # ═══════════════════════════════════════════════════════════

    async def _run_ragas(self, results: list[EvalResult]) -> list[dict]:
        """对一批 EvalResult 执行 RAGAS 批量评分。

        返回与 results 等长的 list[dict]，每个 dict 包含 7 项 RAGAS 指标分数；
        无法评估的题目对应空 dict。
        """
        rows: list[dict] = []
        valid_indices: list[int] = []

        for i, r in enumerate(results):
            if r.error or not r.retrieved_contexts:
                continue
            rows.append(
                {
                    "question": r.question,
                    "contexts": r.retrieved_contexts,
                    "answer": r.answer or "",
                    "ground_truth": r.ground_truth,
                }
            )
            valid_indices.append(i)

        if not rows:
            print("[eval] [WARN] 没有有效题目可供 RAGAS 评分")
            return [{} for _ in results]

        # 惰性加载 RAGAS 客户端（加载模型可能需要时间）
        self._ensure_ragas_clients()

        dataset = Dataset.from_list(rows)
        print(f"[eval] RAGAS 输入: {len(rows)} 题, metrics={len(self._metrics)}")

        try:
            # 降低并发 worker 数，减少 HTTP/2 ping 频率和服务端限流概率
            run_config = RunConfig(
                max_workers=self.ragas_max_workers,
                timeout=self.ragas_timeout,
            )
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )
            df = ragas_result.to_pandas()
        except Exception as e:
            logger.error("RAGAS 批量评估失败: %s", e)
            print(f"[eval] [ERROR] RAGAS 批量评估失败: {e}")
            return [{} for _ in results]

        scores_list: list[dict] = []
        for _, row in df.iterrows():
            scores_list.append(
                {
                    "context_precision": _to_float(
                        row.get("llm_context_precision_with_reference")
                    ),
                    "context_entity_recall": _to_float(
                        row.get("context_entity_recall")
                    ),
                    "faithfulness": _to_float(row.get("faithfulness")),
                    "answer_relevancy": _to_float(row.get("answer_relevancy")),
                    "answer_correctness": _to_float(row.get("answer_correctness")),
                    "answer_similarity": _to_float(row.get("answer_similarity")),
                    "aspect_critique": _to_float(row.get("conciseness")),
                }
            )

        final_scores: list[dict] = [{} for _ in results]
        for idx, scores in zip(valid_indices, scores_list):
            final_scores[idx] = scores

        return final_scores

    @staticmethod
    def _merge_ragas_scores(result: EvalResult, scores: dict) -> None:
        """将 RAGAS 分数合并到 EvalResult。"""
        if not scores:
            return
        result.context_precision = scores.get("context_precision", 0.0)
        result.context_entity_recall = scores.get("context_entity_recall", 0.0)
        result.faithfulness = scores.get("faithfulness", 0.0)
        result.answer_relevancy = scores.get("answer_relevancy", 0.0)
        result.answer_correctness = scores.get("answer_correctness", 0.0)
        result.answer_similarity = scores.get("answer_similarity", 0.0)
        result.aspect_critique = scores.get("aspect_critique", 0.0)



    @staticmethod
    def _compute_context_recall(
        ground_truth: list[str], retrieved: list[str]
    ) -> float:
        """Context Recall = |ground_truth ∩ retrieved| / |ground_truth|。"""
        if not ground_truth:
            return 1.0
        gt_set = set(ground_truth)
        ret_set = set(retrieved)
        return len(gt_set & ret_set) / len(gt_set)

    # ═══════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    async def _load_units(
        session: AsyncSession, uuids: list[str]
    ) -> list[str]:
        """按 UUID 批量加载 KnowledgeUnit 内容。"""
        if not uuids:
            return []
        result = await session.execute(
            select(KnowledgeUnit.content).where(
                KnowledgeUnit.id.in_([UUID(uid) for uid in uuids])
            )
        )
        return [row[0] for row in result.all() if row[0]]

    def _current_config_snapshot(self) -> dict:
        """当前 RAG 配置快照（读取 rag_config 当前值 + 合并实际生效的 overrides）。

        注意：快照在 _restore_config() 之后采集，所以 rag_config 已被复原。
        必须将 self.overrides 合并上去，才能反映本次评估实际使用的参数。
        """
        snapshot = {
            "rrf_k": rag_config.rrf_k,
            "dense_weight": getattr(rag_config, "dense_weight", None),
            "rerank_top_k": rag_config.rerank_top_k,
            "context_max_chars": rag_config.context_max_chars,
            "dense_top_k": rag_config.dense_top_k,
            "sparse_top_k": rag_config.sparse_top_k,
            "enable_rewrite": rag_config.enable_rewrite,
            "enable_sparse": rag_config.enable_sparse,
            "enable_rerank": rag_config.enable_rerank,
            "enable_kp_expand": rag_config.enable_kp_expand,
            "kp_expand_mode": getattr(rag_config, "kp_expand_mode", "full"),
            "kp_neighbor_window": getattr(rag_config, "kp_neighbor_window", 2),
        }
        # 合并实际生效的 overrides（覆盖默认值）
        snapshot.update(self.overrides)
        return snapshot


def _to_float(value) -> float:
    """将 RAGAS 输出值安全转换为 0-1 浮点数。"""
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    return max(0.0, min(1.0, f))
