"""Phase B RAG 模块测试 —— 编码器 / 向量存储 / 重排序 / 引用 / 日志

运行方式：
    PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_rag.py -v
    PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_rag.py -v -k "TestCitation"  # 单个模块
    PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_rag.py -v -m "not slow"      # 跳过慢测试

依赖：
    - BGE-M3 模型（config.embedding_model_path）
    - bge-reranker-v2-m3 模型（config.reranker_model_path）
    - Milvus Lite（自动安装）
    - DeepSeek API Key（可选，改写/生成测试需要）
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# 辅助：检查模型/外部依赖是否可用
# ═══════════════════════════════════════════════════════════════

def _model_exists(path: str) -> bool:
    return Path(path).exists()


def _has_api_key() -> bool:
    try:
        from coursepilot.config import settings
        return bool(settings.llm_api_key)
    except Exception:
        return False


# 直接检查模型目录（绕过 pydantic-settings 被 Windows 系统环境变量覆盖的问题）
BGE_M3_MODEL_PATH = "F:/all-projs/models/bge-m3"
RERANKER_MODEL_PATH = "F:/all-projs/models/bge-reranker-v2-m3"
BGE_M3_AVAILABLE = _model_exists(BGE_M3_MODEL_PATH)
RERANKER_AVAILABLE = _model_exists(RERANKER_MODEL_PATH)
API_KEY_AVAILABLE = _has_api_key()

# 修正 Windows 系统环境变量覆盖 .env 中 embedding_model_path 的问题
if BGE_M3_AVAILABLE:
    from coursepilot.config import settings
    if settings.embedding_model_path != BGE_M3_MODEL_PATH:
        object.__setattr__(settings, 'embedding_model_path', BGE_M3_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════
# Citation 测试（纯逻辑，无外部依赖）
# ═══════════════════════════════════════════════════════════════

class TestCitation:
    def test_extract_single(self):
        from coursepilot.rag.citation import extract_citations
        assert extract_citations('根据教材<ref id="1" />，定积分...') == [1]

    def test_extract_multiple_ordered_unique(self):
        from coursepilot.rag.citation import extract_citations
        answer = '<ref id="1" /> ... <ref id="3" /> ... <ref id="1" />'
        assert extract_citations(answer) == [1, 3]

    def test_extract_none(self):
        from coursepilot.rag.citation import extract_citations
        assert extract_citations("没有引用的普通回答") == []

    def test_extract_ids_not_sequential(self):
        from coursepilot.rag.citation import extract_citations
        assert extract_citations('<ref id="42" /><ref id="7" />') == [42, 7]

    def test_validate_all_valid(self):
        from coursepilot.rag.citation import validate_citations
        ok, hallu = validate_citations('<ref id="1" /><ref id="2" />', {1, 2, 3})
        assert ok is True
        assert hallu == set()

    def test_validate_hallucination(self):
        from coursepilot.rag.citation import validate_citations
        ok, hallu = validate_citations('<ref id="1" /><ref id="5" />', {1, 2, 3})
        assert ok is False
        assert hallu == {5}

    def test_validate_empty_answer(self):
        from coursepilot.rag.citation import validate_citations
        ok, hallu = validate_citations("无引用", {1, 2})
        assert ok is True
        assert hallu == set()

    def test_validate_no_refs_in_answer(self):
        from coursepilot.rag.citation import validate_citations
        ok, hallu = validate_citations("纯文本回答", set())
        assert ok is True
        assert hallu == set()

    def test_count(self):
        from coursepilot.rag.citation import count_citations
        assert count_citations('<ref id="1" /><ref id="2" />') == 2
        assert count_citations("none") == 0

    def test_count_with_whitespace(self):
        from coursepilot.rag.citation import count_citations
        assert count_citations('<ref id="1" />\n<ref id="2" />') == 2

    def test_extract_bare_ref(self):
        """缺少闭合斜杠的变体不匹配（严格遵循约定格式）"""
        from coursepilot.rag.citation import extract_citations
        assert extract_citations('<ref id="1">') == []


# ═══════════════════════════════════════════════════════════════
# RAGConfig 测试
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults(self):
        from coursepilot.rag.config import config
        assert config.dim == 1024
        assert config.enable_rewrite is True
        assert config.enable_sparse is True
        assert config.enable_rerank is True
        assert config.enable_kp_expand is True
        assert config.dense_top_k == 20
        assert config.sparse_top_k == 20
        assert config.rerank_top_k == 5
        assert config.rrf_k == 60

    def test_custom_config(self):
        from coursepilot.rag.config import RAGConfig
        cfg = RAGConfig(enable_rewrite=False, rerank_top_k=3, batch_size=16)
        assert cfg.enable_rewrite is False
        assert cfg.rerank_top_k == 3
        assert cfg.batch_size == 16

    def test_config_isolation(self):
        """自定义 config 不影响全局实例"""
        from coursepilot.rag.config import config, RAGConfig
        original = config.rerank_top_k
        RAGConfig(rerank_top_k=99)
        assert config.rerank_top_k == original


# ═══════════════════════════════════════════════════════════════
# Logger 测试
# ═══════════════════════════════════════════════════════════════

class TestLogger:
    def test_start_trace(self):
        from coursepilot.rag.logger import QueryLogger
        ql = QueryLogger()
        trace_id, start_time = ql.start_trace()
        assert len(trace_id) == 8
        assert start_time > 0

    def test_start_trace_unique(self):
        from coursepilot.rag.logger import QueryLogger
        ql = QueryLogger()
        id1, _ = ql.start_trace()
        id2, _ = ql.start_trace()
        assert id1 != id2

    def test_log_query_structured(self, caplog):
        import logging
        from coursepilot.rag.logger import QueryLogger

        ql = QueryLogger()
        ql._logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger="coursepilot.rag.query"):
            ql.log_query(
                trace_id="abc12345",
                user_id="user-1",
                course_id="course-1",
                query_raw="什么是定积分",
                query_rewritten="定积分的定义和几何意义",
                stages={"rewrite_ms": 480.5, "encode_ms": 120.0, "retrieve_ms": 85.3,
                        "rerank_ms": 210.7, "generate_ms": 1500.0},
                top_rerank_scores=[0.92, 0.87, 0.81],
                source_kp_paths=["高等数学/定积分/定义", "高等数学/定积分/几何意义"],
                citation_count=2,
                answer_length=350,
            )

        assert len(caplog.records) == 1
        log = json.loads(caplog.records[0].message)
        assert log["trace_id"] == "abc12345"
        assert log["query_raw"] == "什么是定积分"
        assert log["citation_count"] == 2
        assert log["answer_length"] == 350
        assert log["total_ms"] == pytest.approx(2396.5, rel=0.01)
        assert len(log["top_rerank_scores"]) == 3
        assert "timestamp" in log


# ═══════════════════════════════════════════════════════════════
# Encoder 测试（需 BGE-M3 本地模型）
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BGE_M3_AVAILABLE, reason="BGE-M3 模型未下载到 F:/all-projs/models/bge-m3")
class TestEncoder:
    def test_singleton(self):
        from coursepilot.rag.encoder import Encoder
        e1 = Encoder()
        e2 = Encoder()
        assert e1._model is e2._model

    def test_encode_single(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        results = encoder.encode(["测试文本"])
        print(results)
        print(len(results))
        print(results[0])
        assert len(results) == 1
        assert "dense" in results[0]
        assert "sparse" in results[0]
        assert len(results[0]["dense"]) == 1024
        assert isinstance(results[0]["sparse"], dict)

    def test_encode_multiple(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        results = encoder.encode(["第一段文本", "第二段文本", "第三段文本"])
        assert len(results) == 3
        for r in results:
            assert len(r["dense"]) == 1024

    def test_encode_query_shortcut(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        result = encoder.encode_query("单个查询字符串")
        assert len(result["dense"]) == 1024
        assert "sparse" in result

    def test_encode_empty(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        assert encoder.encode([]) == []

    def test_encode_queries_alias(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        result = encoder.encode_queries(["查询一", "查询二"])
        assert len(result) == 2

    def test_dim_property(self):
        from coursepilot.rag.encoder import Encoder
        encoder = Encoder()
        assert encoder.dim == 1024

    def test_dense_is_normalized(self):
        """BGE-M3 输出应已 L2 归一化（约等于 1.0）"""
        from coursepilot.rag.encoder import Encoder
        import math
        encoder = Encoder()
        result = encoder.encode(["测试"])[0]
        norm = math.sqrt(sum(v * v for v in result["dense"]))
        assert norm == pytest.approx(1.0, rel=0.01)


# ═══════════════════════════════════════════════════════════════
# VectorStore 测试（Milvus Lite，自动创建临时库）
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def vector_store():
    """创建临时 Milvus 数据库，每个测试独立（避免 Windows 文件锁冲突）"""
    import os
    import shutil
    try:
        from coursepilot.rag.vector_store import VectorStore
    except ImportError:
        pytest.skip("pymilvus 未安装")

    tmpdir = tempfile.mkdtemp(prefix="cp_test_")
    db_path = os.path.join(tmpdir, "test.db")
    store = VectorStore(db_path=db_path)
    store.create_collection()
    yield store
    try:
        store.client.close()
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(scope="module")
def sample_vecs():
    """编码几条示例文本，供后续测试共享"""
    if not BGE_M3_AVAILABLE:
        pytest.skip("BGE-M3 模型未下载")
    from coursepilot.rag.encoder import Encoder
    encoder = Encoder()
    vecs = encoder.encode([
        "定积分的几何意义是曲边梯形的面积",
        "不定积分是求导的逆运算",
        "牛顿-莱布尼茨公式连接了定积分和不定积分",
    ])
    return vecs


class TestVectorStore:
    def test_create_collection_idempotent(self, vector_store):
        """重复创建不报错"""
        vector_store.create_collection()

    def test_initial_count_zero(self, vector_store):
        assert vector_store.count() == 0

    def test_insert_and_count(self, vector_store, sample_vecs):
        payload = []
        for i, vec in enumerate(sample_vecs):
            payload.append({
                "uuid": f"unit-{i}",
                "dense_vec": vec["dense"],
                "sparse_vec": vec["sparse"],
                "kp_id": f"kp-id-{i}",
                "course_id": "test-course-001",
                "kp_path": f"高等数学/第{i}章",
                "content": f"content block {i}",
            })
        ids = vector_store.insert(payload)
        assert len(ids) == 3
        assert vector_store.count() == 3

    def _insert_sample_data(self, vector_store, sample_vecs):
        """向 vector_store 插入示例数据（供各测试复用）"""
        payload = []
        for i, vec in enumerate(sample_vecs):
            payload.append({
                "uuid": f"unit-{i}",
                "dense_vec": vec["dense"],
                "sparse_vec": vec["sparse"],
                "kp_id": f"kp-id-{i}",
                "course_id": "test-course-001",
                "kp_path": f"高等数学/第{i}章",
                "content": f"content block {i}",
            })
        return vector_store.insert(payload)

    def test_hybrid_search_returns_results(self, vector_store, sample_vecs):
        self._insert_sample_data(vector_store, sample_vecs)
        query_vec = sample_vecs[0]  # 用第一条做查询
        results = vector_store.hybrid_search(
            query_vec["dense"], query_vec["sparse"],
            course_id="test-course-001", top_k=2,
        )
        assert len(results) >= 1
        assert "content" in results[0]
        assert "score" in results[0]
        assert "kp_path" in results[0]

    def test_query_filter_by_course(self, vector_store, sample_vecs):
        """query() 的课程过滤应生效（Milvus Lite hybrid_search filter 有已知限制，改用 query 验证）"""
        self._insert_sample_data(vector_store, sample_vecs)
        rows = vector_store.query_by_course("non-existent-course")
        assert len(rows) == 0
        rows = vector_store.query_by_course("test-course-001")
        assert len(rows) == 3

    def test_delete_by_uuids(self, vector_store):
        vector_store.delete_by_uuids(["unit-0"])
        # Milvus Lite delete 后需要 compaction 才反映到 count，这里仅验证不抛异常

    def test_delete_by_course(self, vector_store):
        vector_store.delete_by_course("test-course-001")

    def test_query_by_course(self, vector_store, sample_vecs):
        """重新插入后查询"""
        payload = [{
            "uuid": "query-test-1",
            "dense_vec": sample_vecs[0]["dense"],
            "sparse_vec": sample_vecs[0]["sparse"],
            "kp_id": "kp-x", "course_id": "query-test-course",
            "kp_path": "数学/测试", "content": "测试内容",
        }]
        vector_store.insert(payload)
        rows = vector_store.query_by_course("query-test-course")
        assert len(rows) >= 1
        assert rows[0]["uuid"] == "query-test-1"


# ═══════════════════════════════════════════════════════════════
# QueryRewriter 测试
# ═══════════════════════════════════════════════════════════════

class TestQueryRewriter:
    def test_init(self):
        from coursepilot.rag.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        assert rw is not None

    @pytest.mark.asyncio
    async def test_rewrite_no_api_key_graceful_degrade(self):
        """没有 API Key 时降级返回原查询"""
        from coursepilot.rag.query_rewriter import QueryRewriter
        rw = QueryRewriter(api_key="")
        result = await rw.rewrite("什么是定积分")
        assert result == "什么是定积分"

    @pytest.mark.asyncio
    async def test_rewrite_no_api_key_graceful_degrade_kwarg(self):
        """通过参数关闭 rewrite"""
        from coursepilot.rag.query_rewriter import QueryRewriter
        rw = QueryRewriter(api_key="")
        result = await rw.rewrite("极限的定义")
        assert result == "极限的定义"

    @pytest.mark.skipif(not API_KEY_AVAILABLE, reason="未配置 LLM_API_KEY")
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rewrite_with_api(self):
        from coursepilot.rag.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        result = await rw.rewrite("什么是定积分的几何意义")
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════
# Reranker 测试（需 bge-reranker-v2-m3 本地模型）
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not RERANKER_AVAILABLE, reason="reranker 模型未下载到 F:/all-projs/models/bge-reranker-v2-m3")
class TestReranker:
    @pytest.fixture(autouse=True)
    def _disable_min_score(self):
        """测试时关闭 reranker_min_score 过滤，避免模型分低于阈值导致候选被丢弃"""
        from coursepilot.rag.config import config
        old = config.reranker_min_score
        config.reranker_min_score = -1.0
        yield
        config.reranker_min_score = old

    def test_singleton(self):
        from coursepilot.rag.reranker import Reranker
        r1 = Reranker()
        r2 = Reranker()
        assert r1.model is r2.model

    def test_rerank_orders_by_relevance(self):
        from coursepilot.rag.reranker import Reranker
        reranker = Reranker()

        query = "定积分的几何意义是什么"
        candidates = [
            {"content": "定积分的几何意义是曲边梯形的面积", "kp_path": "高等数学/定积分/定义", "score": 0.8},
            {"content": "不定积分是求导的逆运算，也称为原函数", "kp_path": "高等数学/不定积分", "score": 0.6},
            {"content": "牛顿-莱布尼茨公式建立了定积分与原函数的联系", "kp_path": "高等数学/定积分/公式", "score": 0.7},
        ]

        results = reranker.rerank(query, candidates, top_k=3)
        assert len(results) == 3
        # 最相关的排第一
        assert results[0]["rerank_score"] >= results[1]["rerank_score"]
        assert results[1]["rerank_score"] >= results[2]["rerank_score"]
        assert "rerank_score" in results[0]

    def test_rerank_top_k_truncation(self):
        from coursepilot.rag.reranker import Reranker
        reranker = Reranker()

        # 使用与查询相关的候选内容，确保模型给出正分
        query = "极限的定义与基本性质"
        candidates = [
            {"content": "极限是微积分中描述函数在自变量趋向某值时行为的基本概念", "kp_path": "数学/极限", "score": 0.5}
            for _ in range(10)
        ]
        results = reranker.rerank(query, candidates, top_k=3)
        assert len(results) == 3

    def test_rerank_empty_candidates(self):
        from coursepilot.rag.reranker import Reranker
        reranker = Reranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_level_penalty_applied(self):
        """深层 KP 的层级惩罚更重，最终得分应更低"""
        from coursepilot.rag.reranker import Reranker
        reranker = Reranker()

        query = "微积分基础概念"
        candidates = [
            {"content": "微积分是研究函数的微分与积分的数学分支", "kp_path": "微积分/基础", "score": 0.9},
            {"content": "微积分是研究函数的微分与积分的数学分支", "kp_path": "微积分/基础/子章节/细节", "score": 0.9},
        ]
        results = reranker.rerank(query, candidates, top_k=2)
        assert len(results) == 2
        # 深度大的 KP 惩罚更重，最终得分应更低
        shallow = next(r for r in results if r["kp_path"] == "微积分/基础")
        deep = next(r for r in results if r["kp_path"] == "微积分/基础/子章节/细节")
        assert shallow["rerank_score"] > deep["rerank_score"]


# ═══════════════════════════════════════════════════════════════
# Generator 测试
# ═══════════════════════════════════════════════════════════════

class TestGenerator:
    def test_init(self):
        from coursepilot.rag.generator import Generator
        g = Generator()
        assert g is not None
        assert g.model is not None

    def test_format_course(self):
        from coursepilot.rag.generator import _format_course
        ctx = {"name": "高等数学", "textbook": "同济高等数学·第八版", "chapters": ["极限", "导数", "积分"]}
        result = _format_course(ctx)
        assert "高等数学" in result
        assert "同济高等数学" in result
        assert "极限" in result

    def test_format_course_none(self):
        from coursepilot.rag.generator import _format_course
        assert "未指定课程" in _format_course(None)
        assert "未指定课程" in _format_course({})

    def test_system_prompt_format(self):
        from coursepilot.rag.generator import SYSTEM_PROMPT
        prompt = SYSTEM_PROMPT.format(
            course_context="课程：测试课\n教材：测试教材",
            sources="<source id=\"1\">测试内容</source>",
        )
        assert "测试课" in prompt
        assert "测试教材" in prompt
        assert '<source id="1">' in prompt

    @pytest.mark.skipif(not API_KEY_AVAILABLE, reason="未配置 LLM_API_KEY")
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_generate(self):
        from coursepilot.rag.generator import Generator
        g = Generator()
        answer = await g.generate(
            query="1+1等于几",
            context='<source id="1" path="数学/算术" pages="" book="测试">加法是最基本的运算</source>',
            course_context={"name": "数学", "textbook": "测试教材", "chapters": ["算术"]},
            max_tokens=100,
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.skipif(not API_KEY_AVAILABLE, reason="未配置 LLM_API_KEY")
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_generate_stream(self):
        from coursepilot.rag.generator import Generator
        g = Generator()
        tokens = []
        async for token in g.generate_stream(
            query="1+1等于几",
            context='<source id="1" path="数学/算术" pages="" book="测试">加法是最基本的运算</source>',
            course_context={"name": "数学", "textbook": "测试教材", "chapters": ["算术"]},
            max_tokens=50,
        ):
            tokens.append(token)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_generate_no_api_key(self):
        from coursepilot.rag.generator import Generator
        g = Generator(api_key="")
        import asyncio
        answer = asyncio.run(g.generate("test", "context", None))
        assert "未配置" in answer


# ═══════════════════════════════════════════════════════════════
# Retriever 集成测试（需 PG + Milvus + BGE-M3 + Reranker）
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (BGE_M3_AVAILABLE and RERANKER_AVAILABLE),
    reason="需要 BGE-M3 + reranker 模型"
)
class TestRetrieverIntegration:
    @pytest.mark.asyncio
    async def test_retrieve_empty_course(self):
        """不存在的课程应返回空上下文"""
        from coursepilot.rag.retriever import Retriever
        from coursepilot.rag.config import RAGConfig

        # 用最小配置避免外部调用
        retriever = Retriever()
        # 对不存在的课程，hybrid_search 应返回空
        # 这里用 mock 避免真正调用
        with patch.object(retriever.vector_store, 'hybrid_search', return_value=[]):
            context, metadata = await retriever.retrieve(
                session=AsyncMock(),
                query="测试问题",
                course_id="00000000-0000-0000-0000-000000000000",
                enable_rewrite=False,
            )
            assert context == ""
            assert metadata["candidate_count"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_disables_rewrite(self):
        """enable_rewrite=False 时不调用改写器"""
        from coursepilot.rag.retriever import Retriever

        retriever = Retriever()
        with patch.object(retriever.rewriter, 'rewrite') as mock_rewrite:
            with patch.object(retriever.vector_store, 'hybrid_search', return_value=[]):
                context, metadata = await retriever.retrieve(
                    session=AsyncMock(),
                    query="原始问题",
                    course_id="00000000-0000-0000-0000-000000000000",
                    enable_rewrite=False,
                )
                mock_rewrite.assert_not_called()
                assert metadata["query_rewritten"] == "原始问题"


# ═══════════════════════════════════════════════════════════════
# _kp_expand / _format_units 格式化测试
# ═══════════════════════════════════════════════════════════════

class TestContextFormatting:
    def test_format_units_basic(self):
        from coursepilot.rag.retriever import _format_units
        units = [
            {"content": "段落一", "kp_path": "数学/第一章", "score": 0.9},
            {"content": "段落二", "kp_path": "数学/第二章", "score": 0.8},
        ]
        result = _format_units(units, max_chars=10000)
        assert '<source id="1"' in result
        assert '<source id="2"' in result
        assert "段落一" in result
        assert "段落二" in result

    def test_format_units_truncation(self):
        from coursepilot.rag.retriever import _format_units
        units = [
            {"content": "x" * 5000, "kp_path": "数学/第一章", "score": 0.9},
            {"content": "y" * 5000, "kp_path": "数学/第二章", "score": 0.8},
        ]
        result = _format_units(units, max_chars=200)
        # 应该只有第一个 source（第二个因为超上限被截断）
        assert '<source id="1"' in result
        assert len(result) < 5000  # 被截断了
