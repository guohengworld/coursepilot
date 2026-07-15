"""BM25Indexer + rrf_fuse 单元测试"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# 测试用有效 UUID
CID_A = "11111111-1111-1111-1111-111111111111"
CID_B = "22222222-2222-2222-2222-222222222222"
CID_C = "33333333-3333-3333-3333-333333333333"
CID_D = "44444444-4444-4444-4444-444444444444"
CID_E = "55555555-5555-5555-5555-555555555555"


@pytest.fixture(autouse=True)
def _clear_bm25_cache():
    """每个测试前清除 BM25 模块缓存，避免测试间相互影响。"""
    from coursepilot.rag.bm25 import _caches
    _caches.clear()
    yield
    _caches.clear()


# ═══════════════════════════════════════════════════════════════
# rrf_fuse
# ═══════════════════════════════════════════════════════════════

class TestRRFFuse:
    def test_empty_lists(self):
        from coursepilot.rag.bm25 import rrf_fuse
        assert rrf_fuse([], top_k=10) == []
        assert rrf_fuse([[], []], top_k=10) == []

    def test_single_list(self):
        from coursepilot.rag.bm25 import rrf_fuse
        items = [
            {"uuid": "a", "content": "aaa", "score": 0.9},
            {"uuid": "b", "content": "bbb", "score": 0.8},
            {"uuid": "c", "content": "ccc", "score": 0.7},
        ]
        result = rrf_fuse([items], top_k=10)
        assert len(result) == 3
        assert result[0]["uuid"] == "a"

    def test_single_list_truncated(self):
        from coursepilot.rag.bm25 import rrf_fuse
        items = [{"uuid": str(i), "score": 1.0 - i * 0.1} for i in range(10)]
        result = rrf_fuse([items], top_k=3)
        assert len(result) == 3

    def test_two_lists_with_overlap(self):
        """同一 doc 出现在两个列表中时，RRF 融合分应大于只在一个列表中的 doc。"""
        from coursepilot.rag.bm25 import rrf_fuse

        list_a = [
            {"uuid": "doc1", "content": "x"},
            {"uuid": "doc2", "content": "y"},
            {"uuid": "doc3", "content": "z"},
        ]
        list_b = [
            {"uuid": "doc2", "content": "y"},
            {"uuid": "doc4", "content": "w"},
        ]

        result = rrf_fuse([list_a, list_b], k=60, top_k=10)

        uuids = [r["uuid"] for r in result]
        assert "doc1" in uuids
        assert "doc2" in uuids
        assert "doc3" in uuids
        assert "doc4" in uuids

        # doc2: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522...
        doc2 = [r for r in result if r["uuid"] == "doc2"][0]
        expected = 1.0 / 62 + 1.0 / 61
        assert abs(doc2["score"] - expected) < 1e-6

    def test_items_without_uuid_skipped_in_multi_list(self):
        """多列表融合时，缺少 uuid 的 item 应被跳过。"""
        from coursepilot.rag.bm25 import rrf_fuse
        list_a = [
            {"uuid": "a", "score": 0.9},
            {"id_only": "b", "score": 0.8},  # no uuid field
        ]
        list_b = [
            {"uuid": "c", "score": 0.7},
        ]
        result = rrf_fuse([list_a, list_b], k=60, top_k=10)
        uuids = [r["uuid"] for r in result]
        assert "a" in uuids
        assert "c" in uuids
        assert len(result) == 2

    def test_multi_list_rrf_score_computed(self):
        """多列表融合时，score 应设为 RRF 值而非原始 score。"""
        from coursepilot.rag.bm25 import rrf_fuse
        list_a = [{"uuid": "a", "content": "x", "score": 999}]
        list_b = [{"uuid": "b", "content": "y", "score": 888}]
        result = rrf_fuse([list_a, list_b], k=60, top_k=10)
        # a: 1/(60+1) ≈ 0.01639
        assert result[0]["score"] == pytest.approx(1.0 / 61)


# ═══════════════════════════════════════════════════════════════
# BM25Indexer
# ═══════════════════════════════════════════════════════════════

class TestBM25Indexer:
    """模拟 `session.execute().all()` 返回 fake KU，不需要真实 PG / Milvus。"""

    @staticmethod
    def _mock_session(rows: list[tuple]) -> AsyncMock:
        session = AsyncMock()
        execute_result = MagicMock()
        execute_result.all.return_value = rows
        session.execute = AsyncMock(return_value=execute_result)
        return session

    def _fake_rows(self):
        return [
            ("u1", "定积分是微积分中的重要概念", "定积分概述", "kp1", "微积分/定积分"),
            ("u2", "牛顿-莱布尼茨公式沟通了积分与微分", "牛莱公式", "kp2", "微积分/定积分/牛莱公式"),
            ("u3", "黎曼和是定积分的定义基础", "黎曼和", "kp3", "微积分/定积分/黎曼和"),
        ]

    async def test_search_empty_index(self):
        """无 KU 时返回空列表。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        session = self._mock_session([])
        results = await indexer.search(session, "定积分", CID_A, top_k=5)
        assert results == []

    async def test_search_returns_results(self):
        """正常检索返回按 score 降序的结果。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        session = self._mock_session(self._fake_rows())
        results = await indexer.search(session, "定积分", CID_A, top_k=5)

        assert len(results) > 0
        for r in results:
            assert "uuid" in r
            assert "content" in r
            assert "summary" in r
            assert "kp_id" in r
            assert "kp_path" in r
            assert "score" in r
            assert r["score"] > 0

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_cache_hit(self):
        """第二次检索同一课程应命中缓存，不执行 PG 查询。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        rows = self._fake_rows()
        session = self._mock_session(rows)

        r1 = await indexer.search(session, "定积分", CID_B, top_k=5)
        assert len(r1) > 0
        assert session.execute.call_count == 1

        # 第二次：如果命中缓存则不会调用 session.execute
        session.execute.side_effect = RuntimeError("不应再查询 PG")
        r2 = await indexer.search(session, "定积分", CID_B, top_k=5)
        assert len(r2) > 0
        assert [r["uuid"] for r in r1] == [r["uuid"] for r in r2]

    async def test_ttl_expired(self):
        """TTL 过期后重建索引。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=0)  # TTL=0 → 立即过期
        rows = self._fake_rows()
        session = self._mock_session(rows)

        await indexer.search(session, "定积分", CID_C, top_k=5)

        session2 = self._mock_session(rows)
        await indexer.search(session2, "定积分", CID_C, top_k=5)
        assert session2.execute.call_count == 1

    async def test_invalidate(self):
        """invalidate 后清除缓存，下次搜索触发重建。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        session = self._mock_session(self._fake_rows())

        await indexer.search(session, "定积分", CID_D, top_k=5)
        assert session.execute.call_count == 1

        BM25Indexer.invalidate(CID_D)

        session2 = self._mock_session(self._fake_rows())
        await indexer.search(session2, "定积分", CID_D, top_k=5)
        assert session2.execute.call_count == 1

    async def test_query_without_matching_terms(self):
        """不相关查询返回空列表（无匹配词、score=0 的 item 被过滤）。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        session = self._mock_session(self._fake_rows())
        results = await indexer.search(session, "zzzznotchinese", CID_E, top_k=5)
        assert results == []

    async def test_different_courses_isolated(self):
        """不同课程有独立的索引缓存，搜索结果互不干扰。"""
        from coursepilot.rag.bm25 import BM25Indexer

        indexer = BM25Indexer(ttl=300)
        rows_a = [
            ("a1", "高等数学微积分内容", "高数摘要", "kp_a1", "高等数学"),
        ]
        rows_b = [
            ("b1", "线性代数矩阵运算", "线代摘要", "kp_b1", "线性代数"),
        ]

        session_a = self._mock_session(rows_a)
        session_b = self._mock_session(rows_b)

        r_a = await indexer.search(session_a, "微积分", CID_A, top_k=5)
        r_b = await indexer.search(session_b, "矩阵", CID_B, top_k=5)

        assert any("微积分" in r["content"] for r in r_a)
        assert any("矩阵" in r["content"] for r in r_b)
