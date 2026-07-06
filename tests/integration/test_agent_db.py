"""Agent 数据库集成测试：用真实 PostgreSQL 验证节点函数的 DB 交互

需要运行中的 PostgreSQL（coursepilot_test 数据库），自动建表清表。
LLM 技能全部 mock，仅测试 DB 层。
"""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from coursepilot.models import AgentSession, PracticeRecord, QARecord, UserProfile
from tests.integration.conftest import ZERO_TOKENS, seed_std


class TestBuildContextNode:
    """build_context_node 真实数据库集成测试"""

    @pytest.mark.asyncio
    async def test_build_context_with_real_db(self, raw_session, real_asf):
        """种子 UserProfile + QARecord → build_context 正确加载"""
        d = await seed_std(raw_session)
        raw_session.add(UserProfile(
            user_id=d["user_id"], course_id=d["course_id"],
            mastery_level={"OS": 0.8, "OS/进程管理": 0.6},
            weak_kps=["OS/进程管理"],
            avg_correct_rate=0.7, total_qa_count=3,
        ))
        for i in range(5):
            raw_session.add(QARecord(
                user_id=d["user_id"], course_id=d["course_id"],
                query=f"问题{i}", answer=f"答案{i}",
            ))
        await raw_session.commit()

        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import build_context_node
            result = await build_context_node({
                "user_id": str(d["user_id"]), "course_id": str(d["course_id"]),
            })

        assert result["error"] is None
        cc = result["course_context"]
        assert cc.get("name") == "操作系统"
        assert len(cc.get("chapters", [])) >= 1
        assert result["user_profile"] is not None
        assert result["user_profile"]["mastery_level"]["OS"] == 0.8
        assert len(result["recent_qa"]) == 5

    @pytest.mark.asyncio
    async def test_build_context_no_profile(self, raw_session, real_asf):
        """无 UserProfile 时返回 None"""
        d = await seed_std(raw_session)
        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import build_context_node
            result = await build_context_node({
                "user_id": str(d["user_id"]), "course_id": str(d["course_id"]),
            })
        assert result["error"] is None
        assert result["user_profile"] is None

    @pytest.mark.asyncio
    async def test_build_context_no_qa(self, raw_session, real_asf):
        """无 QARecord 时 recent_qa 为空"""
        d = await seed_std(raw_session)
        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import build_context_node
            result = await build_context_node({
                "user_id": str(d["user_id"]), "course_id": str(d["course_id"]),
            })
        assert result["error"] is None
        assert result["recent_qa"] == []


class TestFinalizeNode:
    """finalize_node 真实数据库集成测试"""

    def _state(self, d, sess_id, llm_calls=None):
        return {
            "user_id": str(d["user_id"]), "course_id": str(d["course_id"]),
            "query": "什么是进程调度？", "answer": "进程调度是操作系统核心功能",
            "session_id": str(sess_id), "intent": "question", "context": "上下文",
            "retrieved_metadata": {"source_kp_paths": ["OS/进程管理"], "top_uuids": []},
            "sources": [{"kp_path": "OS/进程管理"}],
            "llm_calls": llm_calls or [{"node": "classify", **ZERO_TOKENS}],
            "human_review_result": None,
        }

    @pytest.mark.asyncio
    async def test_finalize_writes_qa_record(self, raw_session, real_asf):
        """finalize 后 qa_records 表存在对应行"""
        d = await seed_std(raw_session)
        sess_id = uuid.uuid4()
        raw_session.add(AgentSession(
            id=sess_id, user_id=d["user_id"], course_id=d["course_id"],
            intent="question", status="running",
        ))
        await raw_session.commit()

        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import finalize_node
            result = await finalize_node(self._state(d, sess_id))

        assert result["error"] is None
        rows = (await raw_session.execute(
            select(QARecord).where(QARecord.user_id == d["user_id"])
        )).scalars().all()
        assert len(rows) >= 1
        assert rows[0].query == "什么是进程调度？"

    @pytest.mark.asyncio
    async def test_finalize_updates_session(self, raw_session, real_asf):
        """AgentSession 状态更新为 completed"""
        d = await seed_std(raw_session)
        sess_id = uuid.uuid4()
        raw_session.add(AgentSession(
            id=sess_id, user_id=d["user_id"], course_id=d["course_id"],
            intent="question", status="running",
        ))
        await raw_session.commit()

        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import finalize_node
            await finalize_node(self._state(d, sess_id))

        sess = (await raw_session.execute(
            select(AgentSession).where(AgentSession.id == sess_id)
        )).scalar_one()
        assert sess.status == "completed"
        assert sess.intent == "question"

    @pytest.mark.asyncio
    async def test_finalize_no_crash(self, raw_session, real_asf):
        """finalize 不崩溃"""
        d = await seed_std(raw_session)
        sess_id = uuid.uuid4()
        raw_session.add(AgentSession(
            id=sess_id, user_id=d["user_id"], course_id=d["course_id"],
            intent="question", status="running",
        ))
        await raw_session.commit()

        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import finalize_node
            result = await finalize_node(self._state(d, sess_id))
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_finalize_token_sum(self, raw_session, real_asf):
        """finalize 汇总 llm_calls token 计数"""
        d = await seed_std(raw_session)
        sess_id = uuid.uuid4()
        raw_session.add(AgentSession(
            id=sess_id, user_id=d["user_id"], course_id=d["course_id"],
            intent="question", status="running",
        ))
        await raw_session.commit()

        state = self._state(d, sess_id, [
            {"node": "classify", "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            {"node": "query_rag", "prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        ])
        with patch("coursepilot.agent.nodes.async_session_factory", real_asf):
            from coursepilot.agent.nodes import finalize_node
            result = await finalize_node(state)
        assert result["error"] is None
        assert result["token_count"] == 340


class TestProfileUpdater:
    """profile_updater 真实数据库集成测试"""

    @pytest.mark.asyncio
    async def test_update_profile_aggregates(self, raw_session, real_asf):
        """PracticeRecord → update_profile 计算出掌握度"""
        d = await seed_std(raw_session)
        for correct in [True, True, True, True, False]:
            raw_session.add(PracticeRecord(
                user_id=d["user_id"], question_id=d["question_id"],
                correct_flag=correct,
            ))
        await raw_session.commit()

        with patch("coursepilot.agent.profile_updater.async_session_factory", real_asf):
            from coursepilot.agent.profile_updater import update_profile
            await update_profile(user_id=str(d["user_id"]), course_id=str(d["course_id"]))

        up = (await raw_session.execute(
            select(UserProfile).where(
                UserProfile.user_id == d["user_id"],
                UserProfile.course_id == d["course_id"],
            )
        )).scalar_one_or_none()
        assert up is not None
        assert up.mastery_level.get("OS/进程管理") == 0.8
        assert float(up.avg_correct_rate) == 0.8

    @pytest.mark.asyncio
    async def test_update_profile_no_records(self, raw_session, real_asf):
        """无练习记录时 profile 仍可创建"""
        d = await seed_std(raw_session)
        with patch("coursepilot.agent.profile_updater.async_session_factory", real_asf):
            from coursepilot.agent.profile_updater import update_profile
            await update_profile(user_id=str(d["user_id"]), course_id=str(d["course_id"]))

        up = (await raw_session.execute(
            select(UserProfile).where(
                UserProfile.user_id == d["user_id"],
                UserProfile.course_id == d["course_id"],
            )
        )).scalar_one_or_none()
        assert up is not None
        assert up.mastery_level == {}

    @pytest.mark.asyncio
    async def test_update_profile_identifies_weak_kps(self, raw_session, real_asf):
        """正确率 < 0.6 的 KP 标记为 weak"""
        d = await seed_std(raw_session)
        for correct in [True, False, False]:
            raw_session.add(PracticeRecord(
                user_id=d["user_id"], question_id=d["question_id"],
                correct_flag=correct,
            ))
        await raw_session.commit()

        with patch("coursepilot.agent.profile_updater.async_session_factory", real_asf):
            from coursepilot.agent.profile_updater import update_profile
            await update_profile(user_id=str(d["user_id"]), course_id=str(d["course_id"]))

        up = (await raw_session.execute(
            select(UserProfile).where(
                UserProfile.user_id == d["user_id"],
                UserProfile.course_id == d["course_id"],
            )
        )).scalar_one()
        assert "OS/进程管理" in (up.weak_kps or [])
