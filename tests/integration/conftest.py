"""Agent 集成测试共享夹具

使用 coursepilot_test 数据库（需提前创建），自动建表，每测试后清表。
依赖：PostgreSQL 服务运行在 localhost:5432
"""
import uuid
from unittest.mock import patch

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from coursepilot.config import settings
from coursepilot.db import Base
from coursepilot.models import (
    Course, KnowledgePoint, Question, User,
)

TEST_DB_URL = settings.database_url.replace("/coursepilot", "/coursepilot_test")
ZERO_TOKENS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def seed_std(session):
    """灌入标准数据 User + Course + KP + Question，返回 IDs

    注意：async 模式下 add_all 不保证 FK 插入顺序，因此必须
    逐个 add + flush 保证外键依赖先被满足。
    """
    uid = uuid.uuid4()
    cid = uuid.uuid4()
    kp_root = uuid.uuid4()
    kp_child = uuid.uuid4()
    qid = uuid.uuid4()

    session.add(User(id=uid, username="test_student", password_hash="x", role="student"))
    await session.flush()

    session.add(Course(id=cid, name="操作系统", description="测试课程", created_by=uid))
    await session.flush()

    session.add_all([
        KnowledgePoint(id=kp_root, course_id=cid, kp_path="OS",
                       title="操作系统总论", difficulty=1, source="course"),
        KnowledgePoint(id=kp_child, course_id=cid, parent_id=kp_root,
                       kp_path="OS/进程管理", title="进程管理", difficulty=2, source="course"),
        Question(id=qid, kp_id=kp_child,
                 question_text="什么是进程?", correct_answer="B",
                 options={"A": "程序", "B": "进程实例", "C": "文件", "D": "线程"},
                 explanation="进程是程序在执行过程中的实例",
                 difficulty=2, question_type="choice_4"),
    ])
    await session.commit()
    return {"user_id": uid, "course_id": cid, "kp_root_id": kp_root,
            "kp_child_id": kp_child, "question_id": qid}


@pytest_asyncio.fixture
async def db_engine():
    """function-scoped: 测试数据库引擎 + 建表 + 清空数据

    setup 时 TRUNCATE 全部业务表，保证每个测试都从干净状态开始，
    不受上一次运行残留数据影响（否则首个 seed 会撞唯一约束）。
    """
    from sqlalchemy import text as _text

    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 清空所有业务表（排除 alembic_version），避免跨测试残留
        r = await conn.execute(_text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' "
            "AND tablename <> 'alembic_version';"
        ))
        tables = [row[0] for row in r.fetchall()]
        if tables:
            quoted = ", ".join(f'"{t}"' for t in tables)
            await conn.execute(_text(f"TRUNCATE TABLE {quoted} CASCADE;"))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def raw_session(db_engine):
    """function-scoped: 独立 session，teardown 时清空所有表"""
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    async with factory() as cleaner:
        for table in reversed(Base.metadata.sorted_tables):
            await cleaner.execute(table.delete())
        await cleaner.commit()


@pytest_asyncio.fixture
async def real_asf(db_engine):
    """指向 coursepilot_test 的 async_sessionmaker，供 patch 使用"""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def std_data(raw_session):
    """灌入标准 User + Course + KP + Question，返回 IDs 字典"""
    return await seed_std(raw_session)


@pytest_asyncio.fixture(scope="module")
async def memory_graph():
    """用 MemorySaver 编译完整 Agent 图（代替 PostgresSaver）"""
    from langgraph.checkpoint.memory import MemorySaver
    with patch("coursepilot.agent.graph._get_saver", return_value=MemorySaver()):
        from coursepilot.agent.graph import build_agent_graph
        graph = await build_agent_graph()
    return graph
