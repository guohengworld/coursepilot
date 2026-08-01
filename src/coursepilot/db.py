"""异步数据库引擎与会话管理

引擎采用懒加载（lazy init），避免模块导入时触发 asyncpg 加载。
Windows WMI 死锁修复已在 coursepilot/__init__.py 顶部统一处理
（必须在 import sqlalchemy 之前生效），此处无需重复。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from coursepilot.config import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models.

    不依赖 asyncpg/engine，可以安全地在模型文件中导入。
    """
    pass


# ── 懒加载引擎 ─────────────────────────────────────────────
# engine 和 session_factory 都在首次使用时才初始化。
# 优势：import coursepilot.db（或 import Base）不触发 asyncpg 加载；
#       只有实际连接数据库时才加载驱动。
# 劣势：每次 fastapi get_session() 或手动 get_session_etx() 都可能
#       需要检查 engine 是否已初始化（开销可忽略）。

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=False,
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        _get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


# 对外保留一致的接口名，但现在是 lazy function
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入用的 session 获取器

    每次请求创建一个新的 session，结束请求后自动 commit 或 rollback。
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_etx() -> AsyncIterator[AsyncSession]:
    """非 FastAPI 环境（脚本、测试）用的 session 上下文管理器"""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def async_session_factory() -> AsyncSession:
    """返回一个新的 AsyncSession（每次调用独立，支持 async with）

    用于 MCP Server 等非 FastAPI 场景。
    使用方式：async with async_session_factory() as session:
    """
    factory = _get_session_factory()
    return factory()
