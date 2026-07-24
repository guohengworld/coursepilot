"""验证 LangGraph PostgresSaver checkpoint 持久化 (最终修正版)"""
import asyncio
import sys
from uuid import uuid4
import psycopg  # 仅用于最后的直查验证

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

SYNC_URL = "postgresql://postgres:123456@localhost:5432/coursepilot"

# 【关键修复】在 Windows 下强制使用 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    # 1. 使用 async with 创建 Saver (from_conn_string 是异步上下文管理器)
    async with AsyncPostgresSaver.from_conn_string(SYNC_URL) as saver:
        await saver.setup()  # 创建 checkpoint_* 表
        # Windows + psycopg 3.3: 禁用 pipeline 模式避免 put() 卡死
        saver.supports_pipeline = False

        # 2. 写入一个 checkpoint
        thread_id = f"test-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        checkpoint = {
            "v": 1,
            "ts": "2025-01-01T00:00:00+00:00",
            "id": str(uuid4()),
            "channel_values": {"query": "测试", "answer": "测试回答"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        await saver.aput(config, checkpoint, {}, {})
        print(f"[写入] thread_id={thread_id}, checkpoint_id={checkpoint['id']}")

        # 3. 读取验证
        result = await saver.aget(config)
        if result:
            print(f"[读取成功] channel_values: {result['channel_values']}")
        else:
            print("[读取失败] 未获取到 checkpoint")

    # 4. 直接查数据库确认 (在 async with 外部进行，避免连接冲突)
    async with await psycopg.AsyncConnection.connect(SYNC_URL, autocommit=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)
            )
            cnt = await cur.fetchone()
            print(f"[DB 直查] checkpoints 表记录数: {cnt[0]}")

    print("=== 全部通过 ===")

asyncio.run(main())
