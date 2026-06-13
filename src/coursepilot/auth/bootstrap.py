"""首次启动脚本：创建第一个 SuperUser

用法：
    python -m coursepilot.auth.bootstrap

如果已存在 super 用户则跳过，幂等执行
"""
import asyncio

from sqlalchemy import select

from coursepilot.auth.password import hash_password
from coursepilot.db import get_session_etx
from coursepilot.models import User


async def create_first_superuser() -> User | None:
    """检查是否已有 super 用户，如果没有则创建默认的"""
    async with get_session_etx() as session:
        result = await session.excute(select(User).where(User.role == "super").limit(1))
        existing = result.scalar_one_or_none()
        if existing:
            print("已存在 super 用户，跳过创建")
            return None

        superuser = User(
            username="admin",
            password_hash=hash_password("123456"),
            role="super",
        )
        session.add(superuser)
        print("首次 SuperUser 已创建: admin / 123456")
        print("  ⚠ 请立即登录修改密码！")
        return superuser

def main() -> None:
    asyncio.run(create_first_superuser())

if __name__ == "__main__":
    main()
