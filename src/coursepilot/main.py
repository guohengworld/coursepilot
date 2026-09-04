"""FastAPI 应用入口

启动方式：
    uvicorn coursepilot.main:app --reload
"""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 必须在任何 PyTorch 导入之前设置，防止 CUDA 内存碎片化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 原生库加载顺序约束：
# pymilvus（连带加载 pyarrow/arrow.dll）必须在 torch 之前导入。
# 反之 torch 先加载、ask 接口函数体内才懒加载 pymilvus，arrow.dll 会触发
# 0xc0000005 访问冲突，进程无输出直接退出（Vite 代理侧表现为 502 Bad Gateway）。
import pymilvus  # noqa: F401, E402

# 注意：agent 链（torch/FlagEmbedding）必须最先导入——本机实测若 admin/auth/courses
# 先于 agent 导入，torch 后加载会触发原生库顺序冲突导致堆损坏（0xC0000374），
# 进程无任何输出直接退出。此顺序不可调回（2026-09-03 用户本机终端同样复现）。
from coursepilot.api.agent import router as agent_router
from coursepilot.api.admin import router as admin_router
from coursepilot.api.auth import router as auth_router
from coursepilot.api.courses import router as courses_router
from coursepilot.api.practice import router as practice_router
from coursepilot.api.tasks import router as tasks_router
from coursepilot.db import _get_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时 / 关闭时执行"""
    # 启动：可以在这里初始化 Milvus、加载模型等
    yield
    # 关闭：释放数据库连接池
    engine = _get_engine()
    await engine.dispose()

app = FastAPI(
    title="CoursePilot API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(courses_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")
app.include_router(practice_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # 确保 src 在 sys.path 中，支持 python src/coursepilot/main.py 直接启动
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import uvicorn
    uvicorn.run("coursepilot.main:app", host="0.0.0.0", port=8000, reload=True)
