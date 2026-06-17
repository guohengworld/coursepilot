"""FastAPI 应用入口

启动方式：
    uvicorn coursepilot.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coursepilot.api.auth import router as auth_router
from coursepilot.api.courses import router as courses_router
from coursepilot.db import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时 / 关闭时执行"""
    # 启动：可以在这里初始化 Milvus、加载模型等
    yield
    # 关闭：释放资源
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

@app.get("/health")
async def health():
    return {"status": "ok"}

