"""课程管理 API"""
from uuid import UUID
import time
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.api.deps import get_current_user, require_superuser
from coursepilot.db import get_session
from coursepilot.models import User, Course, Document, KnowledgePoint, KnowledgeUnit
from coursepilot.storage import file_store
from coursepilot.storage.file_store import FileStore
from coursepilot.rag.vector_store import VectorStore

router = APIRouter(prefix="/courses", tags=["courses"])

file_store = FileStore()

# Schema
class CourseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = None

class CourseOut(BaseModel):
    id: str
    name: str
    description: str | None
    created_by: str
    created_at: str

class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int | None
    status: str
    page_count: int | None
    uploaded_at: str

class KPTreeNodeOut(BaseModel):
    id: str
    title: str
    kp_path: str
    children: list["KPTreeNodeOut"] = []

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    trace_id: str
    rewritten_query: str
    citations: list[int] = Field(description="引用的页面索引列表")
    top_scores: list[float] = Field(description="相似度得分列表")
    source_kp_paths: list[str] = Field(description="知识点路径列表")


# 课程 CRUD

@router.get("")
# 依赖注入：获取数据库异步会话、获取当前登录用户（下划线开头表示该变量仅作权限拦截，函数内不使用）
async def list_courses(
        session: AsyncSession = Depends(get_session),
        _user: User = Depends(get_current_user),
) -> list[CourseOut]:
    """获取课程列表"""
    result = await session.execute(select(Course).order_by(Course.created_at.desc()))
    return [
        CourseOut(
            id=str(course.id),
            name=course.name,
            description=course.description,
            created_by=str(course.created_by),
            created_at=course.created_at.isoformat(),
        )
        for course in result.scalars()
    ]


@router.post("", status_code=201)
# 依赖注入：获取请求体、数据库会话、校验当前用户必须是超级用户
async def create_course(
        body: CourseCreate,
        session: AsyncSession = Depends(get_session),
        user: User = Depends(require_superuser),
) -> CourseOut:
    """创建课程"""
    course = Course(
        name=body.name,
        description=body.description,
        created_by=user.id,
    )

    session.add(course)
    await session.flush()
    await session.refresh(course)
    return CourseOut(
        id=str(course.id),
        name=course.name,
        description=course.description,
        created_by=str(course.created_by),
        created_at=course.created_at.isoformat(),
    )

@router.get("/{course_id}")
async def get_course(
        course_id: UUID,
        session: AsyncSession = Depends(get_session),
        _user: User = Depends(get_current_user),
):
    """获取课程详情"""
    course = await _get_course_or_404(session, course_id)
    return CourseOut(
        id=str(course.id),
        name=course.name,
        description=course.description,
        created_by=str(course.created_by),
        created_at=course.created_at.isoformat(),
    )



@router.delete("/{course_id}")
async def delete_course(
        course_id: UUID,
        session: AsyncSession = Depends(get_session),
        user: User = Depends(require_superuser),
) -> dict:
    """删除课程（级联删除所有相关数据 + 文件）"""
    course = await _get_course_or_404(session, course_id)

    # 先删文件
    file_store.delete_course_files(str(course.id))

    await session.delete(course)
    await session.flush()
    return {"deleted": "True"}


# ========== 文件上传与资料管理 ===========

# 状态码202：请求已接受，但未立即完成。
@router.post("/upload", status_code=202)
async def upload_file(
        file: UploadFile = File(...),
        course_id: UUID = Form(...),
        session: AsyncSession = Depends(get_session),
        user: User = Depends(require_superuser),
) -> dict:
    """上传课程资料并触发 ingestion

    支持格式：pdf，docx，md
    文件保存到本地后，异步触发 ingestion pipeline
    """
    # 1. 校验格式
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ["pdf", "docx", "md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: .{ext}，仅支持 pdf/docx/md"
        )

    # 2. 校验课程存在
    course = await _get_course_or_404(session, course_id)

    # 3. 保存文件
    content = await file.read()
    file_info = file_store.save(content, str(course.id), filename)

    # 4. 创建 Document 记录（status=pending）
    doc = Document(
        course_id=course.id,
        filename=filename,
        file_type=ext,
        file_size=file_info["file_size"],
        file_path=file_info["file_path"],
        uploader_id=user.id,
        status="pending"
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)

    # 5. 返回 202，ingestion 在后台执行
    #    注意：MVP 阶段先同步执行（简单可靠），后续可改为 BackgroundTasks
    try:
        from coursepilot.ingestion.pipeline import run_ingestion
        await run_ingestion(session, str(doc.id))
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)
        await session.flush()
        # 不抛异常，让 upload 请求正常返回，用户可在 documents 列表中看到 failed 状态

    return {
        "document_id": str(doc.id),
        "status": doc.status,
        "filename": filename,
    }

@router.get("/{course_id}/documents")
async def list_documents(
        course_id: UUID,
        session: AsyncSession = Depends(get_session),
        _user: User = Depends(get_current_user),
) -> list[DocumentOut]:
    """获取某课程下的资料列表"""
    await _get_course_or_404(session, course_id)
    result = await session.execute(
        select(Document)
        .where(Document.course_id == course_id)
        .order_by(Document.created_at.desc())
    )
    return [
        DocumentOut(
            id=str(d.id), filename=d.filename, file_type=d.file_type,
            file_size=d.file_size, status=d.status, page_count=d.page_count,
            uploaded_at=d.created_at.isoformat(),
        )
        for d in result.scalars()
    ]

@router.delete("/{course_id}/document/{document_id}")
async def delete_document(
        course_id: UUID,
        document_id: UUID,
        session: AsyncSession = Depends(get_session),
        user: User = Depends(require_superuser)
) -> dict:
    """删除某份资料（同时删文件 + DB 知识单元 + Milvus 向量）"""
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.course_id == course_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="资料不存在")

    # 删除关联的知识单元和 Milvus 向量
    ku_result = await session.execute(
        select(KnowledgeUnit.id).where(KnowledgeUnit.document_id == document_id)
    )
    ku_ids = [row[0] for row in ku_result.fetchall()]
    if ku_ids:
        # 从 Milvus 删除向量
        vector_store = VectorStore()
        vector_store.delete_by_uuids([str(uid) for uid in ku_ids])
        # 从 DB 删除知识单元
        await session.execute(
            KnowledgeUnit.__table__.delete().where(KnowledgeUnit.document_id == document_id)
        )

    file_store.delete(doc.file_path)
    await session.delete(doc)
    await session.flush()
    return {"deleted": True, "knowledge_units_deleted": len(ku_ids)}

# ========== 知识点树查询 ==========

@router.get("/{course_id}/knowledge-points")
async def get_knowledge_points(
        course_id: UUID,
        session: AsyncSession = Depends(get_session),
        _user: User = Depends(get_current_user),
) -> list[dict]:
    """获取课程的知识点树（扁平列表，含 parent_id 便于前端渲染）"""
    result = await session.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.course_id == course_id)
        .order_by(KnowledgePoint.sort_order)
    )
    kps = result.scalars().all()
    return [
        {
            "id": str(kp.id),
            "parent_id": str(kp.parent_id) if kp.parent_id else None,
            "kp_path": kp.kp_path,
            "title": kp.title,
            "summary": kp.summary,
            "difficulty": kp.difficulty,
            "sort_order": kp.sort_order,
        }
        for kp in kps
    ]


async def _get_course_or_404(session, course_id):
    result = await session.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")
    return course

# ========== RAG 问答 ==========

@router.post("/{course_id}/ask")
async def ask_course(
    course_id: UUID,
    body: AskRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> AskResponse:
    """RAG 问答：检索教材内容 → LLM 生成回答"""
    from coursepilot.rag.retriever import Retriever
    from coursepilot.rag.generator import Generator, build_course_context
    from coursepilot.rag.citation import extract_citations
    from coursepilot.rag.logger import QueryLogger

    await _get_course_or_404(session, course_id)

    qlogger = QueryLogger()
    trace_id, t_start = qlogger.start_trace()
    stages: dict[str, float] = {}

    # 阶段 1-4：检索
    t0 = time.time()
    retriever = Retriever()
    context, metadata = await retriever.retrieve(
        session, body.question, str(course_id)
    )
    stages["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

    # 课程上下文
    course_context = await build_course_context(session, course_id)

    # 阶段 5：LLM 生成
    t0 = time.time()
    generator = Generator()
    answer, _ = await generator.generate(body.question, context, course_context)
    stages["generate_ms"] = round((time.time() - t0) * 1000, 1)

    # 引用提取
    citations = extract_citations(answer)

    # 结构化日志
    qlogger.log_query(
        trace_id=trace_id,
        user_id=str(user.id),
        course_id=str(course_id),
        query_raw=body.question,
        query_rewritten=metadata["query_rewritten"],
        stages=stages,
        top_rerank_scores=metadata["top_rerank_scores"],
        source_kp_paths=metadata["source_kp_paths"],
        citation_count=len(citations),
        answer_length=len(answer),
    )

    return AskResponse(
        answer=answer,
        trace_id=trace_id,
        rewritten_query=metadata["query_rewritten"],
        citations=citations,
        top_scores=metadata["top_rerank_scores"],
        source_kp_paths=metadata["source_kp_paths"],
    )

@router.post("/{course_id}/ask/stream")
async def ask_course_stream(
    course_id: UUID,
    body: AskRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """RAG 问答（SSE 流式输出）"""
    from coursepilot.rag.retriever import Retriever
    from coursepilot.rag.generator import Generator, build_course_context
    from fastapi.responses import StreamingResponse

    await _get_course_or_404(session, course_id)

    retriever = Retriever()
    context, _metadata = await retriever.retrieve(
        session, body.question, str(course_id)
    )

    course_context = await build_course_context(session, course_id)
    generator = Generator()

    async def event_stream():
        async for token in generator.generate_stream(
            body.question, context, course_context
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

