"""L3 语义记忆抽取器。

从单轮或多轮 QA 中抽取结构化学习事实，写入 user_profile：
- 已掌握知识点（带置信度）
- 薄弱知识点/常见错误模式
- 学习偏好/风格

所有事实带 provenance（来源 session_id / qa_id），便于可观测与溯源。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.config import settings
from coursepilot.db import async_session_factory
from coursepilot.models import QARecord, UserProfile

logger = logging.getLogger(__name__)


_EXTRACTION_SYSTEM = """你是一个学习记忆抽取专家。请从师生问答记录中抽取结构化学习事实，输出 JSON。

输出格式（纯 JSON，不要 markdown）：
{
  "mastered_kps": [
    {"kp_path": "知识点路径", "confidence": 0.85, "evidence": "学生回答正确的证据摘要"}
  ],
  "weak_kps": [
    {"kp_path": "知识点路径", "confidence": 0.75, "evidence": "学生表现出困惑或错误的证据摘要"}
  ],
  "common_mistakes": [
    {"category": "错误类别", "pattern": "具体模式描述", "count": 1, "example": "错误示例"}
  ],
  "learning_style": {
    "style": "visual|textual|practice|mixed",
    "evidence": "判断依据摘要"
  }
}

要求：
- 只输出确实能从对话中推断出的事实，不要编造
- confidence 取值 0.0-1.0
- evidence 必须引用对话原文片段
- 如果没有足够证据，对应字段返回空数组/null
- 使用中文"""


async def extract_facts_from_qa(
    session: AsyncSession,
    qa_record: QARecord,
) -> dict[str, Any] | None:
    """对单条 QARecord 调用 LLM 抽取学习事实。

    返回结构化事实字典，失败返回 None。
    """
    if not settings.llm_api_key:
        logger.warning("LLM API key 未配置，跳过 L3 记忆抽取")
        return None

    prompt = _build_extraction_prompt(qa_record)
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception:
        logger.exception("L3 抽取 QARecord=%s 失败", qa_record.id)
        return None


def _build_extraction_prompt(qa: QARecord) -> str:
    """构造抽取 prompt。"""
    parts = [
        f"知识点路径：{qa.kp_path or '未知'}",
        f"问题：{qa.query}",
        f"回答：{qa.answer[:1200]}",
    ]
    if qa.citations:
        parts.append(f"引用来源：{qa.citations}")
    return "\n\n".join(parts)


def _merge_facts(
    existing: dict[str, Any],
    new_facts: dict[str, Any],
    qa_id: str,
    session_id: str,
) -> dict[str, Any]:
    """把新抽取的事实合并到已有的 profile 记忆结构中，带 provenance。"""
    merged = {
        "mastered_kps": list(existing.get("mastered_kps", [])),
        "weak_kps": list(existing.get("weak_kps", [])),
        "common_mistakes": list(existing.get("common_mistakes", [])),
        "learning_style": existing.get("learning_style"),
    }

    for item in new_facts.get("mastered_kps", []):
        item["provenance"] = {"qa_id": qa_id, "session_id": session_id}
        merged["mastered_kps"].append(item)

    for item in new_facts.get("weak_kps", []):
        item["provenance"] = {"qa_id": qa_id, "session_id": session_id}
        merged["weak_kps"].append(item)

    for item in new_facts.get("common_mistakes", []):
        item["provenance"] = {"qa_id": qa_id, "session_id": session_id}
        # 简单合并：同 category+pattern 增加 count
        found = False
        for existing_mistake in merged["common_mistakes"]:
            if (
                existing_mistake.get("category") == item.get("category")
                and existing_mistake.get("pattern") == item.get("pattern")
            ):
                existing_mistake["count"] = existing_mistake.get("count", 1) + item.get("count", 1)
                existing_mistake.setdefault("provenance", []).append({"qa_id": qa_id, "session_id": session_id})
                found = True
                break
        if not found:
            merged["common_mistakes"].append(item)

    if new_facts.get("learning_style") and new_facts["learning_style"].get("style"):
        merged["learning_style"] = new_facts["learning_style"]
        merged["learning_style"]["provenance"] = {"qa_id": qa_id, "session_id": session_id}

    # 控制列表长度，避免无限增长
    merged["mastered_kps"] = merged["mastered_kps"][-20:]
    merged["weak_kps"] = merged["weak_kps"][-20:]
    merged["common_mistakes"] = merged["common_mistakes"][-20:]

    return merged


async def update_profile_with_facts(
    session: AsyncSession,
    user_id: str,
    course_id: str,
    qa_record: QARecord,
) -> None:
    """把 QARecord 抽取的事实合并进 UserProfile.memory_facts（JSONB）。"""
    facts = await extract_facts_from_qa(session, qa_record)
    if not facts:
        return

    result = await session.execute(
        select(UserProfile).where(
            UserProfile.user_id == qa_record.user_id,
            UserProfile.course_id == qa_record.course_id,
        )
    )
    profile = result.scalar_one_or_none()

    qa_id_str = str(qa_record.id)
    session_id_str = str(qa_record.session_id) if qa_record.session_id else ""

    if profile:
        existing = profile.memory_facts or {}
        profile.memory_facts = _merge_facts(existing, facts, qa_id_str, session_id_str)
    else:
        profile = UserProfile(
            user_id=qa_record.user_id,
            course_id=qa_record.course_id,
            memory_facts=_merge_facts({}, facts, qa_id_str, session_id_str),
        )
        session.add(profile)

    await session.flush()
    logger.info("L3 记忆已更新 user=%s course=%s qa=%s", user_id, course_id, qa_id_str)


async def extract_facts_for_session(
    user_id: str,
    course_id: str,
    session_id: str,
) -> None:
    """后台任务：为某次会话的所有 QARecord 抽取 L3 事实。

    由 finalize_node 异步触发，失败不影响主流程。
    """
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(QARecord).where(
                    QARecord.user_id == user_id,
                    QARecord.course_id == course_id,
                    QARecord.session_id == session_id,
                )
            )
            records = result.scalars().all()
            for qa in records:
                await update_profile_with_facts(session, user_id, course_id, qa)
            await session.commit()
    except Exception:
        logger.exception("L3 抽取任务失败 user=%s course=%s session=%s", user_id, course_id, session_id)
