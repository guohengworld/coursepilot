"""Agent 上下文记忆管理层。

提供 ContextManager 统一装配 LLM prompt 所需的多层记忆，以及 Compactor
负责历史对话的滚动压缩与 micro-compact。
"""

from coursepilot.agent.memory.compactor import compact_conversation, micro_compact_turn
from coursepilot.agent.memory.context_manager import ContextManager, ContextView
from coursepilot.agent.memory.extractor import extract_facts_for_session
from coursepilot.agent.memory.retriever import (
    recall_memory_turns,
    ensure_qa_embeddings,
    estimate_importance,
    score_memory_turn,
)

__all__ = [
    "ContextManager",
    "ContextView",
    "compact_conversation",
    "micro_compact_turn",
    "extract_facts_for_session",
    "recall_memory_turns",
    "ensure_qa_embeddings",
    "estimate_importance",
    "score_memory_turn",
]
