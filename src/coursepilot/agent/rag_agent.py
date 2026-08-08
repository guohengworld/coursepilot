"""Agentic RAG 核心模块：LLM 自主决策的 ReAct 循环（Harness 运行时）。

职责边界：
- Harness（本模块）：循环、工具分发、Guardrails、证据注册表、上下文注入、可观测
- LLM（Agent Core）：每步通过 Function Calling 决定调哪个工具，或停止生成

闭环契约（finalize 必须产出）：answer / context / sources / citation_map / degraded_mode / llm_calls。
对应方案：docs/Agentic_RAG_实现方案.md（v2.1）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

from openai import AsyncOpenAI

from coursepilot.agent.skills.check_sufficiency import check_sufficiency
from coursepilot.agent.skills.decompose_query import decompose_query
from coursepilot.agent.skills.web_search import format_web_context, web_search
from coursepilot.config import settings
from coursepilot.db import async_session_factory
from coursepilot.rag.config import config as rag_config

logger = logging.getLogger(__name__)

_DEGRADED_DISCLAIMER = (
    "[注意] 以下教材内容可能不足以完整回答该问题。"
    "请结合教材原文和课堂笔记使用，以下回答仅供参考。\n"
)

# ── Agent 系统提示词 ──────────────────────────────────────
AGENT_REACT_SYSTEM_PROMPT = """你是 CoursePilot 的数学答疑助手。针对学生的提问，你通过调用工具自主决定"下一步干什么"，
而不是一次性直接给出答案。每轮你只能做一件事：调用一个工具，或（证据足够时）直接输出最终回答。

可用工具：
- search_textbook(query, top_k)：在当前课程教材中检索知识点内容（概念定义、定理、公式、推导）。默认首选。
- plan(query, sub_questions)：把复杂问题拆解为多个可独立检索的子问题（比较类、多步推理、跨章节问题）。调用后系统会自动并行检索所有子问题并返回结果，无需再逐个 search_textbook。
- web_search(query)：教材检索不足时搜索互联网补充。成本较高，最后考虑。
- memory_recall(query)：检索学生在本课程的历史问答记录。涉及"我上次问的"等个性化场景时使用。
- evaluate_context(question, evidence)：评估当前已收集的证据是否足以回答问题。不确定证据是否足够时调用。

检索策略（搜索无果或结果不相关时）：
- rewrite：换同义词改写查询后重试
- expand：把过窄的查询扩展为多个相关关键词
- narrow：结果太泛时收窄范围
- decompose：把问题拆成子问题逐个检索（等价于调用 plan）
- pivot：根据已发现的信息转向新的搜索角度

停止语义：当你认为当前证据已足以回答问题时，不要再调用任何工具，直接输出最终回答（引用教材片段时可带 <ref id="N">）。
不要重复检索同一个 query；若收到 guardrail 拒绝提示，请改写查询或换用其他工具。"""


# ── 工具定义（OpenAI Function Calling JSON Schema，方案 §5） ──
SEARCH_TEXTBOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "search_textbook",
        "description": "在当前课程教材中检索与问题相关的知识点内容。返回教材原文片段（含知识点路径与页码）。用于查找概念定义、定理、公式、推导步骤等教材内容。对需要精确术语或代码的问题也可用。若检索结果为空或明显不相关，可换用同义词改写 query 后再试，或改用 web_search。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索的查询词，建议与教材术语保持一致",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的教材片段数量，默认 5",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": "将复杂问题拆解为多个独立的、可单独检索的子问题。适用于：多概念比较（A 和 B 的区别）、多步推理、跨章节问题、含假设与结论两部分的问题。调用后系统会自动并行检索你提供的每个子问题并返回全部检索结果，无需再逐个调用 search_textbook。对单一概念问题不要调用，直接 search_textbook。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "待拆解的原始问题",
                },
                "sub_questions": {
                    "type": "array",
                    "description": "拆解出的子问题列表，每个子问题应能通过一次教材检索回答",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "子问题表述"},
                            "target_concept": {"type": "string", "description": "对应知识点"},
                            "reason": {"type": "string", "description": "为什么需要这个子问题"},
                        },
                        "required": ["question", "target_concept", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["query", "sub_questions"],
            "additionalProperties": False,
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "当教材检索结果不足以回答问题时，搜索互联网获取补充资料。仅在教材确实未覆盖或需要额外示例时使用，成本较高，应最后考虑。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

MEMORY_RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_recall",
        "description": "检索该学生在本课程的历史问答记录。适用于学生问及之前讨论过的内容（'我上次问的那个'）、或需要基于学生历史掌握情况个性化回答时。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要匹配的历史内容关键词"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

EVALUATE_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_context",
        "description": "评估当前已收集的证据是否足以回答用户问题。返回各维度评分（覆盖度/一致性/时效性/权威性/完整性），以及缺失信息清单。当你不确定证据是否足够时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "需要评估的用户原始问题"},
                "evidence": {"type": "string", "description": "当前已收集的证据摘要"},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

SUMMARIZE_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "summarize_context",
        "description": "把当前已收集的早期证据压缩为一段摘要，释放上下文空间。当证据过多、重复，或回复中提示 token 预算紧张时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "触发压缩的原因（可选，供审计）"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict] = [
    SEARCH_TEXTBOOK_TOOL,
    PLAN_TOOL,
    WEB_SEARCH_TOOL,
    MEMORY_RECALL_TOOL,
    EVALUATE_CONTEXT_TOOL,
    SUMMARIZE_CONTEXT_TOOL,
]

# ── 多维评估（P2.1：覆盖/一致性/时效/权威/完整，作为工具而非必经节点） ──
EVALUATE_MULTIDIM_SYSTEM = """你是 RAG 证据质量评估员。请从 5 个维度评估给定的【证据】对回答【用户问题】的支撑质量，每个维度输出 0.0~1.0 的分数：
1. coverage（覆盖度）：关键知识点、术语、概念是否都被证据覆盖
2. consistency（一致性）：不同证据片段之间是否存在矛盾或冲突
3. timeliness（时效性）：内容是否过时（教材类内容通常为 1.0；网络资料需注意时效）
4. authority（权威性）：来源是否权威（教材原文 > 网络搜索 > 记忆召回）
5. completeness（完整性）：推导步骤、论证过程是否完整，还是仅有结论

输出 JSON，不要包含其他内容：
{
  "coverage": 0.0~1.0,
  "consistency": 0.0~1.0,
  "timeliness": 0.0~1.0,
  "authority": 0.0~1.0,
  "completeness": 0.0~1.0,
  "weaknesses": ["不足维度对应的具体问题描述"]
}"""

_SUMMARY_TRIGGER_RATIO = 0.5  # token 预算用掉 50% 时，harness 自动触发早期证据压缩


# ── 证据注册表（方案 §5.8） ──────────────────────────────
_SOURCE_BLOCK_PAT = re.compile(r"(<source id=\"\d+\"[^>]*>.*?</source>)", re.DOTALL)
_SOURCE_TAG_PAT = re.compile(r'<source id="(\d+)"([^>]*)>')
_PATH_ATTR_PAT = re.compile(r'path="([^"]*)"')


class EvidenceRegistry:
    """全局证据与引用注册表：多轮工具结果合并的唯一出口。

    解决多轮检索产生多套 <source id="1"> 与 citation_map 的引用冲突问题：
    每次 register 把局部 ref_id 重写为全局递增 id，最终 merged_context /
    merged_citation_map 是唯一数据源（不变量：state["context"] 与
    state["retrieved_metadata"]["citation_map"] 必须来自这里）。
    """

    def __init__(self) -> None:
        self._blocks: list[str] = []          # 带全局 <source id> 的 context 片段
        self._block_ref_ids: list[list[str]] = []  # 每块对应分配的全局 ref_id 列表
        self.citation_map: dict[str, dict[str, Any]] = {}  # ref_id -> {uuid, kp_path, page_ref}
        self._next_ref = 1
        self._merged_ctx: str | None = None
        self._included_ref_ids: list[str] = []
        self._summarized = False

    def register(self, context_xml: str, metadata: dict[str, Any] | None = None) -> None:
        """登记一次工具结果，把局部 ref_id 重写为全局 id，并同步 citation_map。

        metadata 缺省（如 web 结果）时，从 <source> 的 path 属性回退取 kp_path。
        """
        if not context_xml or not context_xml.strip():
            return
        metadata = metadata or {}
        local_map = metadata.get("citation_map") or {}

        blocks = _SOURCE_BLOCK_PAT.findall(context_xml)
        if not blocks:
            blocks = [context_xml]

        rewritten: list[str] = []
        ref_ids_per_block: list[list[str]] = []
        for block in blocks:
            m = _SOURCE_TAG_PAT.search(block)
            if not m:
                rewritten.append(block)
                ref_ids_per_block.append([])
                continue
            local_id = m.group(1)
            tag_rest = m.group(2)
            new_ref = str(self._next_ref)
            self._next_ref += 1
            new_block = re.sub(r'<source id="\d+"', f'<source id="{new_ref}"', block, count=1)
            rewritten.append(new_block)

            # 合并 citation_map：优先用 metadata，缺省时从 path 属性回退
            local_meta = local_map.get(local_id)
            if not local_meta:
                path_m = _PATH_ATTR_PAT.search(tag_rest)
                local_meta = {"kp_path": path_m.group(1) if path_m else ""}
            self.citation_map[new_ref] = dict(local_meta)
            ref_ids_per_block.append([new_ref])

        self._blocks.extend(rewritten)
        # 保持 _block_ref_ids 与 _blocks 一一对应（每块一个列表），
        # 否则 _build_merged 按块索引 ref_id 时会越界
        self._block_ref_ids.extend(ref_ids_per_block)
        # 合并缓存失效
        self._merged_ctx = None
        self._included_ref_ids = []
        logger.info("EvidenceRegistry.register: +%d 块, 全局 ref_id 至 %d",
                    len(rewritten), self._next_ref - 1)

    def _build_merged(self) -> None:
        """整块截断（保持 XML 有效），按 config.context_max_chars 只保留前缀块。"""
        max_chars = rag_config.context_max_chars
        parts: list[str] = []
        included: list[str] = []
        total = 0
        for i, block in enumerate(self._blocks):
            if parts and total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
            included.extend(self._block_ref_ids[i])
        self._merged_ctx = "\n".join(parts)
        self._included_ref_ids = included

    def merged_context(self) -> str:
        """合并后的 context（供 Generator 生成 + guard_answer 校验）。"""
        if self._merged_ctx is None:
            self._build_merged()
        return self._merged_ctx or ""

    def merged_citation_map(self) -> dict[str, dict[str, Any]]:
        """只返回被包含块的引用映射（与 merged_context 截断一致）。"""
        if self._merged_ctx is None:
            self._build_merged()
        return {rid: self.citation_map[rid] for rid in self._included_ref_ids
                if rid in self.citation_map}

    def raw_blocks(self) -> list[str]:
        """原始证据块（含全局 ref_id），供 state["evidence"] 审计。"""
        return list(self._blocks)

    # ── P2.2 早期证据压缩（summarize_context） ──
    @property
    def summarized(self) -> bool:
        """是否已执行过一次早期证据压缩（每轮 agent 只压一次）。"""
        return self._summarized

    def can_summarize(self) -> bool:
        """存在 ≥2 块证据且尚未压缩过时，允许压缩早期证据。"""
        return not self._summarized and len(self._blocks) >= 2

    def summarize_early(self, count: int, summary_text: str) -> int:
        """把前 count 块证据替换为一段摘要（无 source 引用的纯文本），返回实际压缩块数。

        被压缩块的 citation_map 条目保留但不再出现在 merged_citation_map()
        （摘要块没有 <source> 标签，_included_ref_ids 不含它们）。
        """
        if count <= 0 or not self._blocks or self._summarized:
            return 0
        count = min(count, len(self._blocks))
        summary_block = f"<summary>早期证据摘要：{summary_text}</summary>"
        self._blocks = [summary_block] + self._blocks[count:]
        self._block_ref_ids = [[]] + self._block_ref_ids[count:]
        self._summarized = True
        self._merged_ctx = None
        self._included_ref_ids = []
        logger.info("EvidenceRegistry.summarize_early: 压缩 %d 块为摘要", count)
        return count


# ── Guardrails（方案 §8：确定性与"LLM 自主"的边界） ──────
class Guardrails:
    """harness 的确定性约束，LLM 只负责"选哪个工具"。

    - 步数上限：超限强制停止（forced_stop → degraded_mode）
    - web 次数上限：超限后拒绝后续 web_search 调用并提示换策略
    - 重复查询：相同 tool+query 已检索 ≥2 次，拒绝并提示改写
    - token 预算：累计超预算强制停止
    - 工具参数校验：必填参数缺失时拒绝执行，错误回传给 LLM
    """

    # 工具必填参数（与 TOOLS schema 的 required 一致；
    # evaluate_context 的 evidence 允许缺省，执行器会回退用已合并证据）
    _REQUIRED_ARGS = {
        "search_textbook": ["query"],
        "plan": ["query", "sub_questions"],
        "web_search": ["query"],
        "memory_recall": ["query"],
        "evaluate_context": ["question"],
        "summarize_context": [],
    }

    def __init__(self, *, max_steps: int, max_web_searches: int, token_budget: int) -> None:
        self.max_steps = max_steps
        self.max_web_searches = max_web_searches
        self.token_budget = token_budget
        self.step_count = 0
        self.web_count = 0
        self.token_used = 0

    def check(self) -> bool:
        """步数+1，超限（步数或 token 预算）返回 True（强制停止）。"""
        self.step_count += 1
        if self.step_count > self.max_steps:
            logger.warning("Guardrails: 步数超限 %d > %d，强制停止", self.step_count, self.max_steps)
            return True
        if self.token_used > self.token_budget:
            logger.warning("Guardrails: token 预算超限 %d > %d，强制停止",
                           self.token_used, self.token_budget)
            return True
        return False

    def accrue_tokens(self, tokens: int) -> None:
        self.token_used += tokens

    def before_tool(self, tool_name: str, args: dict, tool_history: list[dict]) -> str | None:
        """执行前校验。返回 None 表示放行；否则返回给 LLM 的拒绝提示文本。"""
        # 必填参数校验
        missing = [k for k in self._REQUIRED_ARGS.get(tool_name, [])
                   if not args.get(k)]
        if missing:
            return (f"guardrail：工具 {tool_name} 缺少必填参数 {missing}，"
                    "请补全参数后重试。")

        # web 次数上限
        if tool_name == "web_search":
            if self.web_count >= self.max_web_searches:
                return (f"guardrail：web_search 已达上限 {self.max_web_searches} 次，"
                        "请改用 search_textbook 或改写查询，不要继续 web_search。")
            self.web_count += 1

        # 重复 query：相同 tool+query 已检索 ≥2 次
        same = [h for h in tool_history
                if h.get("tool") == tool_name
                and h.get("args", {}).get("query") == args.get("query")]
        if len(same) >= 2:
            return (f"guardrail：query「{args.get('query')}」已被 {tool_name} 检索过 "
                    f"{len(same) + 1} 次，请改写查询或换用其他工具。")

        return None


# ── 会话消息组装 ──────────────────────────────────────────
def _recent_turns(conversation: list[dict], current_query: str,
                  max_turns: int = 8) -> list[dict[str, str]]:
    """提取最近 user/assistant 轮次，去掉与当前 query 重复的末尾 user 轮。"""
    turns: list[dict[str, str]] = []
    for msg in conversation or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str):
            turns.append({"role": role, "content": content})
    # 去掉重复的末尾 user 轮（当前 query 已在 messages 末尾单独注入）
    while turns and turns[-1]["role"] == "user" and turns[-1]["content"] == current_query:
        turns.pop()
    return turns[-max_turns:]


def build_agent_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    """组装 Agent 循环的初始 messages：system + 历史 + 当前 query。

    system 注入课程名/章节（供 Agent 规划检索范围）与学生画像（供个性化）。
    """
    parts = [AGENT_REACT_SYSTEM_PROMPT]
    course_ctx = state.get("course_context") or {}
    if course_ctx.get("name"):
        line = f"当前课程：{course_ctx['name']}"
        chapters = course_ctx.get("chapters") or []
        if chapters:
            line += f"，已学章节：{'、'.join(str(c)[:20] for c in chapters[:10])}"
        parts.append(line)
    user_profile = state.get("user_profile")
    if user_profile:
        parts.append(f"学生画像：{json.dumps(user_profile, ensure_ascii=False)[:400]}")

    messages: list[dict[str, str]] = [{"role": "system", "content": "\n".join(parts)}]

    rolling_summary = state.get("rolling_summary", "")
    if rolling_summary:
        messages.append({"role": "system", "content": f"以下是对话历史摘要：\n{rolling_summary}"})

    conversation = state.get("conversation") or state.get("recent_qa") or []
    messages.extend(_recent_turns(conversation, state["query"]))

    messages.append({"role": "user", "content": state["query"]})
    return messages


def _truncate(text: str, max_chars: int) -> str:
    """截断超长文本，保留尾部提示。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...（已截断，原始 {len(text)} 字符）"


# ── 工具执行器（方案 §5：包装现有技能，零重写） ────────────
async def _retrieve_textbook(
    query: str,
    course_id: str,
    top_k: int | None = None,
) -> tuple[str, dict]:
    """教材检索（Retriever 六阶段管线原样复用）。

    Retriever.retrieve 不接受 top_k 参数：
    - top_k 指定时通过临时覆盖 rag_config.rerank_top_k 实现（复用 query_rag_node
      简单通道的既有模式）；
    - top_k 为 None 时直接用默认配置、不修改全局状态——可安全并行调用（否则
      并发协程互相覆盖/恢复 rag_config.rerank_top_k 会产生竞态）。
    """
    from coursepilot.rag.retriever import Retriever

    if top_k is None:
        async with async_session_factory() as session:
            return await Retriever().retrieve(session, query, course_id)

    saved = rag_config.rerank_top_k
    rag_config.rerank_top_k = top_k
    try:
        async with async_session_factory() as session:
            return await Retriever().retrieve(session, query, course_id)
    finally:
        rag_config.rerank_top_k = saved


async def _tool_search_textbook(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """search_textbook：教材检索（Retriever 六阶段管线原样复用）。"""
    query = args["query"]
    top_k = min(int(args.get("top_k", 5)), 10)
    context, metadata = await _retrieve_textbook(query, state["course_id"], top_k=top_k)
    if not context.strip():
        return "search_textbook：未检索到相关内容。可尝试改写查询，或改用 web_search。"

    evidence.register(context, metadata)
    return f"search_textbook 结果（已登记证据，可用 <ref id=\"N\"> 引用）：\n{_truncate(context, 3000)}"


def _normalize_sub_queries(
    llm_sub_questions: Any,
    decomposed: dict[str, Any],
) -> list[dict[str, Any]]:
    """归一化子问题列表：优先 LLM 传入（schema: question/target_concept/reason），
    无效时回退 decompose_query 结果（schema: query/target_concept/reason）。"""
    candidates: list[dict[str, Any]] | None = None
    if isinstance(llm_sub_questions, list) and llm_sub_questions:
        candidates = llm_sub_questions
    elif decomposed.get("sub_queries"):
        candidates = decomposed["sub_queries"]

    normalized: list[dict[str, Any]] = []
    for sq in candidates or []:
        if not isinstance(sq, dict):
            continue
        q = sq.get("question") or sq.get("query")
        if not q:
            continue
        normalized.append({
            "query": q,
            "target_concept": sq.get("target_concept", ""),
            "reason": sq.get("reason", ""),
        })
    return normalized


async def _tool_plan(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """plan：复杂问题拆解（LLM 传入 sub_questions 优先，回退 decompose_query）。

    分解出的子问题由 harness 异步并行检索教材（asyncio.gather），全部收束后
    按子问题顺序汇总结果并登记证据——LLM 无需再逐个调用 search_textbook。
    """
    query = args["query"]
    llm_sub = args.get("sub_questions")
    decomposed: dict[str, Any] = {}
    if not (isinstance(llm_sub, list) and llm_sub):
        decomposed = await decompose_query(query, state.get("course_context"))
    sub_queries = _normalize_sub_queries(llm_sub, decomposed)
    if not sub_queries:
        return "plan：该问题无需分解，请直接调用 search_textbook 检索。"

    async def retrieve_one(idx: int, sq: dict) -> str:
        """单个子问题检索：异常/空结果独立降级，不阻塞其余子问题（并行收束）。"""
        try:
            context, metadata = await _retrieve_textbook(sq["query"], state["course_id"])
        except Exception as e:
            logger.exception("plan 子问题检索异常: %s", sq["query"])
            return f"[子问题 {idx} 检索失败]：{e}"
        if not context.strip():
            return f"[子问题 {idx} 检索结果]：未检索到相关内容。"
        evidence.register(context, metadata)
        return f"[子问题 {idx} 检索结果]：\n{_truncate(context, 1500)}"

    # 异步并行检索所有子问题；gather 天然收束（全部完成后才继续）
    results = await asyncio.gather(
        *(retrieve_one(idx, sq) for idx, sq in enumerate(sub_queries, start=1))
    )

    lines = [
        f"plan 分解结果（{len(sub_queries)} 个子问题，已并行检索教材）：",
        *[f"- [{idx}] {sq['query']}（目标知识点：{sq.get('target_concept', '')}）"
          for idx, sq in enumerate(sub_queries, start=1)],
        "你可以基于以上全部检索结果直接综合回答；个别子问题资料不足时，可补充 web_search 或 memory_recall。",
    ]
    lines.extend(results)
    return "\n".join(lines)


async def _tool_web_search(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """web_search：网络搜索（多引擎合并，format_web_context 格式化后登记证据）。"""
    query = args["query"]
    results = await web_search(query, top_k=5)
    if not results:
        return "web_search：未找到有效结果。建议回到教材检索。"
    context = format_web_context(results, query)
    evidence.register(context, {})
    return f"web_search 结果（已登记证据）：\n{_truncate(context, 3000)}"


async def _tool_memory_recall(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """memory_recall：历史问答召回（QARecord L4 语义召回）。"""
    query = args["query"]
    from coursepilot.agent.memory.retriever import recall_memory_turns

    async with async_session_factory() as session:
        records = await recall_memory_turns(
            session, state["user_id"], state["course_id"], query, top_k=5,
        )
    if not records:
        return "memory_recall：未找到与该查询相关的历史问答记录。"
    lines = ["memory_recall 结果："]
    for r in records:
        lines.append(f"- [{r.get('qa_id')}] 问：{r.get('query', '')}\n"
                     f"    答：{str(r.get('answer', ''))[:200]}")
    return "\n".join(lines)


async def evaluate_multidim(query: str, context: str) -> dict[str, Any]:
    """P2.1 五维证据质量评估（覆盖/一致性/时效/权威/完整）。

    LLM 无 API key 或解析失败时返回空 dict（由调用方决定降级展示）。
    """
    if not settings.llm_api_key:
        logger.warning("evaluate_multidim: LLM API key 未配置，返回空评分")
        return {}
    try:
        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EVALUATE_MULTIDIM_SYSTEM},
                {"role": "user", "content": f"【用户问题】\n{query}"
                 f"\n\n【证据】\n{_truncate(context, 4000)}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw.strip())
        if not isinstance(data, dict):
            return {}
        return {
            k: float(data.get(k, 0.0))
            for k in ("coverage", "consistency", "timeliness", "authority", "completeness")
        } | {"weaknesses": data.get("weaknesses", []) if isinstance(data.get("weaknesses"), list) else []}
    except Exception as e:
        logger.warning("evaluate_multidim 失败: %s", e)
        return {}


async def _tool_evaluate_context(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """evaluate_context：证据充分性评估（check_sufficiency 规则+LLM）+ P2.1 五维评分。"""
    question = args["question"]
    evidence_text = args.get("evidence") or evidence.merged_context()
    if not evidence_text.strip():
        return ("evaluate_context：当前尚无任何已收集证据，"
                "请先调用 search_textbook 检索教材。")

    kp_paths = [v.get("kp_path", "") for v in evidence.merged_citation_map().values()
                if v.get("kp_path")]
    result = await check_sufficiency(question, evidence_text, kp_paths)

    sufficient = result.get("sufficient", True)
    confidence = result.get("confidence", 0.0)
    lines = [f"evaluate_context 评估结果：sufficient={sufficient}，confidence={confidence:.2f}"]
    if result.get("missing_info"):
        lines.append(f"缺失信息：{result['missing_info']}")
    if result.get("uncovered_aspects"):
        lines.append(f"未覆盖方面：{'；'.join(result['uncovered_aspects'])}")

    # P2.1: 五维评分（无 key / 失败时静默省略该段，不阻塞主结论）
    multidim = await evaluate_multidim(question, evidence_text)
    if multidim:
        labels = {
            "coverage": "覆盖度", "consistency": "一致性", "timeliness": "时效性",
            "authority": "权威性", "completeness": "完整性",
        }
        scores = "，".join(
            f"{labels[k]} {multidim.get(k, 0.0):.2f}"
            for k in labels if k in multidim
        )
        lines.append(f"多维评分（P2.1）：{scores}")
        weaknesses = multidim.get("weaknesses") or []
        if weaknesses:
            lines.append(f"薄弱点：{'；'.join(str(w)[:80] for w in weaknesses[:3])}")

    if sufficient:
        lines.append("结论：证据已足够，可以停止检索并输出最终回答。")
    else:
        lines.append("结论：证据不足，请继续检索（改写查询、检索缺失概念，或最后考虑 web_search）。")
    return "\n".join(lines)


async def _summarize_evidence(evidence: EvidenceRegistry) -> str:
    """P2.2 用 LLM 生成早期证据的摘要（被压缩块 = 前一半的已登记块）。"""
    blocks = evidence.raw_blocks()
    if not blocks:
        return ""
    # 压缩前一半块；保留 <source> 标签会让摘要引用错位，这里剥离标签只取文本
    early = _truncate("\n".join(blocks[: max(1, len(blocks) // 2)]), 6000)
    if not settings.llm_api_key:
        return f"（无 LLM key，摘要省略）共 {len(blocks)} 块证据"
    try:
        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": (
                    "把下面的教材/资料证据片段压缩成一段 200 字以内的中文摘要，"
                    "保留与问题可能相关的关键事实、定义、定理，去掉重复与无关内容。"
                    "直接输出摘要正文，不要加任何前缀。")},
                {"role": "user", "content": early},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("_summarize_evidence 失败: %s", e)
        return f"（摘要生成失败：{e}）"


async def _tool_summarize_context(args: dict, state: dict, evidence: EvidenceRegistry) -> str:
    """summarize_context：压缩早期证据，释放上下文空间。"""
    if not evidence.can_summarize():
        return ("summarize_context：当前证据不足 2 块或已压缩过，无需压缩。"
                "请先调用 search_textbook 检索教材。")
    blocks = evidence.raw_blocks()
    compress_count = max(1, len(blocks) // 2)
    summary = await _summarize_evidence(evidence)
    compressed = evidence.summarize_early(compress_count, summary)
    reason = args.get("reason", "")
    return (f"summarize_context：已把前 {compressed} 块早期证据压缩为摘要"
            f"（触发原因：{reason or 'LLM 自主决定'}）。\n"
            f"摘要：{summary}")


async def _maybe_auto_summarize(
    messages: list[dict], guard: Guardrails, evidence: EvidenceRegistry,
) -> None:
    """P2.2 harness 自动触发：token 预算用掉一半且证据未压缩时压缩早期证据。

    确定性 harness 行为（不经 LLM 决策），把压缩结果以 system 消息注入。
    """
    if (guard.token_used < guard.token_budget * _SUMMARY_TRIGGER_RATIO
            or not evidence.can_summarize()):
        return
    summary = await _summarize_evidence(evidence)
    compressed = evidence.summarize_early(
        max(1, len(evidence.raw_blocks()) // 2), summary,
    )
    logger.info("_maybe_auto_summarize: token=%d/%d，自动压缩 %d 块早期证据",
                guard.token_used, guard.token_budget, compressed)
    messages.append({"role": "system", "content":
        f"harness：早期 {compressed} 块证据已压缩为摘要以节省 token。"
        f"摘要：{_truncate(summary, 500)}"})


# ── 工具分发 ──────────────────────────────────────────────
TOOL_DISPATCH: dict[str, Callable] = {
    "search_textbook": _tool_search_textbook,
    "plan": _tool_plan,
    "web_search": _tool_web_search,
    "memory_recall": _tool_memory_recall,
    "evaluate_context": _tool_evaluate_context,
    "summarize_context": _tool_summarize_context,
}


async def dispatch_tool(tool_name: str, args: dict, state: dict,
                        evidence: EvidenceRegistry) -> str:
    """分发工具调用。未知工具 / 执行异常均返回错误文本给 LLM（不静默）。"""
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return (f"错误：未知工具「{tool_name}」。可用工具："
                f"{', '.join(TOOL_DISPATCH)}，请换用其中一种。")
    try:
        return await handler(args, state, evidence)
    except Exception as e:
        logger.exception("工具 %s 执行异常: args=%s", tool_name, args)
        return f"错误：工具 {tool_name} 执行失败（{e}）。请改写查询或换用其他工具。"


# ── 闭环契约出口（方案 §6.4） ─────────────────────────────
async def finalize_answer(
    state: dict[str, Any],
    *,
    evidence: EvidenceRegistry,
    degraded: bool,
    llm_calls: list[dict],
    agent_steps: list[dict],
    tool_history: list[dict],
) -> dict[str, Any]:
    """产出闭环契约 6 字段：answer / context / sources / citation_map / degraded_mode / llm_calls。

    degraded（guardrail 强制停止）时，生成前在 context 前追加免责声明
    （原 synthesize_node 的降级逻辑迁入）。
    """
    from coursepilot.rag.generator import Generator

    context = evidence.merged_context()
    gen_context = context
    if degraded and context:
        gen_context = _DEGRADED_DISCLAIMER + context

    answer, token_info = await Generator().generate(
        query=state["query"],
        context=gen_context,
        course_context=state.get("course_context", {}),
        conversation=state.get("conversation"),
        rolling_summary=state.get("rolling_summary", ""),
        user_profile=state.get("user_profile"),
    )
    token_info["degraded_mode"] = degraded
    llm_calls.append({"node": "agent_finalize", **token_info})

    citation_map = evidence.merged_citation_map()
    kp_paths: list[str] = []
    for v in citation_map.values():
        p = v.get("kp_path", "")
        if p and p not in kp_paths:
            kp_paths.append(p)
    sources = [{"kp_path": p} for p in kp_paths]

    logger.info("finalize_answer: degraded=%s, sources=%d, total_llm_calls=%d",
                degraded, len(sources), len(llm_calls))

    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "retrieved_metadata": {
            "citation_map": citation_map,
            "source_kp_paths": kp_paths,
        },
        "degraded_mode": degraded,
        "llm_calls": llm_calls,
        "agent_steps": agent_steps,
        "tool_history": tool_history,
        "evidence": evidence.raw_blocks(),
        "error": None,
    }


# ── ReAct 循环入口节点（方案 §6.2） ───────────────────────
async def agentic_rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """Agentic RAG 入口节点：LLM 自主决策的 ReAct 循环。

    Harness 负责：循环、工具分发、Guardrails、证据注册表、上下文注入、可观测。
    LLM 负责：每步决定调哪个工具，或停止。
    """
    if not settings.llm_api_key:
        logger.error("agentic_rag: LLM API Key 未配置，无法执行")
        return {"error": "LLM API Key 未配置，无法执行 Agentic RAG"}

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    messages = build_agent_messages(state)
    guard = Guardrails(
        max_steps=rag_config.agent_max_steps,
        max_web_searches=rag_config.agent_max_web_searches,
        token_budget=rag_config.agent_token_budget,
    )
    evidence = EvidenceRegistry()
    llm_calls = list(state.get("llm_calls") or [])
    agent_steps = list(state.get("agent_steps") or [])
    tool_history = list(state.get("tool_history") or [])
    forced_stop = False

    while True:
        if guard.check():
            forced_stop = True
            break

        # P2.2: token 预算用掉一半且证据未压缩时，harness 自动压缩早期证据
        await _maybe_auto_summarize(messages, guard, evidence)

        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            logger.exception("agentic_rag: LLM 调用失败")
            return {"error": f"Agentic RAG LLM 调用失败：{e}"}

        usage = response.usage
        token_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        guard.accrue_tokens(token_info["total_tokens"])
        llm_calls.append({"node": "agent_step", **token_info})

        message = response.choices[0].message
        if not message.tool_calls:
            break  # LLM 决定停止 → 跳出循环

        messages.append(message)  # assistant 的 tool_calls 必须回传

        # P2.3: 先统一解析 + guardrail 校验，再并发执行合法的工具调用
        prepared: list[tuple[Any, str, dict, str | None]] = []
        for call in message.tool_calls:
            fn = call.function
            tool_name = fn.name
            try:
                args = json.loads(fn.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            agent_steps.append({"tool": tool_name, "args": fn.arguments})
            rejection = guard.before_tool(tool_name, args, tool_history)
            prepared.append((call, tool_name, args, rejection))

        valid = [(c, tn, a) for (c, tn, a, rj) in prepared if rj is None]
        results: list[str] = []
        if valid:
            results = await asyncio.gather(
                *(dispatch_tool(tn, a, state, evidence) for (_, tn, a) in valid)
            )
        result_iter = iter(results)

        # 按原始 tool_calls 顺序回传结果（并发执行、顺序组装，tool_call_id 严格匹配）
        for call, tool_name, args, rejection in prepared:
            if rejection is not None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": rejection,
                })
                continue
            result = next(result_iter)
            tool_history.append({
                "tool": tool_name,
                "args": args,
                "result_summary": result[:200],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return await finalize_answer(
        state=state,
        evidence=evidence,
        degraded=forced_stop,
        llm_calls=llm_calls,
        agent_steps=agent_steps,
        tool_history=tool_history,
    )
