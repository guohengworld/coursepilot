"""ContextManager —— 企业级上下文预算装配器。

设计目标：
1. 为不同 LLM 调用节点提供差异化上下文视图（classify / query_rag / diagnose...）
2. 按 token 预算逐层装配，而不是无界拼接
3. 稳定内容放最前以提升 DeepSeek prompt caching 命中率
4. 滑动窗口 + 滚动摘要作为 L1/L2 记忆

使用 tiktoken (cl100k_base) 进行精确 token 计算，兼容 DeepSeek / GPT-4。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# 全局缓存 tiktoken 编码器，避免重复加载
_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """获取/缓存 tiktoken 编码器。"""
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODER_CACHE[encoding_name]


def estimate_tokens(text: str) -> int:
    """使用 tiktoken 精确估算 token 数（cl100k_base，兼容 DeepSeek / GPT-4）。

    如果 tiktoken 编码失败，回退到启发式估算。
    """
    if not text:
        return 0
    try:
        enc = _get_encoder()
        return len(enc.encode(text))
    except Exception:
        # 回退：启发式估算
        ascii_words = len(re.findall(r"[a-zA-Z0-9_]+", text))
        ascii_len = sum(len(w) for w in re.findall(r"[a-zA-Z0-9_]+", text))
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - ascii_len - cn_chars
        return int(ascii_words * 1.3 + cn_chars * 0.6 + max(0, other_chars) * 0.5)


@dataclass
class ContextView:
    """面向某一次 LLM 调用的完整上下文视图。

    各字段按 prompt 中的实际出现顺序排列，调用方可直接拼接 messages。
    """

    system_prefix: str                    # 固定系统提示（可被缓存）
    course_context: str                   # 课程上下文
    user_profile: str                     # L3 语义记忆摘要
    rolling_summary: str                  # L2 滚动摘要
    recent_turns: list[dict[str, str]]    # L1 最近轮次 [{role, content}]
    rag_context: str                      # 当前检索上下文（动态）
    current_query: str                    # 当前用户 query
    budget: dict[str, Any]                # 预算使用报告
    layer_tokens: dict[str, int]          # 各层 token 估算（P5 可观测）


@dataclass
class _Budget:
    total: int
    reserved_output: int
    safety_margin: int
    available: int


class ContextManager:
    """按预算装配上下文的入口类。

    用法：
        cm = ContextManager(settings.llm_context_budget)
        view = cm.build_view(
            node="query_rag",
            system_prompt=SYSTEM_PROMPT,
            course_context=course_context,
            user_profile=profile,
            conversation=conversation,
            rolling_summary=rolling_summary,
            current_query=query,
            rag_context=context,
        )
        messages = [
            {"role": "system", "content": view.system_prefix},
            ...
        ]
    """

    # 默认预算配置（可在外部配置覆盖）
    DEFAULT_BUDGET = {
        "total_tokens": 64_000,
        "reserved_output": 4_096,
        "safety_margin": 1_024,
        "max_recent_turns": 6,
        "rolling_summary_max_tokens": 1_500,
        "user_profile_max_tokens": 400,
        "rag_default_max_tokens": 8_000,
    }

    # 不同节点的上下文策略：控制是否使用某层记忆以及 RAG 预算
    # 轮次数量由全局 max_recent_turns 控制，调用方可通过 budget_config 覆盖
    NODE_CONFIGS: dict[str, dict[str, Any]] = {
        "classify": {
            "use_rag": False,
            "use_rolling_summary": True,
            "use_user_profile": False,
            "rag_max_tokens": 0,
        },
        "query_rag": {
            "use_rag": True,
            "use_rolling_summary": True,
            "use_user_profile": True,
            "rag_max_tokens": 8_000,
        },
        "generate_quiz": {
            "use_rag": True,
            "use_rolling_summary": True,
            "use_user_profile": True,
            "rag_max_tokens": 6_000,
        },
        "evaluate_quiz": {
            "use_rag": True,
            "use_rolling_summary": True,
            "use_user_profile": False,
            "rag_max_tokens": 4_000,
        },
        "diagnose": {
            "use_rag": False,
            "use_rolling_summary": True,
            "use_user_profile": True,
            "rag_max_tokens": 0,
        },
        "review_plan": {
            "use_rag": False,
            "use_rolling_summary": True,
            "use_user_profile": True,
            "rag_max_tokens": 0,
        },
    }

    def __init__(self, budget_config: dict[str, Any] | None = None):
        cfg = {**self.DEFAULT_BUDGET, **(budget_config or {})}
        self.total = int(cfg["total_tokens"])
        self.reserved_output = int(cfg["reserved_output"])
        self.safety_margin = int(cfg["safety_margin"])
        self.max_recent_turns = int(cfg["max_recent_turns"])
        self.rolling_summary_max = int(cfg["rolling_summary_max_tokens"])
        self.user_profile_max = int(cfg["user_profile_max_tokens"])
        self.rag_default_max = int(cfg["rag_default_max_tokens"])

    def _budget_for(self, node_config: dict[str, Any]) -> _Budget:
        available = self.total - self.reserved_output - self.safety_margin
        return _Budget(
            total=self.total,
            reserved_output=self.reserved_output,
            safety_margin=self.safety_margin,
            available=max(0, available),
        )

    @staticmethod
    def _fmt_course_context(course_context: dict[str, Any] | None) -> str:
        if not course_context:
            return "（未指定课程）"
        chapters = "、".join(course_context.get("chapters", []))
        return (
            f"课程：{course_context.get('name', '未知')}\n"
            f"教材：{course_context.get('textbook', '未知')}\n"
            f"已学章节：{chapters or '暂无'}"
        )

    @staticmethod
    def _fmt_user_profile(profile: dict[str, Any] | None) -> str:
        if not profile:
            return ""
        parts = []
        mastery = profile.get("mastery_level")
        if mastery:
            parts.append("知识点掌握度：")
            for kp, rate in list(mastery.items())[:10]:
                parts.append(f"  - {kp}: {rate}")
        weak_kps = profile.get("weak_kps") or []
        if weak_kps:
            parts.append("薄弱环节：" + "、".join(weak_kps[:10]))
        avg = profile.get("avg_correct_rate")
        if avg is not None:
            parts.append(f"平均正确率：{avg:.0%}")
        return "\n".join(parts)

    @staticmethod
    def _compact_text(text: str, max_tokens: int) -> str:
        """按估算 token 数截断文本，优先保留前缀，避免切断 LaTeX 公式。"""
        if estimate_tokens(text) <= max_tokens:
            return text
        # 二分查找合适长度
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_tokens(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        truncated = text[:lo].rstrip()
        # 避免切断 $...$ 或 $$...$$
        open_single = truncated.count("$") - truncated.replace("$$", "").count("$")
        open_double = truncated.count("$$") % 2
        if open_single % 2 == 1:
            truncated += "$"
        if open_double == 1:
            truncated += "$$"
        return truncated + "\n...（已截断）"

    @staticmethod
    def _trim_turns_to_budget(
        turns: list[dict[str, str]], max_turns: int, max_tokens: int
    ) -> list[dict[str, str]]:
        """先按数量截断，再按预算截断，保留最近的轮次。"""
        recent = turns[-max_turns:] if len(turns) > max_turns else turns
        result = []
        used = 0
        for turn in reversed(recent):
            cost = estimate_tokens(turn.get("content", ""))
            if used + cost > max_tokens and result:
                break
            result.append(turn)
            used += cost
        return list(reversed(result))

    def build_view(
        self,
        *,
        node: str,
        system_prompt: str,
        course_context: dict[str, Any] | None,
        user_profile: dict[str, Any] | None,
        conversation: list[dict[str, Any]] | None,
        rolling_summary: str,
        current_query: str,
        rag_context: str = "",
    ) -> ContextView:
        """为指定节点装配上下文视图，严格遵循预算。

        装配顺序（缓存友好）：system + course_context + user_profile
        → rolling_summary → recent_turns → rag_context → current_query。
        """
        node_cfg = self.NODE_CONFIGS.get(node, self.NODE_CONFIGS["query_rag"])
        budget = self._budget_for(node_cfg)

        layer_tokens: dict[str, int] = {}

        # 1. 固定前缀（最优先，可被 prompt caching 命中）
        course_text = self._fmt_course_context(course_context)
        system_prefix = system_prompt.format(
            course_context=course_text,
            sources="{sources}",  # 占位，后面由调用方替换，避免这里依赖 RAG 上下文
        )

        fixed_parts = [system_prefix]
        if node_cfg.get("use_user_profile", True):
            profile_text = self._compact_text(
                self._fmt_user_profile(user_profile), self.user_profile_max
            )
            if profile_text:
                fixed_parts.append(profile_text)

        fixed_text = "\n\n".join(fixed_parts)
        used = estimate_tokens(fixed_text)
        layer_tokens["system_prefix"] = used

        # 2. L2 滚动摘要
        summary_text = ""
        if node_cfg.get("use_rolling_summary", True) and rolling_summary:
            remaining = budget.available - used
            summary_max = min(self.rolling_summary_max, max(0, remaining // 4))
            summary_text = self._compact_text(rolling_summary, summary_max)
            layer_tokens["rolling_summary"] = estimate_tokens(summary_text)
            used += layer_tokens["rolling_summary"]
        else:
            layer_tokens["rolling_summary"] = 0

        # 3. L1 近期轮次
        recent_turns: list[dict[str, str]] = []
        if conversation:
            remaining = budget.available - used
            # 留给轮次 + query + RAG 的预算
            recent_turns = self._trim_turns_to_budget(
                [
                    {"role": str(t.get("role", "user")), "content": str(t.get("content", ""))}
                    for t in conversation
                    if isinstance(t, dict) and "role" in t and "content" in t
                ],
                max_turns=self.max_recent_turns,
                max_tokens=max(0, remaining // 3),
            )
            layer_tokens["recent_turns"] = sum(estimate_tokens(t["content"]) for t in recent_turns)
            used += layer_tokens["recent_turns"]
        else:
            layer_tokens["recent_turns"] = 0

        # 4. RAG 上下文（动态，按剩余预算截断）
        rag_max = node_cfg.get("rag_max_tokens", self.rag_default_max)
        if node_cfg.get("use_rag", True) and rag_context and rag_max > 0:
            remaining = budget.available - used
            rag_max = min(rag_max, max(0, remaining - estimate_tokens(current_query) - 200))
            rag_context = self._compact_text(rag_context, rag_max)
            layer_tokens["rag_context"] = estimate_tokens(rag_context)
            used += layer_tokens["rag_context"]
        else:
            rag_context = ""
            layer_tokens["rag_context"] = 0

        # 5. 当前 query（必须保留，但要截断异常输入）
        current_query = self._compact_text(current_query, 2_000)
        layer_tokens["current_query"] = estimate_tokens(current_query)
        used += layer_tokens["current_query"]

        view = ContextView(
            system_prefix=fixed_text,
            course_context=course_text,
            user_profile=self._fmt_user_profile(user_profile),
            rolling_summary=summary_text,
            recent_turns=recent_turns,
            rag_context=rag_context,
            current_query=current_query,
            budget={
                "total": budget.total,
                "available": budget.available,
                "used": used,
                "node": node,
                "recent_turns_count": len(recent_turns),
                "rag_truncated": rag_context.endswith("...（已截断）"),
                "cache_friendly": True,  # system 与课程上下文稳定在最前
            },
            layer_tokens=layer_tokens,
        )

        if used > budget.available:
            logger.warning(
                "ContextManager: %s 视图 token 估算 %d 超过可用预算 %d",
                node, used, budget.available
            )

        return view

    def needs_compaction(
        self,
        conversation: list[dict[str, Any]] | None,
        rolling_summary: str,
        threshold_ratio: float = 0.75,
    ) -> bool:
        """判断 L1+L2 是否超过预算阈值，触发滚动压缩。"""
        if not conversation:
            return False
        history_text = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in conversation
        )
        history_tokens = estimate_tokens(history_text) + estimate_tokens(rolling_summary)
        return history_tokens > self.total * threshold_ratio
