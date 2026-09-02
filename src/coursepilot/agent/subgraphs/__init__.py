"""子图构建包：每个技能粒度子图一个 build_xxx_subgraph() 工厂函数。

这些函数返回编译后的 CompiledStateGraph（不传 checkpointer，
由父图 build_agent_graph 统一注入 AsyncPostgresSaver）。
"""
from coursepilot.agent.subgraphs.diagnose import build_diagnose_subgraph
from coursepilot.agent.subgraphs.practice import build_practice_subgraph
from coursepilot.agent.subgraphs.question import build_question_subgraph
from coursepilot.agent.subgraphs.review import build_review_subgraph

__all__ = [
    "build_diagnose_subgraph",
    "build_question_subgraph",
    "build_practice_subgraph",
    "build_review_subgraph",
]
