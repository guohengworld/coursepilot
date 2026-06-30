"""Agent 模块入口 —— 导出编译后的 LangGraph 应用

用法：
    from coursepilot.agent import graph_app
    result = await graph_app.ainvoke(state, config)
"""
from coursepilot.agent.graph import build_agent_graph

graph_app = build_agent_graph()
