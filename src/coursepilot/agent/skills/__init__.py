"""Agent Skill 集合。

每个 skill 封装一个独立的原子能力，供 LangGraph 节点调用。
Skill 之间无直接依赖，全部通过 AgentState 交换数据。
"""
