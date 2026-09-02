"""诊断子图：学情诊断（最独立、只读，首个抽离的子图）。

拓扑：START → diagnose → END

复用现有 diagnose_node（签名 (state: dict) -> dict 不变），通过
DiagnoseInput / DiagnoseOutput 裁剪与父图的边界。本子图无私有中间态。

注意：compile() 不传 checkpointer——checkpointer 由父图统一注入，
父子图共享同一份 checkpoint。
"""
from langgraph.graph import END, START, StateGraph

from coursepilot.agent.nodes import diagnose_node
from coursepilot.agent.sub_state import DiagnoseInput, DiagnoseOutput, DiagnoseState


def build_diagnose_subgraph():
    """构建诊断子图（CompiledStateGraph）。"""
    builder = StateGraph(
        DiagnoseState,
        input_schema=DiagnoseInput,
        output_schema=DiagnoseOutput,
    )
    builder.add_node("diagnose", diagnose_node)
    builder.add_edge(START, "diagnose")
    builder.add_edge("diagnose", END)
    return builder.compile()
