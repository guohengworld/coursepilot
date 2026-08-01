"""CoursePilot RAG 子包。

Windows DLL 加载顺序修复
========================

本 __init__ 在任何 ``coursepilot.rag.*`` 子模块之前执行，是注入预加载的
安全位置。

背景：grpc 的 C 扩展 ``cygrpc`` 在 Windows 上若于 torch/faiss/milvus_lite
（→ Intel libiomp）与 FlagEmbedding/transformers/sklearn（→ MSVC vcomp140）
之后才首次加载，其 DllMain 会因 OpenMP 运行时冲突报
``[WinError 1114] 动态链接库(DLL)初始化例程失败``。

修复：在加载任何 RAG 子模块（它们会间接导入上述库）之前，先 ``import grpc``
把 cygrpc 加载并缓存进 ``sys.modules``，此后子模块内对 grpc/pymilvus 的惰性
导入均命中缓存，不再触发 DllMain。

该预加载仅作用于导入 RAG 子包的进程（Gateway / 单体 Server / 脚本）；
stdio 桥接器（``coursepilot.mcp.cli``）不导入 RAG，因此不受影响。
"""
import os as _os

if _os.name == "nt":
    try:
        import grpc  # noqa: F401  预加载 cygrpc 并缓存进 sys.modules
    except Exception:
        # 预加载失败不应阻断子包导入；最坏情况退回原始冲突报错。
        pass
