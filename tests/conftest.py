"""pytest 全局夹具与进程级初始化。

conftest 会在收集阶段先于所有测试模块导入，是控制原生库加载顺序的可靠位置。
"""
import pymilvus  # noqa: F401

# 原生库加载顺序约束（与 src/coursepilot/main.py 顶部约束同源，2026-09-04 本机验证）：
# pymilvus（连带加载 pyarrow/arrow.dll）必须在 torch 之前导入。测试进程中
# 若 torch 先加载（FlagEmbedding 相关测试），后续任何懒加载 pymilvus 的路径
# 都会触发 arrow.dll 0xc0000005 访问冲突，进程带着崩溃码退出且吞掉测试输出。
