"""CoursePilot 包入口。

必须在导入任何子模块前，先修复 Windows WMI 死锁问题（见下方）。
"""
import collections as _collections
import platform as _platform

__version__ = "0.1.0"

# ── Windows WMI 死锁修复 ──────────────────────────────────
# asyncpg.compat / greenlet / sqlalchemy 等库在导入时会调用
# platform.uname()，该函数在 Windows 上通过 WMI 查询系统信息。
# 在本机环境下 WMI 调用会死锁（无论单次还是并发），导致
# import sqlalchemy 永久挂起。
#
# 修复：用硬编码的 uname 结果替换 platform.uname，彻底避免 WMI 调用。
# 这些值仅供库做平台判断（如 asyncpg 判断 system == "Windows"），
# 硬编码完全够用。
#
# 注意：本补丁必须在 import sqlalchemy 之前执行，因此放在 __init__.py
# 顶部——任何 import coursepilot.xxx 都会先执行此处。
_uname_cached = _collections.namedtuple(
    "uname_result", "system node release version machine"
)("Windows", "localhost", "10", "10.0.19045", "AMD64")


def _safe_uname():
    """返回硬编码的 uname，永不触发 WMI 调用。"""
    return _uname_cached


_platform.uname = _safe_uname

# ── Windows MKL 重复加载修复 ──────────────────────────────
# faiss（milvus_lite 依赖）与 torch 都绑定 Intel MKL（libiomp），
# 在同一进程同时加载第二个 libiomp 时会报错。允许 MKL 重复加载以绕过
# 此冲突（Retriever 实例化时 faiss 与 torch 同时在场）。
# 仅 Windows 需要；Linux 下该环境变量无副作用。setdefault 不覆盖用户已设值。
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Windows OpenMP 运行时冲突修复（vcomp140 vs libiomp）──────
# torch/faiss 加载 Intel libiomp 后，sklearn 的 _distributor_init 再加载
# MSVC 的 vcomp140.dll 时，其 DllMain 检测到已有另一套 OpenMP 运行时在场，
# 拒绝初始化并抛出 "[WinError 1114] 动态链接库(DLL)初始化例程失败"。
# KMP_DUPLICATE_LIB_OK 只安抚 libiomp，对 vcomp 无效。
#
# 修复：在任何子模块（→ torch/faiss → libiomp）导入之前，先预加载
# sklearn 自带的 vcomp140.dll，使其成为进程内首个 OpenMP 运行时；
# 此后 libiomp 仍可正常加载（配合上面的 KMP_DUPLICATE_LIB_OK），
# 而 sklearn 后续 WinDLL(vcomp140) 命中已加载模块，不再触发 DllMain。
# 必须放在 __init__.py 顶部——任何 import coursepilot.xxx 都会先执行此处。
# find_spec 不执行 sklearn/__init__.py，因此不会触发 _distributor_init。
if _os.name == "nt":
    try:
        import importlib.util as _ilu
        import ctypes as _ctypes
        _sk_spec = _ilu.find_spec("sklearn")
        _sk_root = (
            _sk_spec.submodule_search_locations[0]
            if _sk_spec is not None and _sk_spec.submodule_search_locations
            else None
        )
        if _sk_root is not None:
            _vcomp_path = _os.path.join(_sk_root, ".libs", "vcomp140.dll")
            if _os.path.exists(_vcomp_path):
                _ctypes.WinDLL(_vcomp_path)
    except Exception:
        # 预加载失败不应阻断包导入；最坏情况退回原始冲突报错。
        pass
