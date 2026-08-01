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
