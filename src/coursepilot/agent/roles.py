"""确定性 Role 层 —— 编排层访问角色能力的唯一入口。

把「用户角色」从散落在节点内部的 DB 查询 / 硬编码，收敛为
「图内确定性事实 + 复用 governance/rbac.py 的单一真源」。

本层纯同步、零 LLM、零 DB：role 由图外鉴权层（api/agent.py）通过
``current_user.role`` 传入，本层只做「角色字符串 → 层级 / 权限」的确定性映射。

当前为空转阶段：只解析、只供单测锁定行为，不据此改变任何路由或控制流；
角色驱动的权限边界强制属于后续独立提交。
"""
from coursepilot.governance.rbac import RoleHierarchy, has_permission

# 合法角色集合（与 models/user.py 的 User.role 取值、rbac.RoleHierarchy 同源）
VALID_ROLES: tuple[str, ...] = ("student", "teacher", "super")


def resolve_role(user_role: str) -> RoleHierarchy:
    """确定性解析角色层级。

    Args:
        user_role: 角色名（student / teacher / super）。

    Returns:
        RoleHierarchy 层级。未知角色兜底到 student（fail-closed），
        与 rbac.get_role_hierarchy 的兜底语义一致。
    """
    try:
        return RoleHierarchy[user_role]
    except KeyError:
        return RoleHierarchy.student


def role_can(user_role: str, permission: str) -> bool:
    """图内权限判定（复用 rbac.has_permission 的权限矩阵真源）。

    Args:
        user_role: 角色名（student / teacher / super）。
        permission: 权限 key（如 ``"course:write"``）。
    """
    return has_permission(user_role, permission)
