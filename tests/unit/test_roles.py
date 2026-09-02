"""确定性 Role 层 · 角色解析单测。

锁定 agent/roles.py 的确定性解析行为：角色字符串 → RoleHierarchy 层级，
以及权限矩阵关键项的快照。这些是纯同步、零 LLM 的函数，测试零成本、零依赖。

对应重构方案「引入确定性 Role 层（空转，不接管路由）」，只锁解析本身，
不涉及任何图内路由/控制流。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from coursepilot.agent.roles import VALID_ROLES, resolve_role, role_can
from coursepilot.governance.rbac import RoleHierarchy


class TestResolveRole:
    """resolve_role：角色字符串 → 层级的确定性映射"""

    def test_valid_roles_map_to_hierarchy(self):
        assert resolve_role("student") == RoleHierarchy.student
        assert resolve_role("teacher") == RoleHierarchy.teacher
        assert resolve_role("super") == RoleHierarchy.super

    def test_unknown_role_falls_back_to_student(self):
        """未知角色 fail-closed 兜底到 student（最低层级）"""
        assert resolve_role("admin") == RoleHierarchy.student
        assert resolve_role("") == RoleHierarchy.student
        assert resolve_role("STUDENT") == RoleHierarchy.student  # 大小写敏感，未知

    def test_hierarchy_is_ordered(self):
        """层级数值满足 student < teacher < super（rbac 的信任前提）"""
        assert RoleHierarchy.student < RoleHierarchy.teacher < RoleHierarchy.super

    def test_valid_roles_constant(self):
        """VALID_ROLES 与 RoleHierarchy 枚举成员严格一致"""
        assert set(VALID_ROLES) == {r.name for r in RoleHierarchy}


class TestRoleCan:
    """role_can：复用 rbac 权限矩阵的关键项快照"""

    def test_student_baseline_permissions(self):
        """student 拥有只读/自助类权限"""
        assert role_can("student", "agent:chat") is True
        assert role_can("student", "agent:session:list") is True
        assert role_can("student", "practice:create") is True
        assert role_can("student", "diagnosis:view") is True

    def test_student_denied_privileged(self):
        """student 被拒的越权项"""
        assert role_can("student", "course:write") is False
        assert role_can("student", "course:delete") is False
        assert role_can("student", "agent:session:list_all") is False
        assert role_can("student", "audit:view") is False

    def test_teacher_elevated_permissions(self):
        """teacher 拥有写权限但不含 super 专属"""
        assert role_can("teacher", "course:write") is True
        assert role_can("teacher", "document:upload") is True
        assert role_can("teacher", "agent:session:list_all") is True
        assert role_can("teacher", "course:delete") is False
        assert role_can("teacher", "audit:view") is False

    def test_super_full_access(self):
        """super 拥有矩阵中全部权限"""
        assert role_can("super", "course:delete") is True
        assert role_can("super", "audit:view") is True
        assert role_can("super", "user:modify_role") is True

    def test_unknown_permission_denied(self):
        """未在矩阵中声明的权限一律拒绝（fail-closed）"""
        assert role_can("super", "does:not:exist") is False
