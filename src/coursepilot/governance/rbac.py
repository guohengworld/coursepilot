"""基于角色的权限控制 (RBAC)

权限矩阵定义每个角色对每种资源类型的操作权限
当前角色层级: student(0) < teacher(1) < super(2)
"""
from enum import IntEnum

class RoleHierarchy(IntEnum):
    """角色层级：数字越大权限越高"""
    student = 0
    teacher = 1
    super = 2

# 权限矩阵：resource → 允许的操作角色
# key: "resource:action"，value: 最低角色层级
PERMISSION_MATRIX: dict[str, RoleHierarchy] = {
    # 课程
    "course:read": RoleHierarchy.student,
    "course:write": RoleHierarchy.teacher,
    "course:delete": RoleHierarchy.super,
    # 资料
    "document:upload": RoleHierarchy.teacher,
    "document:delete": RoleHierarchy.teacher,
    # Agent
    "agent:chat": RoleHierarchy.student,
    "agent:session:list": RoleHierarchy.student,        # 只能看自己的
    "agent:session:list_all": RoleHierarchy.teacher,    # 可以看课程的所有
    # 学情
    "diagnosis:view": RoleHierarchy.student,
    "practice:create": RoleHierarchy.student,
    # 教师发布任务（⑤）：assign=生成草稿/选学生，publish=发布；
    # view 学生可查发布给自己的任务（teacher 层级更高，天然覆盖草稿查看）
    "task:assign": RoleHierarchy.teacher,
    "task:publish": RoleHierarchy.teacher,
    "task:view": RoleHierarchy.student,
    # 审计
    "audit:view": RoleHierarchy.super,
    # 用户管理
    "user:list": RoleHierarchy.teacher,
    "user:modify_role": RoleHierarchy.super
}

def get_role_hierarchy(role: str) -> int:
    """将角色名转为层级数值"""
    try:
        return RoleHierarchy[role].value
    except KeyError:
        return RoleHierarchy.student.value

def has_permission(user_role: str, permission: str) -> bool:
    """检查是否拥有指定权限

    Args:
        user_role: 用户角色名 (student/teacher/super)
        permission: 权限 key (如 "course:write")

    Return:
        True 如果用户角色层级 >= 权限要求的最低层级
    """
    required = PERMISSION_MATRIX.get(permission)
    if required is None:
        return False
    return get_role_hierarchy(user_role) >= required.value

def filter_own_resources(user_role: str, permission: str) -> bool:
    """判断该权限是否需要只能访问自己的资源

    例如 student 只能查看自己的 agent session,
    而 teacher 可以查看课程下所有学生的
    """
    # student 级别的"列表类"权限通常被限制为自己
    if user_role == "student" and permission.endswith(":list"):
        return True
    return False
