"""知识点树操作：CRUD + 递归 CTE 查询。

依赖 PostgreSQL 的 WITH RECURSIVE 实现子树查询和路径查询。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from coursepilot.models.knowledge_point import KnowledgePoint


@dataclass
class KPTreeNode:
    """内存中的树节点，供 API 返回用。"""
    id: str
    title: str
    kp_path: str
    level: int
    children: list[KPTreeNode]


class KPTree:
    """知识点树操作工具。

    所有方法需要传入 AsyncSession，由调用方管理事务。
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── 批量创建 ─────────────────────────────────────────

    async def create_from_nodes(
        self, nodes: list[dict], course_id: str
    ) -> list[str]:
        """从 SyllabusParser 展平后的节点列表批量插入 knowledge_points。

        节点按 parent_id 为 None 的先插，有 parent_id 的后续更新。
        返回插入后的所有 kp_id 列表（与输入顺序对应）。
        """
        ids: list[str] = []
        # 第一遍：插入所有节点，先不填 parent_id
        title_to_id: dict[str, str] = {}
        for node in nodes:
            kp = KnowledgePoint(
                course_id=uuid.UUID(course_id),
                kp_path=node.get("kp_path", ""),
                title=node.get("title", ""),
                summary=node.get("summary", ""),
                difficulty=node.get("difficulty", 1),
                sort_order=node.get("sort_order", 0),
            )
            self.session.add(kp)
            await self.session.flush()
            kid = str(kp.id)
            ids.append(kid)
            title_to_id[node["title"]] = kid
            node["id"] = kid

        # 第二遍：回填 parent_id（用 title 匹配父节点）
        for node in nodes:
            parent_title = node.get("parent_title")
            if parent_title and parent_title in title_to_id:
                kp = await self.session.get(KnowledgePoint, uuid.UUID(node["id"]))
                if kp:
                    kp.parent_id = uuid.UUID(title_to_id[parent_title])

        await self.session.flush()
        return ids

    # ── 递归 CTE 查询 ────────────────────────────────────

    async def get_subtree(self, root_id: str) -> KPTreeNode | None:
        """获取以 root_id 为根的整棵子树。

        SQL 等效:
            WITH RECURSIVE subtree AS (
                SELECT * FROM knowledge_points WHERE id = :root_id
                UNION ALL
                SELECT kp.* FROM knowledge_points kp
                JOIN subtree s ON kp.parent_id = s.id
            )
            SELECT * FROM subtree;
        """
        cte_sql = text("""
            WITH RECURSIVE subtree AS (
                SELECT id, parent_id, kp_path, title, difficulty, sort_order
                FROM knowledge_points
                WHERE id = :root_id
                UNION ALL
                SELECT kp.id, kp.parent_id, kp.kp_path, kp.title,
                       kp.difficulty, kp.sort_order
                FROM knowledge_points kp
                JOIN subtree s ON kp.parent_id = s.id
            )
            SELECT * FROM subtree ORDER BY sort_order
        """)
        result = await self.session.execute(cte_sql, {"root_id": root_id})
        rows = result.fetchall()
        return self._build_tree(rows, root_id)

    async def get_path(self, leaf_id: str) -> list[dict]:
        """获取从根到 leaf_id 的完整路径（自底向上递归）。

        SQL 等效:
            WITH RECURSIVE path AS (
                SELECT *, 1 AS depth FROM knowledge_points WHERE id = :leaf_id
                UNION ALL
                SELECT kp.*, p.depth + 1
                FROM knowledge_points kp
                JOIN path p ON kp.id = p.parent_id
            )
            SELECT * FROM path ORDER BY depth DESC;
        """
        cte_sql = text("""
            WITH RECURSIVE path AS (
                SELECT id, parent_id, kp_path, title, difficulty, 1 AS depth
                FROM knowledge_points
                WHERE id = :leaf_id
                UNION ALL
                SELECT kp.id, kp.parent_id, kp.kp_path, kp.title,
                       kp.difficulty, p.depth + 1
                FROM knowledge_points kp
                JOIN path p ON kp.id = p.parent_id
            )
            SELECT id, parent_id, kp_path, title, difficulty, depth
            FROM path
            ORDER BY depth DESC
        """)
        result = await self.session.execute(cte_sql, {"leaf_id": leaf_id})
        rows = result.fetchall()
        return [
            {
                "id": str(r.id), "parent_id": str(r.parent_id) if r.parent_id else None,
                "kp_path": r.kp_path, "title": r.title, "difficulty": r.difficulty,
                "depth": r.depth,
            }
            for r in rows
        ]

    # ── 统计 ─────────────────────────────────────────────

    async def count_by_course(self, course_id: str) -> int:
        """某课程下的知识点总数。"""
        from sqlalchemy import select, func
        result = await self.session.execute(
            select(func.count()).where(KnowledgePoint.course_id == uuid.UUID(course_id))
        )
        return result.scalar() or 0

    # ── 内部 ─────────────────────────────────────────────

    def _build_tree(self, rows, root_id: str) -> KPTreeNode | None:
        """将查询结果构建为 KPTreeNode 嵌套结构。"""
        row_map: dict[str, KPTreeNode] = {}
        for r in rows:
            rid = str(r.id)
            row_map[rid] = KPTreeNode(
                id=rid, title=r.title, kp_path=r.kp_path,
                level=0, children=[],
            )
        # 计算 level（通过 parent_id 链）
        for r in rows:
            rid = str(r.id)
            depth = 0
            cur = r.parent_id
            while cur and str(cur) in row_map:
                depth += 1
                # 简化：通过行数据找父
                break  # MVP 阶段保持简单
            row_map[rid].level = depth

        # 嵌套
        root_node = row_map.get(root_id)
        for r in rows:
            rid = str(r.id)
            pid = str(r.parent_id) if r.parent_id else None
            if pid and pid in row_map:
                row_map[pid].children.append(row_map[rid])

        return root_node
