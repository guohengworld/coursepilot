"""本地文件存储服务

所有上传文件存到 data/uploads/<course_id>/ 下，文件以 UUID 命名
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from coursepilot.config import _PROJECT_ROOT

class FileStore:
    """本地文件系统存储

    store = FileStore()
    file_info = store.save(uploaded_bytes, course_id, original_filename)
    store.delete(file_path)
    """

    def __init__(self, base_dir: str | None = None):
        self.base = Path(base_dir or _PROJECT_ROOT / "data" / "uploads")

    def save(self, content: bytes, course_id: str, filename: str) -> dict:
        """保存文件到本地，返回文件信息 dict

        :return: {"file_path": str, "file_size": int, "stored_name": str}
        """
        course_dir = self.base / course_id
        course_dir.mkdir(parents=True, exist_ok=True)

        # 用 UUID + 原扩展名避免冲突
        ext = Path(filename).suffix.lower()
        stored_name = f"{uuid.uuid4().hex}{ext}"
        file_path = course_dir / stored_name

        file_path.write_bytes(content)

        return {
            "file_path": str(file_path),
            "file_size": len(content),
            "stored_name": stored_name,
        }

    def delete(self, file_path: str) -> bool:
        """删除文件。成功返回 True，文件不存在返回 False"""
        path = Path(file_path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def get_path(self, course_id: str, stored_name: str) -> Path:
        """获取文件在文件系统中的完整路径"""
        return self.base / course_id / stored_name

    def delete_course_files(self, course_id: str) -> int:
        """删除某课程下的所有文件（级联删除时使用）。返回删除数量"""
        course_dir = self.base / course_id
        if not course_dir.exists():
            return 0
        count = len(list(course_dir.iterdir()))
        shutil.rmtree(course_dir)
        return count

