"""结构化查询日志 — JSON 格式记录每个查询的完整链路耗时和结果。

用法:
    from coursepilot.rag.logger import QueryLogger
    qlogger = QueryLogger()
    qlogger.log_query(trace_id, user_id, course_id, query_raw, query_rewritten,
                      stages, top_rerank_scores, source_kp_paths, citation_count, answer_length)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("coursepilot.rag.query")


class QueryLogger:
    """结构化查询日志记录器。

    每条查询输出一条 JSON 日志，同时返回 trace_id 用于链路追踪。
    """

    def __init__(self):
        self._logger = logging.getLogger("coursepilot.rag.query")

    def start_trace(self) -> tuple[str, float]:
        """开始一次查询追踪。

        返回: (trace_id, start_time)
        """
        trace_id = str(uuid.uuid4())[:8]
        return trace_id, time.time()

    def log_query(
        self,
        trace_id: str,
        user_id: str,
        course_id: str,
        query_raw: str,
        query_rewritten: str,
        stages: dict[str, float],       # {"rewrite_ms": 480, "encode_ms": 120, ...}
        top_rerank_scores: list[float],
        source_kp_paths: list[str],
        citation_count: int,
        answer_length: int,
    ) -> None:
        """记录一条完整查询日志。"""
        total_ms = sum(stages.values())

        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "course_id": course_id,
            "query_raw": query_raw,
            "query_rewritten": query_rewritten,
            "stages": stages,
            "total_ms": round(total_ms, 1),
            "top_rerank_scores": [round(s, 4) for s in top_rerank_scores],
            "source_kp_paths": source_kp_paths,
            "citation_count": citation_count,
            "answer_length": answer_length,
        }

        self._logger.info(json.dumps(log_entry, ensure_ascii=False))
