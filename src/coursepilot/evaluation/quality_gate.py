"""RAG 质量门禁。

用法：
    from coursepilot.evaluation.quality_gate import QualityGate
    gate = QualityGate()
    passed, failures = gate.check(report)
"""

from __future__ import annotations

from coursepilot.evaluation.metrics_config import THRESHOLDS


class QualityGate:
    """基于 RAGAS 报告的质量门禁。"""

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or THRESHOLDS

    def check(self, report) -> tuple[bool, list[str]]:
        """检查报告是否通过门禁。

        Args:
            report: EvalReport 实例或包含 averages 字段的字典。

        Returns:
            (是否通过, 失败项列表)
        """
        if hasattr(report, "averages"):
            averages = report.averages
        elif isinstance(report, dict):
            averages = report.get("averages", {})
        else:
            averages = {}
        failures = []

        for metric, threshold in self.thresholds.items():
            actual = averages.get(metric, 0.0)
            if actual < threshold:
                failures.append(f"{metric}: {actual:.3f} < {threshold}")

        return not failures, failures

    def format_result(self, report) -> str:
        """格式化门禁结果。"""
        passed, failures = self.check(report)
        if passed:
            return "[GATE PASS] RAG quality gate PASSED"
        lines = ["[GATE FAIL] RAG quality gate FAILED:"]
        for f in failures:
            lines.append(f"   - {f}")
        return "\n".join(lines)
