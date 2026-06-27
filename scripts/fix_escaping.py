"""修复 eval_questions_candidate.json 中双重转义的 LaTeX 反斜杠"""
import json
from pathlib import Path

CANDIDATE = Path("eval/questions/eval_questions_candidate.json")

data = json.loads(CANDIDATE.read_text(encoding="utf-8"))

# 要修复的 LaTeX 命令列表
LATEX_COMMANDS = [
    "sum", "int", "frac", "lim", "infty", "mathbf", "cdot", "cdots",
    "Delta", "partial", "overrightarrow", "cos", "sin", "theta",
    "lambda", "infty", "times", "pi", "int", "iint", "iiint", "oint",
    "nabla", "Phi", "Omega", "sigma", "rho", "mu", "alpha", "beta",
    "gamma", "delta", "epsilon", "varphi", "subseteq", "mathbb",
    "longrightarrow", "Longrightarrow", "longmapsto",
]
# 特殊转义序列
SPECIAL_ESCAPES = ["{", "}", "(", ")", "[", "]"]

for q in data:
    for key in ("question", "answer", "kp_path"):
        if key not in q:
            continue
        s = q[key]
        # 修复 LaTeX 命令: \\sum → \sum
        for cmd in LATEX_COMMANDS:
            s = s.replace("\\\\" + cmd, "\\" + cmd)
        # 修复转义括号: \\{ → \{
        for ch in SPECIAL_ESCAPES:
            s = s.replace("\\\\" + ch, "\\" + ch)
        q[key] = s

CANDIDATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# 验证
data2 = json.loads(CANDIDATE.read_text(encoding="utf-8"))
print(f"共 {len(data2)} 道题")
for i, q in enumerate(data2):
    # 检查是否还有多余的 \\\\
    text = q["question"] + q["answer"]
    if "\\\\\\\\" in text:
        print(f"  Q{i+1}: STILL HAS QUAD BACKSLASH!")
    elif "\\\\sum" in text or "\\\\int" in text or "\\\\frac" in text:
        print(f"  Q{i+1}: double backslash (batch1 style, may be intentional)")
    else:
        print(f"  Q{i+1}: OK")
