"""
LLM 辅助生成 RAG 评估用黄金数据集。

用法：
    PYTHONPATH=src .venv/Scripts/python -m scripts.generate_eval_questions

流程：
    1. 读取 tests/fixtures/exported_units.json（由导出步骤生成）
    2. 按章节分组，每章选取代表性 unit 摘要发送给 DeepSeek
    3. DeepSeek 生成问题 + 标准答案 + 相关 UUID 列表
    4. 输出到 tests/fixtures/eval_questions_candidate.json
    5. 人工校验后改名为 eval_questions.json
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import openai

from coursepilot.config import settings


# ══════════════════════════════════════════════════════════════
# 素材处理：从导出的全量 unit 中为每章提取精简摘要
# ══════════════════════════════════════════════════════════════

EXPORT_PATH = Path("eval/questions/exported_units.json")
OUTPUT_PATH = Path("eval/questions/eval_questions_candidate.json")

# 每章最多选取的 unit 数量（控制 token 消耗）
MAX_UNITS_PER_CHAPTER = 12

# 排除的非教学内容（按路径前缀匹配）
EXCLUDE_PREFIXES = [
    "微积分/习题参考答案",
    "微积分/大学数学",
]


def _get_chapter_key(kp_path: str) -> str | None:
    """从 kp_path 提取章节键，如 '微积分/第7章 向量代数与空间解析几何'。"""
    parts = kp_path.split("/")
    if len(parts) < 2:
        return None
    # 返回前两级作为章节标识
    return "/".join(parts[:2])


def load_and_group() -> list[dict]:
    """加载导出文件，按章节分组，每章选取代表性 unit。"""
    with open(EXPORT_PATH, encoding="utf-8") as f:
        all_kps = json.load(f)

    chapters: dict[str, list[dict]] = {}
    for kp in all_kps:
        ch_key = _get_chapter_key(kp["kp_path"])
        if ch_key is None:
            continue
        # 排除非教学内容
        if any(ch_key.startswith(ex) for ex in EXCLUDE_PREFIXES):
            continue

        if ch_key not in chapters:
            chapters[ch_key] = []
        chapters[ch_key].append(kp)

    # 每章选取代表性 unit（前 3 + 中间 2 + 最后 2，尽量覆盖章节全貌）
    result = []
    for ch_name, kp_list in sorted(chapters.items()):
        selected_units = []
        for kp in kp_list:
            units = kp["units"]
            if not units:
                continue
            n = len(units)
            if n <= 7:
                picked = units
            else:
                indices = [0, 1, 2] + [n // 2, n // 2 + 1] + [n - 2, n - 1]
                picked = [units[i] for i in indices if i < n]
            for u in picked:
                selected_units.append({
                    **u,
                    "kp_path": kp["kp_path"],
                })

        # 如果某章 unit 太多，截断
        if len(selected_units) > MAX_UNITS_PER_CHAPTER:
            # 均匀采样
            step = len(selected_units) // MAX_UNITS_PER_CHAPTER
            selected_units = selected_units[::step][:MAX_UNITS_PER_CHAPTER]

        result.append({
            "chapter": ch_name.split("/")[-1],  # e.g. "第7章 向量代数与空间解析几何"
            "units": selected_units,
        })

    return result


def build_unit_reference(chapter_data: dict) -> str:
    """为某章构建 unit 参考文本（含 UUID + 摘要 + 内容片段）。"""
    lines = [f"## {chapter_data['chapter']}"]
    for u in chapter_data["units"]:
        uuid = u["uuid"]
        kp_path = u["kp_path"]
        summary = (u["summary"] or "")[:120]
        content_preview = u["content"][:400].replace("\n", " ")
        lines.append(
            f'<unit uuid="{uuid}" path="{kp_path}" page="{u["page_ref"]}">\n'
            f"  summary: {summary}\n"
            f"  content: {content_preview}...\n"
            f"</unit>"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Prompt 设计
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是大学数学助教，负责为 RAG 检索系统构建评估用问答对。

你会收到教材某一章的若干知识片段，每个片段有唯一的 uuid。

请为这一章生成 2 道问答题，覆盖不同的知识点和问题类型。
每道题必须标注引用哪些 unit 的 uuid 作为标准答案来源。

问题类型分布：
- concept: 概念解释（"什么是..."、"说明...的含义"）
- calculation: 计算题（"求..."、"计算..."）
- theorem: 定理/公式推导（"证明..."、"推导..."）
- comparison: 辨析比较（"...和...有什么区别"）
- application: 应用题（"用...解决..."）

输出严格的 JSON 数组，不要加任何解释文字：

[
  {
    "question": "问题文字",
    "answer": "标准答案（2-4句话，基于教材内容）",
    "ground_truth_contexts": ["uuid-1", "uuid-2"],
    "question_type": "concept|calculation|theorem|comparison|application",
    "kp_path": "知识点路径"
  }
]

要求：
1. 问题必须能用提供的教材片段回答，不要编造教材中没有的内容
2. 标准答案必须基于教材原意，不要引入外部知识
3. ground_truth_contexts 必须是提供的 unit uuid 中真实存在的
4. 覆盖至少 2 种不同的问题类型
5. 问题用中文，涉及数学公式用 LaTeX $...$ 或 $$...$$ 表达"""


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════

async def generate_for_chapter(client, chapter_data: dict) -> list[dict]:
    """为单个章节生成评估问答对。"""
    unit_ref = build_unit_reference(chapter_data)
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请为以下教材章节生成评估问答对：\n\n{unit_ref}"},
            ],
            temperature=0.7,
            max_tokens=8192,
        )
    except Exception as e:
        print(f"  [ERROR] API 调用失败: {e}")
        return []

    text = response.choices[0].message.content
    return parse_llm_output(text)


def parse_llm_output(text: str) -> list[dict]:
    """从 LLM 回复中提取 JSON 数组，容忍截断和 LaTeX 转义。"""
    text = text.strip()

    # 预处理：LLM 输出的 LaTeX（如 \{、\sum、\int）中的 \ 未按 JSON 规范转义
    # 找出 \ 后面不是合法 JSON 转义字符的情况，补一个 \
    text = _fix_latex_backslashes(text)

    # 尝试1：直接解析完整 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试2：regex 提取 JSON 数组
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 尝试3：逐个提取完整的 JSON 对象（处理被截断的输出）
    objs = _extract_json_objects(text)
    if objs:
        return objs

    try:
        print(f"  [WARN] LLM output not valid JSON, first 500 chars: {text[:500]}")
    except UnicodeEncodeError:
        print(f"  [WARN] LLM output not valid JSON (encoding suppressed)")
    return []


def _extract_json_objects(text: str) -> list[dict]:
    """逐个提取完整的顶级 JSON 对象，容忍截断。"""
    objs = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    objs.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start = -1

    return objs


def _fix_latex_backslashes(text: str) -> str:
    """修复 LLM 输出的 JSON 中未转义的 LaTeX 反斜杠。

    LLM 输出可能混合多种转义风格：
    - 正确 JSON 转义（如 \\\\sum）→ 保持不变
    - 三反斜杠（过度转义）→ 标准化为双反斜杠
    - 单反斜杠（忘记 JSON 转义）→ 补为双反斜杠
    """
    import re as _re
    # Step 1: collapse 3+ consecutive backslashes to double (fix over-escaping)
    text = _re.sub(r'\\{3,}', r'\\\\', text)
    # Step 2: fix single backslashes not part of valid JSON escapes
    text = _re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    return text


def validate_questions(questions: list[dict], chapter_data: dict) -> tuple[list[dict], list[str]]:
    """校验生成的题目：UUID 是否存在、必填字段是否完整。"""
    valid_uuids = {u["uuid"] for u in chapter_data["units"]}
    required_fields = ["question", "answer", "ground_truth_contexts", "question_type"]

    valid, warnings = [], []
    for i, q in enumerate(questions):
        # 字段完整性
        missing = [f for f in required_fields if f not in q or not q[f]]
        if missing:
            warnings.append(f"Q{i}: 缺少字段 {missing}，跳过")
            continue

        # UUID 校验
        invalid_uuids = [uid for uid in q["ground_truth_contexts"] if uid not in valid_uuids]
        if invalid_uuids:
            warnings.append(f"Q{i}: UUID 不存在 {invalid_uuids}，已移除")
            q["ground_truth_contexts"] = [
                uid for uid in q["ground_truth_contexts"] if uid in valid_uuids
            ]
            if not q["ground_truth_contexts"]:
                warnings.append(f"Q{i}: 无合法 UUID 剩余，跳过")
                continue

        # 类型校验
        valid_types = {"concept", "calculation", "theorem", "comparison", "application"}
        if q["question_type"] not in valid_types:
            q["question_type"] = "concept"

        valid.append(q)

    return valid, warnings


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

async def main():
    # 1. 加载素材
    if not EXPORT_PATH.exists():
        print(f"[ERROR] 素材文件不存在: {EXPORT_PATH}")
        print("请先运行导出步骤生成 exported_units.json")
        sys.exit(1)

    chapters = load_and_group()
    print(f"加载 {len(chapters)} 个章节")

    # 2. 初始化 DeepSeek 客户端
    if not settings.llm_api_key:
        print("[ERROR] 未配置 LLM_API_KEY，请检查 .env")
        sys.exit(1)

    client = openai.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    course_id = "d6a221aa-2669-4cd5-bd78-616f846eb2d5"

    # 3. 逐章生成
    all_questions = []
    for ch in chapters:
        unit_count = len(ch["units"])
        print(f"\n处理 {ch['chapter']} ({unit_count} 条 unit)...")
        questions = await generate_for_chapter(client, ch)
        valid, warnings = validate_questions(questions, ch)

        for w in warnings:
            print(f"  {w}")

        # 注入 course_id
        for q in valid:
            q["course_id"] = course_id

        all_questions.extend(valid)
        print(f"  生成 {len(questions)} 道 → 校验通过 {len(valid)} 道")

    # 4. 输出候选文件
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"生成完成: {len(all_questions)} 道候选题目")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"\n问题类型分布:")
    from collections import Counter
    type_counts = Counter(q["question_type"] for q in all_questions)
    for t, c in type_counts.items():
        print(f"  {t}: {c} 道")
    print(f"\n下一步: 人工校验后改名为 eval_questions.json")


if __name__ == "__main__":
    asyncio.run(main())
