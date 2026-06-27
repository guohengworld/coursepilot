"""第二批出题：温度 0.8，每章追加 2 道不同知识点的题，与第一批合并。"""
import asyncio, json, sys
from pathlib import Path
from collections import Counter

import openai
from coursepilot.config import settings

# 复用第一批的素材加载和解析函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_eval_questions import (
    load_and_group, build_unit_reference, parse_llm_output, validate_questions,
    SYSTEM_PROMPT,
)

BATCH1_PATH = Path("eval/questions/eval_questions_candidate.json")
OUTPUT_PATH = Path("eval/questions/eval_questions_candidate.json")


async def main():
    batch1 = json.loads(BATCH1_PATH.read_text(encoding="utf-8"))
    existing_questions = {q["question"] for q in batch1}
    existing_kps = {q["kp_path"] for q in batch1}
    print(f"第一批: {len(batch1)} 道, 覆盖 {len(existing_kps)} 个 KP")

    client = openai.AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    chapters = load_and_group()
    course_id = "d6a221aa-2669-4cd5-bd78-616f846eb2d5"

    # 修改 prompt，要求生成与已有题目不同类型的题
    enrichment_prompt = SYSTEM_PROMPT.replace(
        "请为这一章生成 2 道问答题，覆盖不同的知识点和问题类型。",
        "请为这一章生成 2 道新的问答题。优先出 calculation（计算题）和 comparison（辨析比较）类型。不要重复已有的问题。"
    )

    new_questions = []
    for ch in chapters:
        unit_ref = build_unit_reference(ch)
        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": enrichment_prompt},
                    {"role": "user", "content": f"请为以下教材章节生成评估问答对：\n\n{unit_ref}"},
                ],
                temperature=0.8,
                max_tokens=8192,
            )
        except Exception as e:
            print(f"  {ch['chapter']}: API error: {e}")
            continue

        text = resp.choices[0].message.content
        qs = parse_llm_output(text)
        valid, warns = validate_questions(qs, ch)
        for q in valid:
            q["course_id"] = course_id

        # 去重
        unique = [q for q in valid if q["question"] not in existing_questions]
        new_questions.extend(unique)
        existing_questions.update(q["question"] for q in unique)
        print(f"  {ch['chapter']}: {len(qs)} gen -> {len(valid)} valid -> {len(unique)} new")

    merged = batch1 + new_questions
    OUTPUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    tc = Counter(q["question_type"] for q in merged)
    print(f"\n最终: {len(merged)} 道 (新增 {len(new_questions)} 道)")
    for t, c in tc.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    asyncio.run(main())
