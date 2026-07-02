"""批改答案 Skill

将学生提交的答案与正确答案比对，归因错误到知识点
Phase 2 为纯逻辑实现（无需 LLM），Phase 3 加入 LLM 分析错误原因
"""

async def grade_answers(
    quiz_data: dict,
    student_answers: dict[str, str]
) -> dict:
    """批改学生答案

    Args:
        quiz_data: {"questions": [...]}
        student_answers: {"0": "A", "1": "B", "2": "C",} → 索引到答案

    Returns:
        {"result": [...], "total": 3, "correct": 2,
        "score": 0.67, "error_kps": ["OS/xxx"]}
    """
    questions = quiz_data.get("questions", [])
    results = []
    error_kps = []

    for i, q in enumerate(questions):
        student_ans = student_answers.get(str(i), "")
        correct = student_ans == q.get("correct_answer", "")
        results.append({
            "index": i,
            "question": q.get("question_text", "")[:50],
            "correct": correct,
            "student_answer": student_ans,
            "correct_answer": q.get("correct_answer", ""),
            "kp_path": q.get("kp_path", ""),
        })
        if not correct:
            error_kps.append(q.get("kp_path", ""))

    total = len(questions)
    correct_count = sum(1 for r in results if r["correct"])
    score = correct_count / total if total > 0 else 0.0

    return {
        "results": results,
        "total": total,
        "correct": correct_count,
        "score": round(score, 2),
        "error_kps": error_kps,
    }
