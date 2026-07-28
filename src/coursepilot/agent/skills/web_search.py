"""网络搜索 Skill：教材检索不足时的备用知识源（P3）。

多引擎合并搜索：搜狗（主，中文高数搜索最佳）→ Bing（补充）。
结果去重 + 噪声过滤后返回 Top-K。
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_MAX_SNIPPET_LEN = 500
_TIMEOUT = 10.0

# ── 搜索引擎配置 ──────────────────────────────────────────
# 按优先级排列，全部尝试后合并去重
_SEARCH_ENGINES: list[dict[str, Any]] = [
    {   # 搜狗：中文高数内容最佳，测试显示持续出高质量结果
        "name": "Sogou",
        "url": "https://www.sogou.com/web",
        "method": "GET",
        "params": lambda q: {"query": q},
        "result_sel": ".vrwrap",
        "title_sel": ".vr-title a, h3 a",
        "snippet_sel": ".star-wiki, .str-text, .vr-title ~ div",
        "url_attr": "href",
    },
    {   # Bing：中文数学容易返回字典释义，但偶有补充价值
        "name": "Bing",
        "url": "https://cn.bing.com/search",
        "method": "GET",
        "params": lambda q: {"q": q},
        "result_sel": ".b_algo",
        "title_sel": "h2 a",
        "snippet_sel": ".b_caption p",
        "url_attr": "href",
    },
]

# ── 噪声关键词：搜索引擎经常返回的无关结果特征 ──
_NOISE_TITLE_PAT = re.compile(
    r"(汉语|汉字|拼音|部首|笔顺|新华字典|意思|解释|笔画|读音|"
    r"怎么了|怎么读|多少钱|官网|下载|手机版|游戏|攻略|价格|怎么样)",
    re.UNICODE,
)
_NOISE_URL_PAT = re.compile(
    r"(hanyu\.baidu|zdic|guoxue|hanzi|chengyu|"
    r"game|play|sina\.com\.cn/.*(ent|sports|tech))",
    re.IGNORECASE,
)

# 数学/高数相关关键词：结果标题含其一即视为相关
_MATH_KW = frozenset({
    "定理", "公式", "极限", "导数", "微分", "积分", "函数", "方程",
    "证明", "推导", "求解", "计算", "罗尔", "拉格朗日", "柯西",
    "泰勒", "费马", "洛必达", "中值", "连续", "可导", "收敛",
    "级数", "向量", "矩阵", "概率", "统计", "线性", "代数",
    "几何", "微积分", "高等数学", "高数", "数学分析",
    "考研", "真题", "习题", "例题", "解题", "作业帮",
    "CSDN", "知乎", "百度文库", "博客园", "bilibili", "文库",
    "考点", "知识点", "条件", "结论",
})


def _is_noise(title: str, url: str) -> bool:
    """判断搜索结果是否为噪声（字典解释、游戏、无关内容）"""
    if _NOISE_TITLE_PAT.search(title):
        return True
    if _NOISE_URL_PAT.search(url):
        return True
    return False


def _is_relevant(title: str, snippet: str) -> bool:
    """判断结果是否与数学/学术内容相关"""
    text = title + snippet
    return any(kw in text for kw in _MATH_KW)


async def _search_one(
    engine: dict[str, Any],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """尝试一个搜索引擎，返回结果列表或空列表"""
    name = engine["name"]
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True,
        ) as client:
            params = engine["params"](query)
            if engine["method"] == "GET":
                resp = await client.get(
                    engine["url"],
                    params=params,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
            else:
                resp = await client.post(
                    engine["url"],
                    data=params,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("%s 搜索失败: %s", name, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result_div in soup.select(engine["result_sel"]):
        if len(results) >= top_k:
            break
        title_el = result_div.select_one(engine["title_sel"])
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        url = title_el.get(engine["url_attr"], "")

        snippet_el = result_div.select_one(engine["snippet_sel"])
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        if len(snippet) > _MAX_SNIPPET_LEN:
            snippet = snippet[:_MAX_SNIPPET_LEN] + "..."

        results.append({
            "title": title,
            "snippet": snippet,
            "url": url,
            "content": snippet,
            "source": name,
        })

    logger.info("web_search[%s]: query=%s, raw=%d", name, query, len(results))
    return results


async def web_search(query: str, top_k: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """多引擎搜索，合并、去重、过滤噪声后返回 Top-K。

    策略：
    1. 所有引擎依次尝试，汇总初始结果
    2. 按标题去重
    3. 过滤噪声（字典、游戏等无关内容）
    4. 按相关度排序：含数学关键词的优先
    5. 取 Top-K
    """
    all_raw: list[dict[str, Any]] = []
    for engine in _SEARCH_ENGINES:
        results = await _search_one(engine, query, top_k + 2)  # 多取几个给过滤留余量
        all_raw.extend(results)

    if not all_raw:
        logger.warning("web_search: 所有引擎均失败，query=%s", query)
        return []

    # 去重
    seen_titles = set()
    deduped: list[dict[str, Any]] = []
    for r in all_raw:
        t = r["title"]
        if t not in seen_titles:
            seen_titles.add(t)
            deduped.append(r)

    # 过滤噪声
    filtered = []
    noise_count = 0
    for r in deduped:
        if _is_noise(r["title"], r["url"]):
            noise_count += 1
            continue
        filtered.append(r)

    if noise_count:
        logger.debug("web_search: 过滤噪声 %d 条", noise_count)

    if not filtered:
        logger.warning("web_search: 全部被过滤为噪声")
        return []

    # 按相关度排序：含数学关键词的排前面
    filtered.sort(key=lambda r: (0 if _is_relevant(r["title"], r["snippet"]) else 1))

    result = filtered[:top_k]
    logger.info(
        "web_search: 最终 %d 条 (引擎=%s, 总原始=%d, 去重后=%d, 过滤后=%d)",
        len(result),
        "+".join(r["source"] for r in result),
        len(all_raw), len(deduped), len(filtered),
    )
    return result


def format_web_context(results: list[dict[str, Any]], query: str) -> str:
    """将搜索结果格式化为 XML context（与 RAG retriever 兼容）。"""
    if not results:
        return ""

    # 标注来源引擎
    sources = sorted(set(r.get("source", "?") for r in results if r.get("source")))
    source_tag = "+".join(sources) if sources else "网络"

    parts = [f'<source id="1" path="网络搜索/{source_tag}" pages="" book="网络搜索结果">']
    parts.append(f"查询：{query}")
    parts.append("")

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        src = r.get("source", "")
        parts.append(f"[{i}] {title}" + (f" (来自{src})" if src else ""))
        if snippet:
            parts.append(f"    {snippet}")
        if url:
            parts.append(f"    来源：{url}")
        parts.append("")

    parts.append("</source>")
    return "\n".join(parts)
