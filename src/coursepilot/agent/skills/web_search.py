"""网络搜索 Skill：教材检索不足时的备用知识源（P3）。

搜索引擎优先级：Bing（国内可用）→ Sogou → 空列表。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_MAX_SNIPPET_LEN = 500
_TIMEOUT = 10.0

# ── 搜索引擎配置 ──────────────────────────────────────────
# 按优先级排列，第一个成功的返回
_SEARCH_ENGINES: list[dict[str, Any]] = [
    {
        "name": "Bing",
        "url": "https://cn.bing.com/search",
        "method": "GET",
        "params": lambda q: {"q": q},
        "result_sel": ".b_algo",
        "title_sel": "h2 a",
        "snippet_sel": ".b_caption p",
        "url_attr": "href",
    },
    {
        "name": "Sogou",
        "url": "https://www.sogou.com/web",
        "method": "GET",
        "params": lambda q: {"query": q},
        "result_sel": ".vrwrap",
        "title_sel": ".vr-title a, h3 a",
        "snippet_sel": ".star-wiki, .str-text, .vr-title ~ div",
        "url_attr": "href",
    },
]


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
        })

    if results:
        logger.info("web_search[%s]: query=%s, results=%d", name, query, len(results))
    return results


async def web_search(query: str, top_k: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """多引擎并行搜索，取最先成功的引擎结果。

    按 _SEARCH_ENGINES 优先级顺序依次尝试，
    第一个返回非空结果的引擎即被采用。
    """
    for engine in _SEARCH_ENGINES:
        results = await _search_one(engine, query, top_k)
        if results:
            return results
    logger.warning("web_search: 所有搜索引擎均失败，query=%s", query)
    return []


def format_web_context(results: list[dict[str, Any]], query: str) -> str:
    """将搜索结果格式化为 XML context（与 RAG retriever 兼容）。"""
    if not results:
        return ""

    parts = ['<source id="1" path="网络搜索" pages="" book="网络搜索结果">']
    parts.append(f"查询：{query}")
    parts.append("")

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        parts.append(f"[{i}] {title}")
        if snippet:
            parts.append(f"    {snippet}")
        if url:
            parts.append(f"    来源：{url}")
        parts.append("")

    parts.append("</source>")
    return "\n".join(parts)
