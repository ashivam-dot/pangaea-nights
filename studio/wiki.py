"""English Wikipedia articles as plain text, pinned to a revision so every claim can be traced.

Wikipedia text is CC BY-SA 4.0. The writer uses it only as research: narration is an original script, and the
episode description lists every article (with its revision) as a source.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
UA = "PangaeaNightsStudio/1.0 (https://github.com/ashivam-dot/pangaea-nights; 335661989+ashivam-dot@users.noreply.github.com)"
DROP_SECTIONS = re.compile(
    r"^(references|see also|external links|further reading|notes|bibliography|sources|citations|gallery|"
    r"footnotes|explanatory notes|in popular culture|popular culture|cultural significance|in culture|"
    r"cultural depictions|in fiction|in media|human uses|uses|trade|commercial|economic importance|"
    r"conservation status|status|legal status|tourism|history of research|research history|"
    r"history of discovery|discovery and naming|discovery|naming|etymology|taxonomic history|"
    r"relationship with humans|interaction with humans|human interaction|threats|legacy)$", re.I)


def _get(params: dict, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            r = requests.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=60,
                             headers={"User-Agent": UA})
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


def clean(text: str) -> str:
    """Drop sections about people, culture and bibliography; tidy whitespace and leftover markup."""
    out, keep = [], True
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
        if m:
            keep = not DROP_SECTIONS.match(m.group(2).strip())
            if keep:
                out.append(f"\n## {m.group(2).strip()}")
            continue
        if keep:
            out.append(line)
    s = "\n".join(out)
    s = re.sub(r"\{\\displaystyle[^}]*\}", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fetch(title: str, cache: Path | None = None) -> dict:
    """{title, revid, url, text}; the text is cleaned plain text of the whole article."""
    if cache:
        path = cache / (re.sub(r"[^A-Za-z0-9]+", "_", title) + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    d = _get({"action": "query", "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
              "rvprop": "ids", "inprop": "url", "redirects": 1, "titles": title})
    page = d["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"Wikipedia has no article '{title}'")
    revid = page["revisions"][0]["revid"]
    doc = {"title": page["title"], "revid": revid,
           "url": f"https://en.wikipedia.org/w/index.php?title={page['title'].replace(' ', '_')}&oldid={revid}",
           "page_url": page.get("fullurl", ""), "text": clean(page.get("extract", ""))}
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def fetch_all(titles: list[str], cache: Path | None = None, max_chars: int = 60000) -> list[dict]:
    docs = []
    for t in titles:
        doc = fetch(t, cache)
        doc["text"] = doc["text"][:max_chars]
        docs.append(doc)
    return docs
