"""The proven-topics feed (weekly, .github/workflows/topics.yml): what is already working on the top AI channels,
retold as our own deep-time premises.

    python -m studio.cli topics-scan [--dry]

1. Scan (studio/outliers.py): the competitor channels' recent uploads, public metadata only, and their outliers.
2. Shorts outliers: Gemini maps each outlier's TITLE (the only thing of theirs it ever sees) to a deep-time idea for
   one of our four ladder series, or rejects it, and names an English Wikipedia article. The article is fetched at
   its current revision, which is pinned (wiki_title and wiki_revid come from the fetch, never from the model).
   Gemini then writes a full premise in the bank schema, grounded in one sentence copied from that revision.
3. Every gate a bank premise passes (ladder.premise_problems, the same check ladder-check runs), plus: the sentence
   is really in the revision; the article and the premise are deep-time Earth history (no people, no mythology);
   nothing reuses a scanned competitor title or shares a 4-word run with one, or names a competitor's channel,
   character or brand (copy_problems); and it duplicates nothing in the banks or already published. One failure
   rejects the premise: a second draft may fix it, no gate is ever loosened.
4. Accepted premises go to the ledger on the state branch (ladder.proven) with their provenance; the daily picker
   takes them ahead of untested bank premises (ladder.pick). At most WEEKLY_CAP are added in any 7 days; main is
   never written by CI.
5. Long-form outliers: matched to an existing backlog topic by shared words (no new topics); a clear match records a
   boost the planner uses to move that topic forward (planner.boosts).
6. Supply: days of premises left (banks + proven) under the real picker and rotation rules; under SUPPLY_DAYS_MIN
   it alerts. A dry run computes and prints everything but saves nothing and messages no one.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone

import requests

from . import config, gates, ladder, ledger, outliers

WEEKLY_CAP = 7  # proven premises added in any 7 days
SUPPLY_DAYS_MIN = 14
MAP_BATCH = 12  # Shorts outliers Gemini sees per scan
DRAFTS = 2  # a premise gets one rewrite with its problems listed
PROVEN_FROM = 101  # proven ids are <series>-101 and up, clear of hand-written bank ids
NEAR_TITLE = 0.6  # word overlap that makes two titles the same premise (the banks' closest pair is 0.46)
COPY_NGRAM = 4
ARTICLE_CHARS = 18000
KEEP_OUTLIERS, KEEP_TRIED, KEEP_SCANS = 400, 2000, 26
MATCH_MIN = 4  # long-form: weighted shared words a backlog topic needs to be boosted
# topics.yml gives the Scan step 30 minutes (the job 40): the Gemini stage starts no new idea after this, and each
# request is held to LLM_TIMEOUT_S x LLM_ATTEMPTS and the overall deadline, so the run always reaches the save.
DRAFT_BUDGET_S = 20 * 60
LLM_TIMEOUT_S, LLM_ATTEMPTS = 150, 2

# Geological periods before recorded history. The Holocene (and the Quaternary, which contains it) is the age of
# recorded human history and its disasters, and the eon/era umbrellas (Phanerozoic, Cenozoic) name no one period:
# none of them is ever a premise's era. No bank premise uses any of them.
NOT_ERAS = {"Holocene", "Quaternary", "Phanerozoic", "Cenozoic"}
ERAS = set(gates.GEOLOGY[:gates.GEOLOGY.index("Earth")]) - NOT_ERAS
# Not deep-time Earth history: people, and myth or fiction. Checked on every line we publish, on the art direction
# and on every image prompt (as composed for the image model). "god rays" is a lighting term, not a god.
OFF_NICHE = re.compile(
    r"\b(people|person|humans?|mankind|man|men|woman|women|child|children|boy|girl|tribes?|villages?|caveman|"
    r"cavemen|neanderthals?|hominins?|homo|kings?|queens?|army|armies|soldiers?|gods?(?![-\s]rays?\b)|goddess\w*|"
    r"myth\w*|"
    r"legend\w*|folklore|dragons?|demons?|ghosts?|spirits?|magic\w*|zombies?|aliens?|curse\w*|angels?|wizards?|"
    r"witch\w*|bible|jesus|celebrit\w*|influencers?)\b", re.I)
DEEP_TIME = re.compile(r"million years|billion years|fossil|extinct|prehistoric|geolog|paleontolog|palaeontolog",
                       re.I)
# A date of at least ten thousand years ago, as Wikipedia writes one.
DEEP_DATE = re.compile(r"\b\d[\d,.]*\s*(?:million|billion|thousand)\s+years|\b\d{1,3}(?:,\d{3})+\s+years|"
                       r"\b\d+(?:\.\d+)?\s*(?:Ma|Ga|ka|mya|kya)\b")
# A biography's lead: "(born 1950)", "(died 1850)", "(1799–1847)", "(21 May 1799 – 9 March 1847)", with or without a
# pronunciation first ("( DAR-win; 12 February 1809 – 19 April 1882)") or US-style dates; or "X was an English
# fossil collector" (a nationality before the occupation, so "a site studied by geologists" is not a person).
BIOGRAPHY = re.compile(
    r"\(born\b|\bborn \d{3,4}\b|\((?:died|d\.)\s|\bdied \d{3,4}\b|"
    r"\([^()]{0,80}?\b\d{3,4}\s*[–-]\s*[^()]{0,40}?\b\d{3,4}\)|"
    r"\b(?:was|is) an? (?:[A-Z][\w-]+ )+(?:[a-z]+ )?(?:palaeontologist|paleontologist|geologist|naturalist|biologist|"
    r"scientist|explorer|collector|anatomist|zoologist|botanist|physician|writer|author|engineer)s?\b")
# Recorded history: a calendar year in the title, an AD/BC date or casualty words in the lead. A real disaster with
# real victims is never an entertainment premise.
HISTORY_YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
HISTORY_LEAD = re.compile(r"\b\d+\s*(?:AD|BC|BCE|CE)\b|\bA\.?D\.? \d+")
CASUALTIES = re.compile(r"death toll|casualt|fatalit|victims|people (?:were )?killed|"
                        r"killing (?:over |at least |more than |about |some )?\d", re.I)
SERIES_FORMAT = {
    "survive_night": "One night in a prehistoric place: eight places to sleep or hide, from safest to deadliest.",
    "level_1_8": "Level 1 to 8: eight escalating versions of one prehistoric thing (pets, armour, rides, the "
                 "biggest of a kind), mildest to most extreme.",
    "countdown": "A deep-time catastrophe is seconds or hours away (asteroid, tsunami, eruption, freeze): eight "
                 "places to run or hide.",
    "how_deep": "An ancient ocean or lake: eight depths or places to wait out the night, surface to abyss.",
}


# ---------------------------------------------------------------------------------------------------- gates


def tokens(text: str) -> list[str]:
    """Case-, accent- and punctuation-insensitive words."""
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", folded)


def _grams(toks: list[str], n: int = COPY_NGRAM) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _published_lines(p: dict) -> dict[str, str]:
    """Every line a viewer reads or hears, by field."""
    out = {k: str(p.get(k, "")) for k in ("title", "banner", "hook_vo", "verdict_vo", "description")}
    for o in p.get("options", []):
        out[f"option {o.get('n')}"] = f"{o.get('label', '')}. {o.get('vo', '')}"
    out["fact"] = str((p.get("fact") or {}).get("text", ""))
    out["tags"] = ", ".join(p.get("tags", []))
    return out


def copy_problems(p: dict, titles: list[str], names: list[str]) -> list[str]:
    """The competitor gate. No published line may contain a scanned competitor title verbatim or share a run of
    COPY_NGRAM words with one, and nothing at all (image prompts and art direction included) may name a competitor's
    channel, character or brand."""
    out = []
    theirs = [(t, tokens(t)) for t in titles if tokens(t)]
    their_grams = [(t, _grams(tt)) for t, tt in theirs]
    for field_, text in _published_lines(p).items():
        mine = tokens(text)
        joined = f" {' '.join(mine)} "
        hit = next((t for t, tt in theirs if (len(tt) >= 3 or tt == mine) and f" {' '.join(tt)} " in joined), None)
        if hit:
            out.append(f"{field_} reuses a competitor title verbatim: {hit!r}")
            continue
        g = _grams(mine)
        hit = next((t for t, tg in their_grams if g & tg), None)
        if hit:
            shared = " ".join(sorted(g & dict(their_grams)[hit])[0])
            out.append(f"{field_} shares '{shared}' with a competitor title: {hit!r}")
    every = " ".join([*_published_lines(p).values(), str(p.get("art_direction", "")),
                      *(str(o.get("image_prompt", "")) for o in p.get("options", []))])
    low = f" {' '.join(tokens(every))} "
    out += [f"names a competitor's channel, character or brand: {n}" for n in names
            if tokens(n) and f" {' '.join(tokens(n))} " in low]
    return out


def _image_lines(p: dict) -> dict[str, str]:
    """What the image model is told: the art direction, each option's scene, and the composed prompts."""
    out = {"art_direction": str(p.get("art_direction", ""))}
    opts = [o for o in p.get("options") or [] if isinstance(o, dict)]
    for o in opts:
        out[f"option {o.get('n')} image"] = str(o.get("image_prompt", ""))
    if opts and len(opts) == len(p.get("options") or []):
        try:
            out.update({f"{j['id']} prompt": j["prompt"] for j in ladder.image_jobs(p)})
        except (KeyError, TypeError, AttributeError):
            pass
    return out


def niche_problems(p: dict) -> list[str]:
    """Deep-time Earth history only: a geological era, and no people or mythology in anything we publish or draw."""
    out = [] if p.get("era") in ERAS else [f"era {p.get('era')!r} is not a geological period"]
    for field_, text in {**_published_lines(p), **_image_lines(p)}.items():
        out += [f"{field_} is off the niche: '{m.group(0)}'" for m in OFF_NICHE.finditer(text)]
    return out


def article_problems(doc: dict, era: str | None = None) -> list[str]:
    """The pinned article must be about deep time (and name the premise's era or a date of at least ten thousand
    years ago), never a biography, and never a disaster of recorded history."""
    text, title = doc.get("text", ""), doc.get("title", "")
    body = text[:ARTICLE_CHARS]
    out = [] if DEEP_TIME.search(body) else [f"{title!r} is not about deep time"]
    if era is not None and not (re.search(r"\b" + re.escape(era) + r"\b", body) or DEEP_DATE.search(body)):
        out.append(f"{title!r} names neither the {era} nor a deep-time date")
    if BIOGRAPHY.search(text[:400]):
        out.append(f"{title!r} is about a person")
    if HISTORY_YEAR.search(title) or HISTORY_LEAD.search(text[:600]):
        out.append(f"{title!r} is an event of recorded history")
    if CASUALTIES.search(text[:600]):
        out.append(f"{title!r} is about a disaster with human victims")
    out += [f"article names a competitor brand: {b}" for b in ladder.blocklist_hits(title)]
    return out


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def source_problems(p: dict, doc: dict) -> list[str]:
    """The fact rests on the revision it cites: its sentence is in that revision's text, word for word."""
    fact = p.get("fact") or {}
    out = []
    if fact.get("wiki_revid") != doc.get("revid") or fact.get("wiki_title") != doc.get("title"):
        out.append("fact does not cite the fetched revision")
    src = _squash(fact.get("source_sentence", ""))
    if not src or len(src) > 400 or src not in _squash(doc.get("text", "")):
        out.append("fact.source_sentence is not a sentence of the pinned revision")
    return out


def _title_words(title: str) -> set[str]:
    stop = {"the", "a", "an", "of", "in", "to", "and", "for", "on", "at", "with", "your", "you", "one", "night",
            "is", "it", "do", "who", "what", "how"}
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in tokens(title)} - stop


def title_similarity(a: str, b: str) -> float:
    wa, wb = _title_words(a), _title_words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def dedupe_problems(p: dict, pool: list[dict], used_titles: list[str]) -> list[str]:
    """Nothing the banks, the proven list or the channel already has: no shared id, title, banner, hook, art
    direction or image prompt, no near-duplicate title, and the bank novelty ceiling against every script."""
    others = [q for q in pool if q is not p]
    out = []
    if p["id"] in {q["id"] for q in others}:
        out.append(f"id {p['id']} is taken")
    for key in ("title", "banner", "hook_vo", "art_direction"):
        if p[key].strip().lower() in {q[key].strip().lower() for q in others}:
            out.append(f"{key} duplicates another premise")
    theirs = {o["image_prompt"].strip().lower() for q in others for o in q["options"]}
    if any(o["image_prompt"].strip().lower() in theirs for o in p["options"]):
        out.append("an image prompt duplicates another premise")
    near = max(((title_similarity(p["title"], t), t) for t in [q["title"] for q in others] + used_titles),
               default=(0.0, ""))
    if near[0] >= NEAR_TITLE:
        out.append(f"title is too close to {near[1]!r} ({near[0]:.2f})")
    nov = ladder.overlap(ladder.script_text(p), [ladder.script_text(q) for q in others])
    if nov > ladder.NOVELTY_MAX:
        out.append(f"script shares {nov:.2f} of its 4-grams with another premise (ceiling {ladder.NOVELTY_MAX})")
    return out


def problems(p: dict, doc: dict, pool: list[dict], used_titles: list[str], titles: list[str],
             names: list[str]) -> list[str]:
    """Every gate: the bank gates first (the same ones ladder-check runs), then source, niche, copy and dedupe."""
    return (ladder.premise_problems(p, p["series"]) + source_problems(p, doc) + niche_problems(p) +
            copy_problems(p, titles, names) + dedupe_problems(p, pool, used_titles))


def next_id(series: str, taken: set[str]) -> str:
    n = PROVEN_FROM
    while f"{series}-{n}" in taken:
        n += 1
    return f"{series}-{n}"


# ---------------------------------------------------------------------------------------------------- Gemini

SYSTEM = """You write for Pangaea Nights, a YouTube channel of deep-time Earth history: prehistoric life, ancient
seas, ice ages, extinctions and the changing planet, all grounded in English Wikipedia. Its Shorts are choice
ladders: a question, eight options from safest to deadliest, then "Real answer: number N, because ..." resting on one
sentence of a Wikipedia article.

Rules you never break: no people of any kind (no humans, hominins, historical or living persons, explorers or
scientists), no mythology, folklore, religion, fiction, films, games or brands. Another channel's video title may be
given to you only as a signal of what an audience enjoyed: keep its underlying appeal (a dilemma, an extreme of
scale, a countdown, a survival test) and never its words, names, characters, catchphrases or branding. Write every
line fresh, in plain English."""

_S, _I = {"type": "STRING"}, {"type": "INTEGER"}
MAP_SCHEMA = {"type": "OBJECT", "properties": {"ideas": {"type": "ARRAY", "items": {
    "type": "OBJECT", "properties": {"video_id": _S, "series": {"type": "STRING", "enum": list(ladder.SERIES)},
                                     "era": _S, "idea": _S, "wiki_title": _S, "reject": _S},
    "required": ["video_id", "series", "era", "idea", "wiki_title", "reject"]}}}, "required": ["ideas"]}
OPTION_SCHEMA = {"type": "OBJECT", "properties": {"n": _I, "label": _S, "vo": _S, "danger": _I, "image_prompt": _S},
                 "required": ["n", "label", "vo", "danger", "image_prompt"]}
PREMISE_SCHEMA = {"type": "OBJECT", "properties": {
    "era": _S, "title": _S, "banner": _S, "hook_vo": _S, "art_direction": _S,
    "options": {"type": "ARRAY", "items": OPTION_SCHEMA}, "answer_n": _I, "verdict_vo": _S,
    "fact": {"type": "OBJECT", "properties": {"text": _S, "source_sentence": _S}, "required": ["text", "source_sentence"]},
    "description": _S, "tags": {"type": "ARRAY", "items": _S}, "related_topic_hint": _S},
    "required": ["era", "title", "banner", "hook_vo", "art_direction", "options", "answer_n", "verdict_vo", "fact",
                 "description", "tags", "related_topic_hint"]}


def map_ideas(llm, rows: list[dict]) -> list[dict]:
    """[{video_id, series, era, idea, wiki_title}] for the outliers that map honestly onto deep time. Only each
    outlier's id, title and numbers are sent; anything the model invents outside them is dropped."""
    if not rows:
        return []
    ask = ("For each video below, propose one Pangaea Nights Short in one of these series, or reject it when there "
           "is no honest deep-time version (then say why in 'reject'; otherwise leave 'reject' empty).\n\nSeries:\n" +
           "\n".join(f"- {k}: {v}" for k, v in SERIES_FORMAT.items()) +
           f"\n\nera: one geological period name, e.g. {', '.join(sorted(ERAS)[:12])}.\nidea: one sentence, in your "
           "own words, about prehistoric life or Earth. wiki_title: the exact title of one English Wikipedia article "
           "about a prehistoric organism, event or place that grounds the answer.\n\nVideos:\n" +
           "\n".join(json.dumps({"video_id": r["video_id"], "title": r["title"], "views": r["views"],
                                 "times_channel_median": r["ratio"]}, ensure_ascii=False) for r in rows))
    out = llm.generate(SYSTEM, ask, MAP_SCHEMA, temperature=0.4)
    known = {r["video_id"] for r in rows}
    ideas = []
    got = out.get("ideas") if isinstance(out, dict) else None
    if not isinstance(got, list):  # a malformed reply maps nothing: no outlier counts as looked at
        raise ValueError(f"malformed map reply ({type(out).__name__})")
    for i in got:
        if not isinstance(i, dict):
            continue
        if (i.get("video_id") in known and not str(i.get("reject", "")).strip() and i.get("series") in ladder.SERIES
                and i.get("era") in ERAS and str(i.get("wiki_title", "")).strip() and str(i.get("idea", "")).strip()):
            ideas.append({k: str(i[k]).strip() for k in ("video_id", "series", "era", "idea", "wiki_title")})
    return ideas


def _example(banks: dict, series: str) -> str:
    p = (banks.get(series) or [None])[0]
    if not p:
        return ""
    keep = {k: v for k, v in p.items() if k not in ("series", "id", "provenance", "added_at")}
    keep["fact"] = {k: p["fact"][k] for k in ("text", "source_sentence")}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _rules() -> str:
    lim = ladder.LIMITS
    bans = sorted({w.strip() for w in gates.HUMAN_WORDS + gates.TEXT_WORDS + ladder.EXTRA_PROMPT_BANS})
    return "\n".join([
        f"- title: {ladder.TITLE_CHARS[0]}-{ladder.TITLE_CHARS[1]} characters, at most {ladder.MAX_EMOJI} emoji, and "
        f"one of these within its first five words: {', '.join(ladder.FAMOUS)}.",
        f"- banner at most {lim['banner_words']} words; hook_vo at most {lim['hook_words']} words.",
        f"- exactly {ladder.OPTIONS} options numbered 1..{ladder.OPTIONS}; danger an integer 1..8 that never goes down "
        f"and ends higher than it starts; labels at most {lim['label_words']} words and all different; vo at most "
        f"{lim['option_vo_words']} words.",
        f"- answer_n 1..8; verdict_vo starts 'Real answer: number N, because' and has at most {lim['verdict_words']} "
        "words.",
        "- the whole script (hook, eight vo lines, verdict) is about 85 words and never more than 95: it must be "
        "spoken in under 40 seconds.",
        "- fact.source_sentence: ONE sentence copied character for character from the article above. fact.text: a "
        "plain restatement of it. Every number and every capitalised name in fact.text and in the verdict must "
        "appear in that sentence (or be the article title).",
        "- if the source sentence hedges (may, might, likely, thought, suggests, estimated...), the verdict hedges "
        "too (probably, may, likely).",
        "- art_direction: exactly one sentence (place, light, lens).",
        f"- every image_prompt: at least 18 words describing one scene, with a warm focal light (one of: "
        f"{', '.join(ladder.WARM)}); never these words: {', '.join(bans)}; no '-like' and no 'resembles'.",
        f"- description: 2-3 sentences. tags: 1-{ladder.MAX_TAGS} plain short phrases, no '#'.",
        "- no people, no mythology, no brand, film, game or channel names anywhere.",
    ])


def draft(llm, idea: dict, doc: dict, example: str, fix: list[str] | None = None) -> dict:
    ask = (f"Series: {idea['series']} ({SERIES_FORMAT[idea['series']]})\nEra: {idea['era']}\nIdea: {idea['idea']}\n\n"
           f"Wikipedia article {doc['title']!r} (the only source you may use):\n<<<\n{doc['text'][:ARTICLE_CHARS]}\n"
           f">>>\n\nA finished premise of this series, for its shape and tone only (never reuse its wording):\n"
           f"{example}\n\nRules:\n{_rules()}\n\nWrite one new premise as JSON.")
    if fix:
        ask += "\n\nYour previous draft failed these checks; fix every one:\n- " + "\n- ".join(fix[:20])
    return llm.generate(SYSTEM, ask, PREMISE_SCHEMA, temperature=0.7)


def shape(raw: dict, series: str, pid: str, doc: dict) -> dict:
    """The bank schema exactly, from a model reply: unknown keys dropped, the citation taken from the fetch. A reply
    of the wrong shape (a fallback model is not held to the schema) becomes an empty premise the gates reject."""
    raw = raw if isinstance(raw, dict) else {}
    opts = raw.get("options") if isinstance(raw.get("options"), list) else []
    opts = [{"n": o.get("n"), "label": str(o.get("label", "")), "vo": str(o.get("vo", "")),
             "danger": o.get("danger"), "image_prompt": str(o.get("image_prompt", ""))}
            for o in opts if isinstance(o, dict)]
    fact = raw.get("fact") if isinstance(raw.get("fact"), dict) else {}
    tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
    return {"id": pid, "era": str(raw.get("era", "")), "title": str(raw.get("title", "")),
            "banner": str(raw.get("banner", "")), "hook_vo": str(raw.get("hook_vo", "")),
            "art_direction": str(raw.get("art_direction", "")), "options": opts, "answer_n": raw.get("answer_n"),
            "verdict_vo": str(raw.get("verdict_vo", "")),
            "fact": {"text": str(fact.get("text", "")), "wiki_title": doc["title"], "wiki_revid": doc["revid"],
                     "source_sentence": str(fact.get("source_sentence", ""))},
            "description": str(raw.get("description", "")), "tags": [str(t) for t in tags][:20],
            "related_topic_hint": str(raw.get("related_topic_hint", "")), "series": series}


def write_premise(llm, idea: dict, doc: dict, pid: str, banks: dict, pool: list[dict], used_titles: list[str],
                  titles: list[str], names: list[str]) -> tuple[dict | None, list[str]]:
    """(premise, []) when a draft passes every gate, else (None, the last draft's problems)."""
    fix: list[str] = []
    for _ in range(DRAFTS):
        p = shape(draft(llm, idea, doc, _example(banks, idea["series"]), fix), idea["series"], pid, doc)
        if p["era"] != idea["era"]:
            p["era"] = idea["era"]  # the era the idea was mapped to, which the article was checked against
        fix = problems(p, doc, pool, used_titles, titles, names)
        if not fix:
            return p, []
    return None, fix


# ---------------------------------------------------------------------------------------------------- long-form

GENERIC = set("""the a an of in to and for on at with from by is are was were be it its what if how why when who which
that this these those did does do had has have ever really most all every our your you we they their there into after
before during about as or not no long last first one two three went go gone hours hour sleep sleeping documentary
documentaries history boring fall asleep relaxing calm ambient full movie film story stories tonight night nights
bedtime earth world ancient prehistoric life animals time might been could would will part video""".split())


def _stems(text: str) -> set[str]:
    out = set()
    for w in tokens(text):
        if w in GENERIC or len(w) < 3:
            continue
        if len(w) > 7 and w.endswith("ion"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


def match_topic(title: str, backlog: list[dict], used: set[str]) -> tuple[dict, int] | None:
    """The unused backlog topic a long-form title is about: words in the topic's id, title, lead or promise weigh 2,
    words in its article titles 1. Only a clear best of at least MATCH_MIN counts; a tie boosts nothing."""
    theirs = _stems(title)
    scored = []
    for t in backlog:
        if t["id"] in used:
            continue
        strong = _stems(" ".join([t["id"].replace("-", " "), t.get("title", ""), t.get("lead", ""),
                                  t.get("promise", "")]))
        weak = _stems(" ".join(t.get("articles", [])))
        scored.append((sum(2 if w in strong else 1 for w in theirs if w in strong | weak), t))
    scored.sort(key=lambda s: -s[0])
    if not scored or scored[0][0] < MATCH_MIN or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1], scored[0][0]


def record_boosts(data: dict, rows: list[dict], backlog: list[dict], at: datetime) -> list[dict]:
    """Boost the backlog topic each long-form outlier matches (the strongest ratio wins). Returns what was boosted."""
    used = ledger.used_topics(data)
    boosts = data["topics"]["boosts"]
    made = []
    for r in rows:
        m = match_topic(r["title"], backlog, used)
        if not m:
            continue
        topic, score = m
        old = boosts.get(topic["id"])
        if old and old.get("ratio", 0) >= r["ratio"] and old.get("video_id") != r["video_id"]:
            continue
        boosts[topic["id"]] = {"ratio": r["ratio"], "score": score, "channel_id": r["channel_id"],
                               "video_id": r["video_id"], "views": r["views"], "median": r["median"],
                               "title": r["title"], "at": at.strftime("%Y-%m-%dT%H:%M:%SZ")}
        made.append({"topic": topic["id"], "video_id": r["video_id"], "ratio": r["ratio"]})
    return made


# ---------------------------------------------------------------------------------------------------- supply


def supply_days(banks: dict, data: dict, at: date, per_day: int = len(ladder.SLOTS), horizon: int = 120) -> int:
    """Whole days of Shorts the premises left can fill, picked by the real picker under the rotation rules at full
    cadence (every slot, every day): the supply ends when two series are left, not when the last premise is used."""
    sim = {"shorts": [dict(s) for s in data.get("shorts", [])],
           "ladder": {**(data.get("ladder") or {}), "queue": []}}
    for day in range(horizon):
        d = at + timedelta(days=day)
        planned: list[dict] = []
        for _ in range(per_day):
            try:
                planned.append(ladder.pick(banks, sim, d, planned))
            except ladder.BankError:
                return day
        sim["shorts"] += [{"premise": p["id"], "series": p["series"], "era": p["era"], "video_id": "sim",
                           "publish_at": f"{d.isoformat()}T14:30:00Z"} for p in planned]
    return horizon


def added_since(data: dict, at: datetime, days: int = 7) -> int:
    since = (at - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return sum(1 for p in data["ladder"]["proven"] if p.get("added_at", "") > since)


# ---------------------------------------------------------------------------------------------------- run


class _Deadline(Exception):
    pass


def run(data: dict, api, llm, fetch, banks: dict | None = None, backlog: list[dict] | None = None,
        at: datetime | None = None, cap: int = WEEKLY_CAP, budget_s: float | None = None,
        checkpoint=None, clock=time.monotonic) -> dict:
    """One weekly pass. Changes `data` in place (the caller decides whether to save); returns the summary.
    api: the YouTube Data API client or None (no scan); llm: a Gemini or None (no new premises); fetch(title) ->
    wiki.fetch-style {title, revid, text}. budget_s: wall-clock seconds for the Gemini stage (default
    DRAFT_BUDGET_S), after which no new idea is started. checkpoint(data): called whenever `data` is consistent
    (after the scan and after each idea), so a run cut short still keeps what it finished.

    An outlier counts as tried (and is never looked at again) only once it was really processed: the model rejected
    it, or its idea was fetched and drafted. A model outage, a non-retried HTTP error or a malformed reply in the
    mapping step marks nothing; ideas left over at the weekly cap or the time budget wait for next week."""
    at = at or datetime.now(timezone.utc)
    stamp = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    banks = banks if banks is not None else ladder.load_banks(strict=False)
    backlog = backlog if backlog is not None else config.backlog()
    budget_s = DRAFT_BUDGET_S if budget_s is None else budget_s
    started = clock()
    save = checkpoint or (lambda d: None)
    top = data["topics"]
    sm: dict = {"at": stamp, "scanned": 0, "units": 0, "errors": {}, "outliers": {"short": 0, "long": 0},
                "accepted": [], "rejected": [], "boosts": [], "notes": []}
    found: list[dict] = []
    titles = [o["title"] for o in top["outliers"]]
    if api is None:
        sm["notes"].append("no YouTube credentials: no scan")
    else:
        scan = outliers.scan(api, outliers.competitors())
        found = outliers.flag(scan["videos"])
        titles += [v["title"] for v in scan["videos"]]
        sm.update(scanned=len(scan["channels"]), units=scan["units"],
                  errors={k: v["error"] for k, v in scan["channels"].items() if "error" in v})
        for k in ("short", "long"):
            sm["outliers"][k] = sum(1 for o in found if o["kind"] == k)
        seen = {o["video_id"] for o in top["outliers"]}
        top["outliers"] = (top["outliers"] + [{**o, "seen": stamp} for o in found if o["video_id"] not in seen]
                           )[-KEEP_OUTLIERS:]
        sm["boosts"] = record_boosts(data, [o for o in found if o["kind"] == "long"], backlog, at)
        save(data)
    room = max(0, cap - added_since(data, at))
    tried = set(top["tried"])
    fresh = [o for o in found if o["kind"] == "short" and o["video_id"] not in tried][:MAP_BATCH]

    def mark(video_id: str) -> None:
        if video_id not in top["tried"]:
            top["tried"] = (top["tried"] + [video_id])[-KEEP_TRIED:]

    if fresh and not room:
        sm["notes"].append(f"weekly cap of {cap} proven premises reached")
    elif fresh and llm is None:
        sm["notes"].append("no Gemini key: no new premises")
    elif fresh:
        from .llm import NoModel

        names = outliers.do_not_copy()
        pool = [p for ps in ladder.with_proven(banks, data).values() for p in ps]
        used_titles = [s.get("title", "") for s in data.get("shorts", []) if s.get("video_id")]
        # every id the ledger or the banks hold, gated or not: a proven premise left out by a tightened gate keeps
        # its id, so a new premise never shares it
        taken = ({p["id"] for p in pool} | {str(p.get("id")) for p in data["ladder"]["proven"]} |
                 {p["id"] for ps in banks.values() for p in ps} | ladder.used_premises(data))
        by_id = {o["video_id"]: o for o in fresh}
        try:
            ideas = map_ideas(llm, fresh)
        except Exception as exc:  # an outage, a refused key or a malformed reply costs this week's premises only
            ideas = None
            sm["notes"].append(f"Gemini map: {type(exc).__name__}: {exc}"[:200])
        if ideas is not None:
            mapped = {i["video_id"] for i in ideas}
            for v in by_id:
                if v not in mapped:
                    sm["rejected"].append({"video_id": v, "why": ["no honest deep-time version"]})
                    mark(v)
            save(data)
            try:
                for idea in ideas:
                    if len(sm["accepted"]) >= room:
                        sm["notes"].append(f"weekly cap of {cap} reached: the other ideas wait for next week")
                        break
                    if clock() - started > budget_s:
                        raise _Deadline()
                    o = by_id[idea["video_id"]]
                    try:
                        doc = fetch(idea["wiki_title"])
                    except Exception as exc:  # a missing article rejects this idea, not the run
                        sm["rejected"].append({"video_id": o["video_id"], "why": [f"article: {type(exc).__name__}"]})
                        mark(o["video_id"])
                        save(data)
                        continue
                    bad = article_problems(doc, idea["era"])
                    if bad:
                        sm["rejected"].append({"video_id": o["video_id"], "why": bad})
                        mark(o["video_id"])
                        save(data)
                        continue
                    pid = next_id(idea["series"], taken)
                    try:
                        p, why = write_premise(llm, idea, doc, pid, banks, pool, used_titles, titles, names)
                    except (NoModel, requests.RequestException):
                        raise  # the model or the key, not this idea: stop, and leave it for next week
                    except Exception as exc:  # a reply of the wrong shape rejects this idea, not the run
                        p, why = None, [f"draft: {type(exc).__name__}"]
                    mark(o["video_id"])
                    if not p:
                        sm["rejected"].append({"video_id": o["video_id"], "why": why[:6]})
                        save(data)
                        continue
                    p.update(added_at=stamp, provenance={
                        "channel_id": o["channel_id"], "video_id": o["video_id"], "title": o["title"],
                        "views": o["views"], "median": o["median"], "ratio": o["ratio"], "published": o["published"],
                        "wiki_title": doc["title"], "wiki_revid": doc["revid"]})
                    data["ladder"]["proven"].append(p)
                    pool.append(p)
                    taken.add(pid)
                    sm["accepted"].append({"id": pid, "title": p["title"], "video_id": o["video_id"],
                                           "ratio": o["ratio"]})
                    save(data)
            except _Deadline:
                sm["notes"].append(f"time budget of {budget_s / 60:.0f} min spent: the other ideas wait for next "
                                   "week")
            except Exception as exc:  # NoModel, an HTTP error: never the scan, the boosts or the supply check
                sm["notes"].append(f"Gemini: {type(exc).__name__}: {exc}"[:200])
    full = ladder.with_proven(banks, data)
    sm["proven_unused"] = sum(1 for p in data["ladder"]["proven"] if p["id"] not in ladder.used_premises(data))
    sm["supply_days"] = supply_days(full, data, at.date())
    sm["low_supply"] = sm["supply_days"] < SUPPLY_DAYS_MIN
    top["scans"] = (top["scans"] + [{k: sm[k] for k in ("at", "scanned", "units", "outliers", "supply_days")} |
                                    {"accepted": [a["id"] for a in sm["accepted"]]}])[-KEEP_SCANS:]
    return sm


def low_supply_text(sm: dict) -> str:
    return (f"LOW SUPPLY: {sm['supply_days']} days of Shorts premises left (alert under {SUPPLY_DAYS_MIN}): add "
            "premises to bible/ladder/ or let the proven-topics scan fill the gap") if sm["low_supply"] else ""


def message(sm: dict) -> str:
    lines = [f"Pangaea Nights proven topics, {sm['at'][:10]}",
             f"- scanned {sm['scanned']} channels ({sm['units']} API units): {sm['outliers']['short']} Shorts and "
             f"{sm['outliers']['long']} long-form outliers"]
    lines += [f"- channel {k}: {v}" for k, v in sm["errors"].items()]
    lines += [f"- NEW premise {a['id']}: {a['title']} (from {a['video_id']}, {a['ratio']}x its channel median)"
              for a in sm["accepted"]]
    if sm["rejected"]:
        lines.append(f"- rejected {len(sm['rejected'])}: " + "; ".join(
            f"{r['video_id']} ({r['why'][0]})" for r in sm["rejected"][:5]))
    lines += [f"- long-form boost: {b['topic']} (from {b['video_id']}, {b['ratio']}x)" for b in sm["boosts"]]
    lines += [f"- {n}" for n in sm["notes"]]
    lines.append(f"- supply: {sm['supply_days']} days ({sm['proven_unused']} proven premises unused)")
    if sm["low_supply"]:
        lines.append(f"- {low_supply_text(sm)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- CLI


def _youtube():
    from . import youtube

    if not os.environ.get("YOUTUBE_OAUTH_JSON"):
        return None
    api = youtube.client()
    youtube.identity(api)  # our own channel, as every caller checks first (1 unit)
    return api


def _gemini():
    from .llm import Gemini, NoModel

    try:  # held well inside the Scan step's 30 minutes, so a slow Gemini day still reaches the save
        return Gemini(timeout=LLM_TIMEOUT_S, attempts=LLM_ATTEMPTS,
                      deadline=time.monotonic() + DRAFT_BUDGET_S + 4 * 60)
    except NoModel as exc:
        print(f"llm: {exc}")
        return None


def scan_cmd(args) -> None:
    from . import alert, wiki

    out = config.WORK / "topics"
    out.mkdir(parents=True, exist_ok=True)
    data = ledger.load()
    # a dispatch dry run saves nothing; a real one saves at every consistent point, so a run cut short (the Scan
    # step's timeout) still leaves the scan, the tried outliers and every accepted premise for the Save step
    keep = None if args.dry else ledger.save
    try:
        sm = run(data, _youtube(), _gemini(), lambda t: wiki.fetch(t, cache=out / "wiki"), checkpoint=keep)
    except Exception as exc:
        if not args.dry:
            alert.send(f"Pangaea Nights proven-topics scan failed: {type(exc).__name__}: {exc}"[:400])
        raise
    msg = message(sm)
    print(msg)
    (out / "summary.json").write_text(json.dumps(sm, indent=1, ensure_ascii=False))
    # topics.yml opens an "alert" issue from this file
    (out / "alerts.txt").write_text(low_supply_text(sm) + ("\n" if sm["low_supply"] else ""))
    if args.dry:  # a dispatch dry run: compute and print, but neither save the ledger nor message anyone
        return
    ledger.save(data)
    alert.send(msg)
    if sm["low_supply"]:
        print(low_supply_text(sm), file=sys.stderr)


def register(sub) -> None:
    s = sub.add_parser("topics-scan")
    s.add_argument("--dry", action="store_true")
    s.set_defaults(fn=scan_cmd)
