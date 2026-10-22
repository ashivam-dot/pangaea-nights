"""The "Survive the Night" choice-ladder Shorts: the premise banks, the rotation and the editorial gates.

A premise is written and fact-checked ahead of time (bible/ladder/<series>.json): a question, eight options that
climb from cosy to deadly, and a verdict ("Real answer: number N, because ...") that rests on one verbatim sentence
of a pinned Wikipedia revision. Nothing here needs a language model, so a Gemini outage never blocks a Short; an
LLM may only rephrase, and a rephrased line must pass the same gates.

The Shorts gate profile is not the sleep profile: hooks, questions and a call to action are the format, so the calm
style gates (gates.check_style) never apply here. The fact, people and text gates do.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import config, gates, numbers

SERIES = ("survive_night", "level_1_8", "countdown", "how_deep")
BANKS = config.BIBLE / "ladder"
OPTIONS = 8
CTA = "Comment your number."
FIRST_COMMENT = "Which number did you pick? Real answer is in the video."
PROFILE = "short"  # gates profile: facts, people and text only; never the sleep-style checks

TITLE_CHARS = (40, 70)
MAX_EMOJI = 2
MAX_TAGS = 8
LIMITS = {"banner_words": 7, "hook_words": 10, "label_words": 5, "option_vo_words": 9, "verdict_words": 25}
NOVELTY_MAX = 0.35  # 4-gram overlap with any of the last HISTORY scripts
HISTORY = 30
SAME_SERIES_MAX = 2  # in a row
SERIES_PER_WEEK = 3  # distinct series in any 7 days

# A famous token in the first five words of the title: what people already search for.
FAMOUS = ["dinosaur", "t. rex", "t rex", "t-rex", "tyrannosaurus", "asteroid", "ocean", "sea", "ice age", "pangaea",
          "giant bug", "giant insect", "megalodon", "mammoth", "jurassic", "cretaceous", "triassic", "volcano",
          "supervolcano", "prehistoric", "snowball earth", "ichthyosaur", "mosasaur", "sabre-tooth", "saber-tooth",
          "great dying", "extinction", "ammonite", "deep sea", "coal forest"]
# REPORT section 7 (do-not-copy): other channels' names, characters and title strings, and IP we must never touch.
BLOCKLIST = ["MonAI", "el brayan", "insanespidey", "Miranha", "Goats of football", "Lamine Yamal",
             "Risaventuras", "Larín Junior", "Larin Junior", "Bandar Apna Dost", "Boltu", "Halku", "Tuntun mausi",
             "Pagli mausi", "Motu bhai", "Bablu", "Dogesh", "Pouty Frenchie", "Miracle Animal Rescues",
             "Animal Robots", "Nyra", "Riki TV", "Aalu Kachalu", "Baby Baingan", "Mota Hathi", "AI HISTORY",
             "aihistory", "Thirteen Cycles", "the Basilisk", "Niiro Cradle", "WOODNUTS", "Age of Kingdoms",
             "Boring History For Sleep", "As We Understand It", "SELECTED SOURCES", "Gossip Goblin",
             "Epic AI Films", "Chloe VS History", "Silence of the Earth", "Sleepy Science", "Sleepless Historian",
             "Kapwing", "KPop Demon Hunters", "Wednesday", "Mario", "SpongeBob", "Hulk", "Italian brainrot",
             "brainrot", "El Chavo", "Roblox", "Labubu", "Dragon Ball", "Kevin MacLeod", "Moss says"]
WARM = ["warm", "glow", "glowing", "ember", "embers", "lava", "molten", "firelight", "fire", "sunset", "dawn",
        "amber", "lantern-bright", "bioluminescent", "golden"]
HEDGE_SOURCE = re.compile(r"\b(may|might|could|likely|probably|possibly|thought|suggest\w*|estimated|perhaps|"
                          r"hypothes\w*|believed|apparently|unlikely)\b", re.I)
EXTRA_PROMPT_BANS = ["raft", "canoe", "paddle", "rope", "torch", "spear", "footprint of a", "sleeping bag", "backpack",
                     "flag", "house", "cabin", "road", "car ", "phone", "camera", "pov of a"]
STYLE = ("photoreal painterly deep-time scene, tall vertical composition, vivid well-exposed cinematic lighting, "
         "luminous blue-hour twilight sky, one bright warm glowing focal light, clear readable midtones, "
         "cinematic depth, rich fine detail")
# Pictographs, dingbats, misc symbols and arrows (variation selectors are dropped, not counted).
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿←-⇿㊗㊙]")


class BankError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------- banks


def emoji_count(text: str) -> int:
    return len(_EMOJI.findall(text))


def words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", _EMOJI.sub(" ", text).replace("\ufe0f", "")) if re.search(r"\w", w)]


def blocklist_hits(text: str) -> list[str]:
    low = text.lower()
    return [b for b in BLOCKLIST if re.search(r"(?<![a-z0-9])" + re.escape(b.lower()) + r"(?![a-z0-9])", low)]


def famous_in_first_words(title: str, n: int = 5) -> str | None:
    head = " ".join(w.lower().strip("—–-:,.!?\"'") for w in words(title)[:n])
    for tok in FAMOUS:
        if re.search(r"(?<![a-z])" + re.escape(tok) + r"s?(?![a-z])", head):
            return tok
    return None


def title_problems(title: str) -> list[str]:
    out = []
    n = len(title)
    if not TITLE_CHARS[0] <= n <= TITLE_CHARS[1]:
        out.append(f"title is {n} characters (want {TITLE_CHARS[0]}-{TITLE_CHARS[1]})")
    if emoji_count(title) > MAX_EMOJI:
        out.append(f"title has {emoji_count(title)} emoji (at most {MAX_EMOJI})")
    if not famous_in_first_words(title):
        out.append("no famous token (dinosaur, asteroid, ocean, Ice Age, Pangaea, ...) in the first five words")
    out += [f"title hits the blocklist: {b}" for b in blocklist_hits(title)]
    return out


def tag_problems(tags: list[str]) -> list[str]:
    out = []
    if not tags or len(tags) > MAX_TAGS:
        out.append(f"{len(tags or [])} tags (want 1-{MAX_TAGS})")
    for t in tags or []:
        if "#" in t or len(t) > 40:
            out.append(f"tag {t!r} is not a plain short phrase")
        out += [f"tag hits the blocklist: {b}" for b in blocklist_hits(t)]
    return out


def _names(text: str) -> list[str]:
    """Capitalised words that are not the first word of a sentence: the names a fact rests on."""
    out = []
    for sent in gates.sentences(text):
        toks = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", sent)
        for i, tok in enumerate(toks):
            w = re.sub(r"['’]s$", "", tok.strip("'’-"))
            if i and len(w) >= 4 and w[0].isupper() and w.lower() not in gates.ALLOW and w != "Real":
                out.append(w)
    return out


def fact_problems(text: str, source_sentence: str, article: str = "") -> list[str]:
    """Every number in `text` must be in the one source sentence (numbers.py rules), and every name in that sentence
    or in the title of the article it comes from (Wikipedia sentences name their subject as "It ...")."""
    vals = numbers.values(source_sentence)
    src = gates.Sources([source_sentence, article])
    out = [f"number '{q.text}' is not in the source sentence" for q in numbers.unsupported(text, vals)]
    out += [f"name '{w}' is not in the source sentence" for w in _names(text) if not src.has_word(w)]
    return out


def _hedged(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in gates.HEDGES) or bool(HEDGE_SOURCE.search(low))


def prompt_problems(prompt: str) -> list[str]:
    out = gates.check_prompt(prompt)
    low = " " + prompt.lower() + " "
    out += [f"mentions '{w.strip()}'" for w in EXTRA_PROMPT_BANS if re.search(r"\b" + re.escape(w.strip()) + r"\b",
                                                                               low)]
    if not any(re.search(r"\b" + w + r"\b", low) for w in WARM):
        out.append("no warm focal light (glow, ember, lava, firelight, sunset, ...)")
    return out


def premise_problems(p: dict, series: str | None = None) -> list[str]:
    """Everything wrong with one premise (empty = usable)."""
    out = []
    need = ["id", "era", "title", "banner", "hook_vo", "art_direction", "options", "answer_n", "verdict_vo", "fact",
            "description", "tags", "related_topic_hint"]
    missing = [k for k in need if k not in p]
    if missing:
        return [f"missing {', '.join(missing)}"]
    pid = p["id"]
    if series and not re.fullmatch(re.escape(series) + r"-\d{2,3}", pid):
        out.append(f"id {pid!r} is not {series}-<nn>")
    out += title_problems(p["title"])
    out += tag_problems(p["tags"])
    for key, field_ in (("banner_words", "banner"), ("hook_words", "hook_vo"), ("verdict_words", "verdict_vo")):
        n = len(words(p[field_]))
        if n > LIMITS[key] or n == 0:
            out.append(f"{field_} has {n} words (1-{LIMITS[key]})")
    if not p["art_direction"].strip() or len(gates.sentences(p["art_direction"])) != 1:
        out.append("art_direction must be one sentence")
    opts = p["options"]
    if len(opts) != OPTIONS or [o.get("n") for o in opts] != list(range(1, OPTIONS + 1)):
        out.append(f"options must be numbered 1..{OPTIONS}")
    else:
        danger = [o.get("danger") for o in opts]
        if any(not isinstance(d, int) or not 1 <= d <= OPTIONS for d in danger):
            out.append("danger must be an integer 1..8")
        elif any(b < a for a, b in zip(danger, danger[1:])):
            out.append("danger must never go down")
        elif danger[-1] <= danger[0]:
            out.append("danger must climb")
        for o in opts:
            if not 1 <= len(words(o.get("label", ""))) <= LIMITS["label_words"]:
                out.append(f"option {o['n']} label has {len(words(o.get('label', '')))} words (1-5)")
            if not 1 <= len(words(o.get("vo", ""))) <= LIMITS["option_vo_words"]:
                out.append(f"option {o['n']} vo has {len(words(o.get('vo', '')))} words (1-9)")
            out += [f"option {o['n']} image: {x}" for x in prompt_problems(o.get("image_prompt", ""))]
            out += [f"option {o['n']} hits the blocklist: {b}" for b in blocklist_hits(" ".join(
                [o.get("label", ""), o.get("vo", ""), o.get("image_prompt", "")]))]
        labels = [o.get("label", "").lower() for o in opts]
        if len(set(labels)) != len(labels):
            out.append("two options share a label")
    n = p["answer_n"]
    if not isinstance(n, int) or not 1 <= n <= OPTIONS:
        out.append("answer_n must be 1..8")
    if not re.match(rf"Real answer: number {n}\b", p["verdict_vo"]):
        out.append(f"verdict_vo must start 'Real answer: number {n}'")
    if " because" not in p["verdict_vo"].lower() and "," not in p["verdict_vo"]:
        out.append("verdict_vo must give the reason ('..., because ...')")
    fact = p["fact"]
    lacking = [k for k in ("text", "wiki_title", "wiki_revid", "source_sentence") if not fact.get(k)]
    if lacking:
        out.append(f"fact is missing {', '.join(lacking)}")
    else:
        if not isinstance(fact["wiki_revid"], int) or fact["wiki_revid"] <= 0:
            out.append("fact.wiki_revid must be a positive integer")
        src = fact.get("source_sentence", "")
        out += [f"fact: {x}" for x in fact_problems(fact.get("text", ""), src, fact["wiki_title"])]
        out += [f"verdict: {x}" for x in fact_problems(_verdict_claim(p["verdict_vo"]), src, fact["wiki_title"])]
        if HEDGE_SOURCE.search(src) and not _hedged(p["verdict_vo"]):
            out.append("verdict states as certain what the source only suggests: word it as opinion")
    if not 2 <= len(gates.sentences(p["description"])) <= 3:
        out.append("description must be 2-3 sentences")
    out += [f"description hits the blocklist: {b}" for b in blocklist_hits(p["description"])]
    # Everything that ships: the banner is burned into frame 0 (the thumbnail), the lines are spoken.
    for field_ in ("banner", "hook_vo", "verdict_vo", "art_direction"):
        out += [f"{field_} hits the blocklist: {b}" for b in blocklist_hits(p[field_])]
    out += [f"fact hits the blocklist: {b}" for b in blocklist_hits(str(fact.get("text", "")))]
    if len(opts) == OPTIONS and all(isinstance(o, dict) for o in opts):
        # The prompts the image model actually gets: art_direction rides on all nine, and is most of the end plate.
        for job in image_jobs(p):
            out += [f"{job['id']} prompt: {x}" for x in composed_prompt_problems(job["prompt"])]
        if not out:
            from .shorts_ladder import fit_problems

            out += fit_problems(p)
    return out


def composed_prompt_problems(prompt: str) -> list[str]:
    """The people, text and prop bans on a composed image prompt (scene + art direction + house style). The warm-light
    rule is checked on each scene alone: the house style always names a warm focal light."""
    low = " " + prompt.lower() + " "
    out = gates.check_prompt(prompt)
    return out + [f"mentions '{w.strip()}'" for w in EXTRA_PROMPT_BANS
                  if re.search(r"\b" + re.escape(w.strip()) + r"\b", low)]


def _verdict_claim(verdict: str) -> str:
    """The verdict without its fixed 'Real answer: number N' lead, which is not a sourced number."""
    return re.sub(r"^Real answer: number \d+\s*[,.:;]?\s*", "", verdict)


def load_bank(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"series", "premises"} or not isinstance(data["premises"], list):
        raise BankError(f"{path.name}: want exactly {{series, premises}}")
    return data


def bank_problems(bank: dict) -> dict[str, list[str]]:
    """{premise id: problems} for every bad premise in one bank (empty = the bank is good)."""
    series = bank["series"]
    out: dict[str, list[str]] = {}
    if series not in SERIES:
        out["<bank>"] = [f"unknown series {series!r}"]
    seen = Counter(p.get("id") for p in bank["premises"])
    for p in bank["premises"]:
        probs = premise_problems(p, series)
        if seen[p.get("id")] > 1:
            probs.append("duplicate id")
        if probs:
            out[str(p.get("id"))] = probs
    return out


def load_banks(root: Path = BANKS, strict: bool = True) -> dict[str, list[dict]]:
    """{series: [premise, ...]} with the series attached to each premise. strict: any bad premise is an error."""
    out: dict[str, list[dict]] = {}
    errors = []
    for path in sorted(root.glob("*.json")):
        bank = load_bank(path)
        probs = bank_problems(bank)
        errors += [f"{path.name} {pid}: {'; '.join(p)}" for pid, p in probs.items()]
        good = [p for p in bank["premises"] if str(p.get("id")) not in probs]
        out.setdefault(bank["series"], []).extend({**p, "series": bank["series"]} for p in good)
    if strict and errors:
        raise BankError("\n".join(errors))
    return out


def find(banks: dict[str, list[dict]], premise_id: str) -> dict | None:
    return next((p for ps in banks.values() for p in ps if p["id"] == premise_id), None)


# ---------------------------------------------------------------------------------------------------- script


def segments(p: dict) -> list[dict]:
    """What the narrator says, in order: hook, eight options, verdict, call to comment."""
    out = [{"kind": "hook", "n": 0, "text": p["hook_vo"]}]
    out += [{"kind": "option", "n": o["n"], "text": o["vo"]} for o in p["options"]]
    out += [{"kind": "verdict", "n": p["answer_n"], "text": p["verdict_vo"]}, {"kind": "cta", "n": 0, "text": CTA}]
    return out


def script_text(p: dict) -> str:
    return " ".join(s["text"] for s in segments(p))


def _novel_text(script: str) -> str:
    """The script without the lines every Short shares (the verdict lead and the call to comment)."""
    s = script.replace(CTA, " ")
    return re.sub(r"Real answer: number \d+,?( because)?", " ", s)


def ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9']+", _novel_text(text).lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def overlap(script: str, others: list[str]) -> float:
    """Largest share of this script's 4-grams found in any one earlier script."""
    mine = ngrams(script)
    if not mine:
        return 0.0
    return max((len(mine & ngrams(o)) / len(mine) for o in others), default=0.0)


def recent_scripts(data: dict, n: int = HISTORY) -> list[str]:
    return [s["script"] for s in data.get("shorts", [])[-n:] if s.get("script")]


# ---------------------------------------------------------------------------------------------------- rotation


def _day(text: str) -> date:
    return datetime.fromisoformat(text.replace(".000Z", "Z").replace("Z", "+00:00")).date()


def history(data: dict) -> list[dict]:
    """Published (scheduled) Shorts, oldest first, each with a release day."""
    rows = [s for s in data.get("shorts", []) if s.get("video_id") and s.get("publish_at")]
    return sorted(rows, key=lambda s: s["publish_at"])


def used_premises(data: dict) -> set[str]:
    return {s["premise"] for s in data.get("shorts", []) if s.get("video_id")}


def rotation_problems(series_by_slot: list[tuple[date, str]], candidate: str, at: date) -> list[str]:
    """series_by_slot: (release day, series) of earlier Shorts, oldest first."""
    out = []
    seq = [s for _, s in series_by_slot]
    if len(seq) >= SAME_SERIES_MAX and all(s == candidate for s in seq[-SAME_SERIES_MAX:]):
        out.append(f"{candidate} would run {SAME_SERIES_MAX + 1} times in a row")
    window = [s for d, s in series_by_slot if at - timedelta(days=6) <= d <= at] + [candidate]
    if len(window) >= SERIES_PER_WEEK and len(set(window)) < min(SERIES_PER_WEEK, len(window)):
        out.append(f"only {len(set(window))} series in the last 7 days (want {SERIES_PER_WEEK})")
    return out


def weight(data: dict, series: str, era: str) -> float:
    return float((data.get("ladder") or {}).get("weights", {}).get(f"{series}|{era}", 1.0))


FAILED_MAX = 2  # publish builds of one premise that made nothing, within FAILED_DAYS: then it rests
FAILED_DAYS = 14


def demoted(data: dict, at: date) -> dict[str, int]:
    """{premise: failed publish builds in the last FAILED_DAYS} for every premise at or over FAILED_MAX. A premise
    that keeps failing a gate that does not change from day to day would otherwise be picked again every day (its
    series stays the longest rested) and hold every slot; it rests until a human fixes it or the window passes."""
    count: Counter = Counter()
    for b in builds(data):
        if b.get("mode") != "publish" or b.get("published") or not b.get("premise"):
            continue
        try:
            day = _day(b["at"]) if b.get("at") else at
        except ValueError:
            day = at
        if at - timedelta(days=FAILED_DAYS) < day <= at:
            count[b["premise"]] += 1
    return {k: n for k, n in count.items() if n >= FAILED_MAX}


def recent_alerts(data: dict, kind: str, at: date, days: int = FAILED_DAYS) -> list[dict]:
    out = []
    for a in (data.get("ladder") or {}).get("alerts", []):
        try:
            if a.get("kind") == kind and at - timedelta(days=days) < _day(a.get("at") or "") <= at:
                out.append(a)
        except ValueError:
            continue
    return out


def pick(banks: dict[str, list[dict]], data: dict, at: date, planned: list[dict] | None = None,
         override: str | None = None, exclude: set[str] | None = None) -> dict:
    """The next premise: a queued pt.2 first, then the best-weighted series that has rested longest, within the
    rotation rules; never a premise already published, one that keeps failing (demoted) or one in exclude.
    planned = premises already chosen for earlier slots today. An override is taken as given."""
    planned = planned or []
    taken = used_premises(data) | {p["id"] for p in planned}
    if override:
        p = find(banks, override)
        if not p:
            raise BankError(f"no usable premise {override!r}")
        if p["id"] in taken:
            raise BankError(f"premise {override} was already published")
        return p
    taken |= set(demoted(data, at)) | set(exclude or ())
    past = [(_day(s["publish_at"]), s["series"]) for s in history(data)] + [(at, p["series"]) for p in planned]
    last_used = {s: i for i, (_, s) in enumerate(past)}
    queue = [q for q in (data.get("ladder") or {}).get("queue", []) if q not in taken]
    for qid in queue:
        p = find(banks, qid)
        if p and not rotation_problems(past, p["series"], at):
            return p
    choices = []
    for series in sorted(banks):
        premises = banks[series]
        free = [p for p in premises if p["id"] not in taken]
        if not free or rotation_problems(past, series, at):
            continue
        best = max(free, key=lambda p: (weight(data, series, p["era"]), -premises.index(p)))
        choices.append((last_used.get(series, -1), weight(data, series, best["era"]), best))
    if not choices:  # rotation cannot be met with what is left: say so rather than break the rule
        raise BankError("no premise satisfies the rotation rules: the banks need more premises")
    # Among the two series that have rested longest, the better learned weight leads (ties keep rest order).
    rested = sorted(choices, key=lambda c: c[0])[:2]
    return max(rested, key=lambda c: (c[1], -c[0]))[2]


def plan_problems(p: dict, data: dict, at: date, planned: list[dict] | None = None) -> dict:
    """The editorial gates for one chosen premise, as {gate: {value, ok, ...}}."""
    script = script_text(p)
    past = [(_day(s["publish_at"]), s["series"]) for s in history(data)]
    past += [(at, q["series"]) for q in planned or []]
    others = recent_scripts(data) + [script_text(q) for q in planned or []]
    rot = rotation_problems(past, p["series"], at)
    nov = overlap(script, others)
    prem = premise_problems(p, p["series"])
    return {"premise": {"ok": not prem, "problems": prem},
            "novelty_4gram": {"value": round(nov, 3), "ceiling": NOVELTY_MAX, "ok": nov <= NOVELTY_MAX},
            "rotation": {"ok": not rot, "problems": rot},
            "reuse": {"ok": p["id"] not in used_premises(data)}}


IMAGE_IDS = [f"o{n}" for n in range(1, OPTIONS + 1)] + ["end"]
# The CPU model draws prehistoric flyers as feathered birds (dry run 3: "a giant Quetzalcoatlus" came out an eagle),
# which a deep-time audience notices at once. Prompts naming one get the anatomy spelled out.
ANATOMY = {("pterosaur", "quetzalcoatlus", "pteranodon", "nyctosaurus", "rhamphorhynchus"):
           "the pterosaur has bare leathery membrane wings stretched from one long finger, a long beak, no feathers, "
           "not a bird"}


def anatomy(prompt: str) -> str:
    low = prompt.lower()
    return "".join(f" {text}." for names, text in ANATOMY.items() if any(n in low for n in names))


def image_jobs(p: dict) -> list[dict]:
    """Nine vertical stills: the eight options (option 1 is also frame 0, the thumbnail) and the end-card plate."""
    art = p["art_direction"].strip().rstrip(".")
    jobs = [{"id": f"o{o['n']}", "n": o["n"], "scene": o["image_prompt"],
             "prompt": f"{o['image_prompt'].strip().rstrip('.')}.{anatomy(o['image_prompt'])} {art}. {STYLE}.",
             "luma": (92.0, 168.0) if o["n"] == 1 else (62.0, 198.0)} for o in p["options"]]
    end = (f"A wide quiet view of the same world at the end of the night, the horizon beginning to glow warm, open "
           f"calm sky in the upper half of the frame. {art}. {STYLE}.")
    jobs.append({"id": "end", "n": 0, "scene": end, "prompt": end, "luma": (62.0, 198.0)})
    return jobs


def now_day() -> date:
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------------------------------- cadence

# Slot A goes out at 14:30 UTC the day it is built; slot B at 00:30 UTC the next morning.
SLOTS = {"A": (14, 30), "B": (0, 30)}
RAMP_DAYS = 7  # launch days with slot A only
RAMP_CLEAN = 7  # slot B needs this many consecutive gate-clean Shorts before it


def launch_day(text: str | None) -> date | None:
    try:
        return date.fromisoformat((text or "").strip())
    except ValueError:
        return None


def builds(data: dict) -> list[dict]:
    return (data.get("ladder") or {}).get("builds", [])


def slots_allowed(launch: date | None, today: date, past_builds: list[dict]) -> tuple[list[str], str]:
    """Which slots may publish today, and why. Days 1-7 after SHORTS_LAUNCH: slot A only. From day 8, slot B too,
    but only while the previous seven publish builds all passed every gate."""
    if launch is None:
        return [], "SHORTS_LAUNCH is not set (YYYY-MM-DD): no Shorts publish before a launch date"
    day = (today - launch).days + 1
    if day < 1:
        return [], f"Shorts launch on {launch.isoformat()}"
    if day <= RAMP_DAYS:
        return ["A"], f"launch day {day}: slot A only for the first {RAMP_DAYS} days"
    recent = [b for b in past_builds if b.get("mode") == "publish"][-RAMP_CLEAN:]
    if len(recent) < RAMP_CLEAN or not all(b.get("ok") for b in recent):
        bad = sum(1 for b in recent if not b.get("ok"))
        return ["A"], (f"slot B waits for {RAMP_CLEAN} gate-clean Shorts in a row "
                       f"(have {len(recent)}, {bad} failed)")
    return ["A", "B"], f"launch day {day}: both slots"


def stale(data: dict, launch: date | None, now: datetime, hours: float) -> str | None:
    """Why the Shorts look stalled (None = fine): no Short uploaded for `hours` once the launch day has begun. The
    long-form health check reads last_upload, which a Short never touches, so this is the Shorts' own check."""
    if launch is None or now.date() < launch:
        return None
    ups = [s["uploaded_at"] for s in data.get("shorts", []) if s.get("video_id") and s.get("uploaded_at")]
    since = datetime(launch.year, launch.month, launch.day, tzinfo=timezone.utc)
    if ups:
        since = max(since, max(datetime.fromisoformat(u.replace("Z", "+00:00")) for u in ups))
    gap = (now - since).total_seconds() / 3600
    if gap <= hours:
        return None
    return (f"no Short uploaded for {gap:.0f} hours (limit {hours:g})" if ups else
            f"no Short uploaded since the launch on {launch.isoformat()} ({gap:.0f} hours)")
