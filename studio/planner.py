"""Pick the next topic and its title.

Topics: never a published one (a topic is used once, forever) and never the same series as either of the last
two episodes. Within that, bible/backlog.json is read in order, so the famous topics at its top go first; the
channel's learned series lift can reorder the three front-most series.

Titles follow one formula, famous thing first: "<lead>: <promise> | Documentary for Sleep" (or "| Sleep
Documentary"), where the lead is a plain, searchable subject ("The Last Day of the Dinosaurs", "Megalodon") and
never a period name. title_problems() is the gate every backlog title must pass.
"""
from __future__ import annotations

import re

from . import config, learn, ledger


class BacklogExhausted(RuntimeError):
    pass


def next_topic(data: dict, backlog: list[dict] | None = None, series: dict | None = None,
               override: str | None = None) -> dict:
    backlog = backlog if backlog is not None else config.backlog()
    series = series if series is not None else config.series()
    used = ledger.used_topics(data)
    if override:
        topic = next((t for t in backlog if t["id"] == override), None)
        if topic is None:
            raise ValueError(f"no topic '{override}' in the backlog")
        return topic
    free = [t for t in backlog if t["id"] not in used and t["series"] in series]
    if not free:
        raise BacklogExhausted("every backlog topic has been published; add topics to bible/backlog.json")
    history = [e["series"] for e in data["episodes"]]
    fresh = ([t for t in free if t["series"] not in set(history[-2:])] or
             [t for t in free if t["series"] not in set(history[-1:])] or free)
    order = list(dict.fromkeys(t["series"] for t in fresh))  # series in backlog order of their first open topic
    chosen = learn.prefer(order, data)[0]
    return next(t for t in fresh if t["series"] == chosen)


# --------------------------------------------------------------------------------------------- titles

SUFFIXES = ("| Documentary for Sleep", "| Sleep Documentary")
TITLE_PATTERNS = ("{lead}: {promise} | Documentary for Sleep", "{lead}: {promise} | Sleep Documentary",
                  "{lead} | Documentary for Sleep")
TITLE_LEN = (30, 100)
# Competitor formulas and phrases the channel never uses, and one competitor's exact title.
BANNED = ("boring history for sleep", "(as we understand it)", "selected sources", "fall asleep in",
          "what if dinosaurs never went extinct")
# Period and eon names viewers do not search for. "Jurassic", "Cretaceous" and "Ice Age" are famous and allowed.
JARGON = ("ediacar", "hadean", "archean", "archaean", "proterozoic", "carbonifer", "cambrian", "ordovician",
          "silurian", "devonian", "permian", "triassic", "cryogenian", "precambrian", "paleozoic", "palaeozoic",
          "mesozoic", "cenozoic", "paleocene", "eocene", "oligocene", "miocene", "pliocene", "pleistocene",
          "holocene", "phanerozoic", "neoproterozoic", "paleogene", "neogene")
# The lead must name something people already know and search for.
FAMOUS = ("dinosaur", "asteroid", "ocean", "sea", "deep", "ice", "mammoth", "megalodon", "shark", "pangaea",
          "earth", "insect", "giant", "whale", "squid", "volcano", "extinction", "dying", "snowball", "jurassic",
          "cretaceous", "forest", "reptile", "rhino", "fish", "turtle", "bear", "animal", "life", "reef",
          "continent", "supercontinent", "planet", "lake", "island", "arctic", "antarctica", "flower", "tree",
          "oxygen", "rex", "australia", "america", "atlantic", "mediterranean", "swamp", "land", "grass", "plant",
          "millipede", "scorpion", "dragonfl", "bird", "fossil", "shell", "trench", "floor", "glacier")
# Uncommon words allowed in a lead because they are themselves the famous, searched-for name.
KNOWN = {"megalodon", "pangaea", "rex", "millipede", "supercontinent", "supercontinents", "squid", "woolly",
         "gondwana", "mammoths", "armoured", "dragonflies", "plesiosaurs", "titanosaurs", "trilobites", "ammonites",
         "mosasaurs", "pterosaurs", "dinosaurs"}
RARE_ZIPF = 2.5


def title_problems(title: str, lead: str | None = None) -> list[str]:
    """Why a long-form title breaks the packaging rules (empty when it is fine)."""
    problems = []
    low = title.lower()
    if not TITLE_LEN[0] <= len(title) <= TITLE_LEN[1]:
        problems.append(f"length {len(title)} outside {TITLE_LEN}")
    if not title.endswith(SUFFIXES):
        problems.append("must end with '| Documentary for Sleep' or '| Sleep Documentary'")
    problems += [f"banned phrase '{b}'" for b in BANNED if b in low]
    words = re.findall(r"[a-z]+", low)
    problems += [f"period jargon '{w}'" for w in words if w.startswith(JARGON)]
    lead = (lead if lead is not None else re.split(r"[:|]", title)[0]).strip()
    lead_words = re.findall(r"[a-z]+", lead.lower())
    if not any(w.startswith(FAMOUS) for w in lead_words):
        problems.append(f"lead '{lead}' names nothing famous")
    from .gates import zipf

    rare = [w for w in lead_words if len(w) > 3 and w not in KNOWN and zipf(w) < RARE_ZIPF]
    if rare:
        problems.append(f"lead has uncommon words {rare}")
    return problems


def _words(title: str) -> set[str]:
    return set(re.findall(r"[a-z]+", title.lower())) - {"the", "of", "a", "in", "to", "and", "for", "hours", "hour",
                                                        "documentary", "sleep"}


def similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def title(topic: dict, core: str, hours: int, data: dict, series: dict | None = None) -> str:
    """The formula title from the topic's lead and promise (the writer's title_core stands in for a missing
    lead). Patterns rotate by episode; a title too close to a used one, too long or failing the gate is skipped.
    `hours` and `series` stay for the caller's signature: the length is not in the title."""
    lead = (topic.get("lead") or core).strip()
    promise = (topic.get("promise") or "").strip()
    previous = [e.get("title", "") for e in data["episodes"]]
    done = len(data["episodes"])
    patterns = TITLE_PATTERNS if promise else TITLE_PATTERNS[2:]
    options = []
    for k in range(len(patterns)):
        text = patterns[(done + k) % len(patterns)].format(lead=lead, promise=promise)
        if len(text) <= TITLE_LEN[1]:
            options.append(text)
    good = [t for t in options if not title_problems(t, lead)]
    for t in good + options:
        if all(similar(t, p) < 0.75 for p in previous):
            return t
    return (good or options or [f"{lead[:70]} | Documentary for Sleep"])[0]
