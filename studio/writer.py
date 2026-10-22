"""The episode script: an outline, then each chapter written, gated and revised against the Wikipedia sources.

`write()` uses Gemini and can produce a publishable script. `write_stub()` arranges real Wikipedia sentences into
chapters so the image, voice and render stages can be proven without a key; its output is marked unpublishable
and the upload guard refuses it.
"""
from __future__ import annotations

import json
import re

from . import config, gates
from .llm import Gemini

STYLE = ("painterly photorealistic nature documentary still, cinematic composition, natural soft light, "
         "rich fine detail, calm and quiet mood")
# The thumbnail overrides the episode's dark sleep palette: one bright, sharply lit focal creature (thumbnail.py
# gates its mean luminance between a floor and a ceiling).
THUMB_LOOK = ("but for this frame bright daylight or golden-hour light, one single focal creature large and right of "
              "centre, sharply lit against a brighter sky or sunlit water, simple open uncluttered space on the left")

SYSTEM = """You write Pangaea Nights, a calm English sleep-documentary channel about Earth's deep past and its
ancient and deep oceans. Listeners play an episode at bedtime and drift off while it plays.

Binding rules:
- Ground every statement in the SOURCES given. Every number you write (dates, sizes, ages, counts, percentages)
  must appear in the sources exactly; if a figure is not in the sources, describe it without a number. Write
  numbers as digits with scale words, for example "66 million years ago" or "about 2 metres".
- Name only organisms, places and formations that appear in the sources. Never invent a species, genus or name.
- Hedge uncertain science honestly: "scientists think", "may have", "probably", "the evidence suggests".
- No people. Do not narrate human history, named scientists, museums, discoveries or modern human impacts.
  Stay with the animals, plants, seas, rocks and climates themselves. No mythology. Nothing about outer space
  beyond the Earth and its Sun.
- Sleep-friendly voice: slow, warm, sensory, present tense where natural; gentle second-person invitations
  ("imagine", "you might notice", "let your breathing slow"). Long flowing sentences mixed with short quiet ones.
- No exclamation marks, no cliffhangers, no suspense, no alarming or violent words. Predators and extinctions are
  described calmly and briefly, never graphically. At most two gentle questions per chapter.
- Gentle transitions between chapters; never tease what comes next.
- Plain prose only: no headings, lists, markdown or stage directions.
"""

OUTLINE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title_core": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "art_direction": {"type": "STRING"},
        "chapters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "title": {"type": "STRING"}, "focus": {"type": "STRING"},
            "facts": {"type": "ARRAY", "items": {"type": "STRING"}}},
            "required": ["title", "focus", "facts"]}},
        "thumbnail_words": {"type": "STRING"},
        "thumbnail_prompt": {"type": "STRING"},
        "tail_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["title_core", "summary", "art_direction", "chapters", "thumbnail_words", "thumbnail_prompt",
                 "tail_scenes", "tags"],
}
CHAPTER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["paragraphs", "scenes"],
}
TRAILER_SCHEMA = {
    "type": "OBJECT",
    "properties": {"hook": {"type": "STRING"}, "narration": {"type": "STRING"},
                   "scenes": {"type": "ARRAY", "items": {"type": "STRING"}}},
    "required": ["hook", "narration", "scenes"],
}
SCENE_RULES = """Image prompts: each describes one still image as a nature-documentary frame: the setting, the light,
the creatures or landforms, with their anatomy described concretely from the sources (body shape, limbs, shell,
colour only if known, size relative to surroundings). Mix wide landscapes or seascapes, medium shots of animals,
and close-ups of textures (sand, bark, shell, water, ice, rock). 35 to 80 words each. Never include people, hands,
boats, buildings, text, signs, labels, maps or diagrams. Never compare a creature to something else ("leaf-like",
"resembling a fern"): the image model draws the thing compared. Describe its real shape instead ("a flat oval
frond of fine parallel ridges"). Do not mention the art style; it is added separately."""


def sources_block(docs: list[dict]) -> str:
    return "\n\n".join(f"### SOURCE: {d['title']}\n{d['text']}" for d in docs)


def scene_count(shape: config.Shape) -> int:
    lo, hi = shape.scenes_per_chapter
    return max(lo, min(hi, round(shape.words_per_chapter / 105)))


def _clean_paragraphs(raw: list[str]) -> list[str]:
    out = []
    for p in raw:
        p = re.sub(r"\s+", " ", str(p)).strip()
        p = p.replace("!", ".").replace("..", ".")
        if p and not re.match(r"^(chapter|part)\s+\w+[:.]?$", p, re.I):
            out.append(p)
    return out


def _fix_prompts(llm: Gemini, prompts: list[str], context: str) -> list[str]:
    bad = {i: gates.check_prompt(p) for i, p in enumerate(prompts)}
    bad = {i: v for i, v in bad.items() if v}
    if not bad:
        return prompts
    ask = "\n".join(f"{i}: {prompts[i]}  -- problems: {', '.join(v)}" for i, v in bad.items())
    try:
        fixed = llm.generate(SYSTEM, f"{SCENE_RULES}\n\nContext: {context}\n\nRewrite each numbered image prompt to "
                                     f"fix its problems, keeping the scene. Return JSON "
                                     f"{{\"scenes\": [...]}} in the same order.\n\n{ask}",
                             {"type": "OBJECT", "properties": {"scenes": {"type": "ARRAY", "items": {"type": "STRING"}}},
                              "required": ["scenes"]}, temperature=0.5, max_tokens=4096)["scenes"]
        for (i, _), new in zip(bad.items(), fixed):
            if not gates.check_prompt(new):
                prompts[i] = new
    except Exception as e:  # noqa: BLE001 - a failed fix falls back to dropping the prompt
        print(f"writer: prompt fix failed ({e})", flush=True)
    return [p for p in prompts if not gates.check_prompt(p)]


def _extend(llm, brief: str, title: str, paragraphs: list[str], scenes: list, target: int, n_scenes: int,
            closing: bool, block: str) -> tuple[list[str], list]:
    """Models reliably write ~550 words per call: continue a short draft instead of rewriting it from scratch."""
    for _ in range(2):
        words = sum(len(p.split()) for p in paragraphs)
        missing = target - words
        if missing < 0.2 * target:
            break
        end = ("Keep winding down and finish with a peaceful image and a quiet goodnight." if closing else
               "Do not end the evening; stop at a natural pause.")
        ask = f"""{brief}
Here is the beginning of chapter "{title}" ({words} words). CONTINUE it seamlessly from its last sentence with
about {missing} more words in {max(2, round(missing / 120))} to 8 new paragraphs (60 to 170 words each). Cover facts
from the sources that the draft has not used yet, never repeat what it already said, keep the same calm second-person
voice and frame uncertain science as uncertain. {end} Also write {max(1, n_scenes - len(scenes))} image prompts for
the new paragraphs, in order.
{SCENE_RULES}

DRAFT SO FAR:
{json.dumps(paragraphs)}

SOURCES:
{block}"""
        out = llm.generate(SYSTEM, ask, CHAPTER_SCHEMA, temperature=0.7)
        more = _clean_paragraphs(out.get("paragraphs", []))
        if not more:
            break
        paragraphs = paragraphs + more
        scenes = list(scenes) + list(out.get("scenes", []))
        print(f"writer: extended \"{title}\" by {sum(len(p.split()) for p in more)} words", flush=True)
    return paragraphs, scenes


def write(topic: dict, docs: list[dict], shape: config.Shape, llm: Gemini | None = None) -> dict:
    llm = llm or Gemini()
    series = config.series()[topic["series"]]
    src = gates.Sources([d["text"] for d in docs])
    block = sources_block(docs)
    n_scenes = scene_count(shape)
    brief = (f"EPISODE: {topic['title']}\nSERIES: {series['name']} ({series['era']}). {series['promise']}\n"
             f"ANGLE: {topic['angle']}\nSERIES PALETTE: {series['palette']}\nSERIES CAMERA: {series['camera']}")
    outline = llm.generate(SYSTEM, f"""{brief}

Plan an episode of about {shape.chapters * shape.words_per_chapter} spoken words in exactly {shape.chapters}
chapters that flow like one slow evening journey through this subject, each chapter about one aspect, in a sensible
order (setting, life, daily rhythms, change over time, a gentle closing). Return JSON with:
- title_core: the episode's subject in 2 to 6 plain words for the title card (the famous thing first, for example
  "The Last Day of the Dinosaurs"; no period names such as Ediacaran, Hadean or Carboniferous; say "before the
  dinosaurs", "coal forests" or "Earth's first oceans" instead).
- summary: 2 to 3 calm sentences for the video description, no hype.
- art_direction: one sentence fixing this episode's visual palette, light and lens, consistent with the series
  palette, dark and calm enough for sleep.
- chapters: for each, a calm 2 to 6 word title, its focus, and 5 to 8 specific facts from the sources it will use.
- thumbnail_words: 2 to 4 short, bold, plain words naming the famous subject, for example "LAST DAY OF DINOSAURS"
  or "MEGALODON" plus one word (no numbers, no question, no period names).
- thumbnail_prompt: one image prompt (rules below) for a bright, clearly lit thumbnail: ONE focal creature, large
  and right of centre, sharply lit against a brighter sky or sunlit water, with simple open uncluttered space on
  the left for words. Daylight or golden hour, not night.
- tail_scenes: {shape.tail_scenes} image prompts for a wordless ending: ever darker, emptier, peaceful views of
  this world at night.
- tags: 10 to 15 YouTube search tags.
{SCENE_RULES}

SOURCES:
{block}""", OUTLINE_SCHEMA, temperature=0.7)
    chapters_plan = outline["chapters"][: shape.chapters]
    if len(chapters_plan) < shape.chapters:
        raise RuntimeError(f"outline has {len(chapters_plan)} chapters, need {shape.chapters}")
    chapters, log = [], []
    previous = ""
    for i, plan in enumerate(chapters_plan):
        opening = i == 0
        closing = i == len(chapters_plan) - 1
        role = ("This is the OPENING chapter: begin with a soft welcome to Pangaea Nights and an invitation to get "
                "comfortable and let the breath slow; you may include one quiet line inviting listeners to subscribe "
                "if they enjoy these slow journeys, nothing more. Then ease into the subject." if opening else
                "This is the FINAL chapter: wind down even more slowly, end with a peaceful image and a quiet "
                "goodnight, no summary of lessons." if closing else
                f"Begin with a gentle transition from where the last chapter left off: \"{previous[-400:]}\"")
        ask = f"""{brief}
OUTLINE: {json.dumps([c['title'] for c in chapters_plan])}
NOW WRITE chapter {i + 1} of {len(chapters_plan)}: "{plan['title']}". Focus: {plan['focus']}
Facts to draw on: {json.dumps(plan['facts'])}
{role}
Write about {shape.words_per_chapter} words in 6 to 10 paragraphs of narration (paragraphs of 60 to 170 words).
Use at least {shape.hedges_per_chapter + 1} hedges where the science is uncertain, and invite the listener in the
second person several times. Also write exactly {n_scenes} image prompts that follow the narration in order.
{SCENE_RULES}

SOURCES:
{block}"""
        best = None
        for attempt in range(3):
            out = llm.generate(SYSTEM, ask, CHAPTER_SCHEMA, temperature=0.85 if attempt == 0 else 0.6)
            paragraphs = _clean_paragraphs(out.get("paragraphs", []))
            report = gates.chapter_report(paragraphs, src, opening, shape.hedges_per_chapter,
                                          shape.words_per_chapter)
            if any(p.startswith("too short") for p in report.problems):
                paragraphs, more_scenes = _extend(llm, brief, plan["title"], paragraphs, out.get("scenes", []),
                                                  shape.words_per_chapter, n_scenes, closing, block)
                out = {**out, "scenes": more_scenes}
                report = gates.chapter_report(paragraphs, src, opening, shape.hedges_per_chapter,
                                              shape.words_per_chapter)
            log.append({"chapter": i + 1, "attempt": attempt + 1, "problems": report.problems})
            print(f"writer: chapter {i + 1} attempt {attempt + 1}: {sum(len(p.split()) for p in paragraphs)} words, "
                  f"{len(report.problems)} problems {report.problems[:4]}", flush=True)
            if best is None or len(report.problems) < len(best[1].problems):
                best = (paragraphs, report, out.get("scenes", []))
            if report.ok:
                break
            ask = f"""{brief}
Here is a draft of chapter {i + 1}, "{plan['title']}". It fails these editorial checks:
{json.dumps(report.problems[:40])}
Sentences that must change (fix the number or name using the sources, or rewrite without it):
{json.dumps(report.bad_sentences[:40])}
Rewrite the whole chapter fixing every problem while keeping its calm voice, its length (about
{shape.words_per_chapter} words) and its {n_scenes} image prompts. {role}
{SCENE_RULES}

DRAFT:
{json.dumps({'paragraphs': paragraphs, 'scenes': out.get('scenes', [])})}

SOURCES:
{block}"""
        paragraphs, report, scenes = best
        if not report.ok:
            paragraphs = gates.drop_sentences(paragraphs, report.bad_sentences)
            report = gates.chapter_report(paragraphs, src, opening, shape.hedges_per_chapter,
                                          shape.words_per_chapter)
        hard = [p for p in report.problems if p.startswith(("unsupported", "name not", "exclamation", "alarm",
                                                            "call to", "markdown", "more than one"))]
        words = sum(len(p.split()) for p in paragraphs)
        if hard or words < 0.65 * shape.words_per_chapter:
            raise RuntimeError(f"chapter {i + 1} could not pass the gates: {(hard or report.problems)[:5]}")
        scenes = _fix_prompts(llm, [str(s).strip() for s in scenes], plan["title"])
        if len(scenes) < shape.scenes_per_chapter[0]:
            raise RuntimeError(f"chapter {i + 1} has only {len(scenes)} usable image prompts")
        chapters.append({"n": i + 1, "title": plan["title"].strip().rstrip("."), "paragraphs": paragraphs,
                         "scenes": scenes[: shape.scenes_per_chapter[1]], "warnings": report.problems})
        previous = paragraphs[-1]
    trailers = []
    for k in range(config.trailer_count(shape)):  # none while config.TRAILER_SHORTS is off
        chapter = chapters[min(len(chapters) - 1, k * 3 + 1)]
        t_ask = f"""{brief}
Write vertical Short trailer {k + 1} for this episode, drawn from the chapter "{chapter['title']}":
{json.dumps(chapter['paragraphs'][:4])}
- hook: 3 to 8 calm, intriguing words shown on screen (no question mark, no exclamation, no number).
- narration: 70 to 100 words, calm and wondering, grounded in the same facts, ending with the line
  "The full journey is on Pangaea Nights."
- scenes: exactly {shape.trailer_scenes} image prompts, composed for a tall vertical frame, the most beautiful views
  of this subject.
{SCENE_RULES}"""
        for _ in range(3):
            tr = llm.generate(SYSTEM, t_ask, TRAILER_SCHEMA, temperature=0.8)
            rep = gates.Report()
            gates.check_facts(tr["narration"] + " " + tr["hook"], src, rep)
            style = gates.Report()
            gates.check_style([tr["narration"]], style, False, 0, 85)
            for p in style.problems:
                if p.startswith(("exclamation", "alarm", "call to")):
                    rep.add(p)
            if "!" in tr["hook"]:
                rep.add("exclamation in hook")
            if rep.ok:
                break
        else:
            raise RuntimeError(f"trailer {k + 1} failed the fact gate: {rep.problems[:5]}")
        scenes = _fix_prompts(llm, [str(s) for s in tr["scenes"]], chapter["title"])
        trailers.append({"hook": tr["hook"].strip().rstrip("."), "narration": tr["narration"].strip(),
                         "scenes": scenes[: shape.trailer_scenes], "chapter": chapter["n"]})
    thumb = _fix_prompts(llm, [outline["thumbnail_prompt"]], "thumbnail") or [chapters[0]["scenes"][0]]
    tail = _fix_prompts(llm, [str(s) for s in outline["tail_scenes"]], "ending") or chapters[-1]["scenes"][-2:]
    return {
        "writer": "gemini", "models": [m.split("/")[-1] for m in llm.used], "publishable": True,
        "title_core": outline["title_core"].strip().rstrip("."), "summary": outline["summary"].strip(),
        "art_direction": outline["art_direction"].strip(), "chapters": chapters, "trailers": trailers,
        "thumbnail": {"words": outline["thumbnail_words"].strip().rstrip("."), "prompt": thumb[0]},
        "tail_scenes": tail[: shape.tail_scenes], "tags": [t.strip() for t in outline["tags"]][:15],
        "writer_log": log,
    }


# --------------------------------------------------------------------------------------------- dry-run stub

_LINKS = ["Let your breathing slow as we stay here a while.", "Imagine the stillness around you.",
          "There is no hurry in this place.", "You might picture it gently, without effort.",
          "Rest your thoughts on this for a moment."]


def _source_sentences(doc: dict) -> list[str]:
    out = []
    for para in doc["text"].split("\n"):
        if para.startswith("##") or len(para) < 80:
            continue
        for s in gates.sentences(para):
            s = re.sub(r"\s*\([^)]*\)", "", s).strip()
            if 8 <= len(s.split()) <= 45 and not re.search(r"[\[\]{}=|]|\bet al\b|\bhuman|\bpeople\b", s, re.I):
                out.append(s)
    return out


def write_stub(topic: dict, docs: list[dict], shape: config.Shape) -> dict:
    """Unpublishable stand-in script made of Wikipedia sentences, used only to prove the media stages."""
    series = config.series()[topic["series"]]
    pools = [_source_sentences(d) for d in docs]
    n_scenes = scene_count(shape)
    chapters = []
    cursor = [0] * len(pools)
    for i in range(shape.chapters):
        k = i % len(pools)
        paragraphs, words = [], 0
        if i == 0:
            paragraphs.append("Welcome to Pangaea Nights. Settle in, let your shoulders soften, and let your "
                              f"breathing slow. Tonight we drift gently into {topic['title'].lower()}.")
        while words < shape.words_per_chapter:
            pool = pools[k] if cursor[k] < len(pools[k]) else max(pools, key=len)
            idx = cursor[k] % max(1, len(pool))
            chunk = pool[idx: idx + 4] or ["The water is calm, and the light is soft."]
            cursor[k] += 4
            para = " ".join(chunk) + " " + _LINKS[(i + len(paragraphs)) % len(_LINKS)]
            paragraphs.append(para)
            words += len(para.split())
            if cursor[k] > 4 * len(pool) + 40:
                break
        d = docs[k]
        lead = re.sub(r"\s*\([^)]*\)", "", d["text"].split("\n")[0])[:420]
        lead = re.sub(r"\b(humans?|people|museum|scientists?)\b", "", lead, flags=re.I)
        scenes = []
        views = ["A wide calm view of the world of", "A quiet medium shot showing", "A gentle close-up texture study of",
                 "An evening panorama of the habitat of", "A soft underwater or ground-level view of",
                 "A misty distant view of", "A detailed natural portrait of", "A still twilight scene of",
                 "A serene overhead view of", "A peaceful night view of"]
        for j in range(n_scenes):
            scenes.append(f"{views[j % len(views)]} {d['title']}, in deep prehistoric time. {lead} "
                          f"Natural setting, {series['camera']}.")
        chapters.append({"n": i + 1, "title": d["title"], "paragraphs": paragraphs, "scenes": scenes,
                         "warnings": ["stub"]})
    trailers = [{"hook": topic["title"], "narration": " ".join(chapters[0]["paragraphs"][1].split()[:80]) +
                 " The full journey is on Pangaea Nights.",
                 "scenes": [c for c in chapters[0]["scenes"][: shape.trailer_scenes]], "chapter": 1}
                for _ in range(config.trailer_count(shape))]
    lead = topic.get("lead", topic["title"])
    return {
        "writer": "stub", "models": [], "publishable": False,
        "title_core": lead, "summary": topic["angle"], "art_direction": series["palette"],
        "chapters": chapters, "trailers": trailers,
        "thumbnail": {"words": " ".join(lead.split()[:4]), "prompt": chapters[0]["scenes"][0]},
        "tail_scenes": [f"A dark, peaceful, empty night view of {topic['title']}, faint moonlight on still water "
                        f"or land, deep in prehistoric time, nothing moving, {series['palette']}."
                        for _ in range(shape.tail_scenes)],
        "tags": [topic["title"], series["name"], "sleep documentary", "deep time"], "writer_log": [],
    }


def image_prompt(scene: str, art_direction: str) -> str:
    return f"{scene.strip().rstrip('.')}. {art_direction.strip().rstrip('.')}. {STYLE}."
