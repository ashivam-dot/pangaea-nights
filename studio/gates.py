"""Editorial gates every chapter must pass before it can be narrated, and that a publishable episode must pass.

- Facts: every quantity in the narration appears in the episode's sources.
- Names: every taxon or proper name (any rare or capitalised word) appears in the sources or the allowlist.
- Hedging: uncertain science is framed as such ('scientists think', 'may have', ...).
- Sleep style: no exclamation, no cliffhangers or alarm words, no calls to action except one quiet line in the
  opening, second person invitations, calm paragraph lengths.
- Image prompts: no people, no text, enough description.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import numbers

GEOLOGY = """Hadean Archean Archaean Eoarchean Paleoarchean Mesoarchean Neoarchean Proterozoic Paleoproterozoic
Mesoproterozoic Neoproterozoic Siderian Rhyacian Orosirian Statherian Calymmian Ectasian Stenian Tonian Cryogenian
Ediacaran Phanerozoic Paleozoic Palaeozoic Cambrian Ordovician Silurian Devonian Carboniferous Mississippian
Pennsylvanian Permian Mesozoic Triassic Jurassic Cretaceous Cenozoic Paleogene Palaeogene Neogene Quaternary
Paleocene Eocene Oligocene Miocene Pliocene Pleistocene Holocene Precambrian Cisuralian Guadalupian Lopingian
Earth Sun Moon Pangaea Pangea Gondwana Laurasia Panthalassa Tethys Rodinia Africa Antarctica Asia Australia Europe
America Americas Arctic Antarctic Atlantic Pacific Indian Ocean Southern North South East West Celsius Fahrenheit
January February March April May June July August September October November December Monday Tuesday Wednesday
Thursday Friday Saturday Sunday Pangaea Nights I""".split()
HEDGES = ["scientists think", "scientists believe", "researchers think", "researchers believe", "may have",
          "might have", "probably", "likely", "perhaps", "it seems", "seems to", "evidence suggests",
          "suggests", "is thought", "are thought", "was thought", "were thought", "is believed", "are believed",
          "possibly", "may ", "might ", "could have", "some scientists", "paleontologists think",
          "palaeontologists think", "geologists think", "it is possible", "not yet certain", "debated",
          "uncertain", "estimated", "may well", "we cannot be sure", "no one knows", "remains unclear"]
ALARM = ["terrifying", "horrifying", "horrific", "brutal", "gruesome", "shocking", "nightmare", "nightmarish",
         "deadliest", "monster", "monstrous", "massacre", "bloodthirsty", "scream", "screaming", "violent",
         "violently", "savage", "you won't believe", "insane", "mind-blowing", "mind blowing", "epic",
         "stay tuned", "but wait", "what happened next", "find out", "smash", "obliterat", "annihilat", "doom",
         "apocalyp", "horror", "panic", "jump scare", "unbelievable", "incredible"]
CALLS = ["subscribe", "like this video", "hit the like", "notification bell", "comment below", "in the comments",
         "smash that", "patreon", "sponsor"]
HUMAN_WORDS = ["person", "people", "human", "man ", "men ", "woman", "women", "child", "children", "boy", "girl",
               "diver", "scientist", "explorer", "hunter", "tribe", "figure of a", "silhouette of a person", "hand ",
               "hands", "face of a", "crowd", "village", "hut", "camp", "fire pit", "tent", "boat", "ship",
               "submarine", "researcher", "paleontologist", "archaeologist", "museum", "skeleton display",
               "caveman", "neanderthal", "hominin", "homo "]
TEXT_WORDS = ["text", "caption", "label", "sign", "letters", "words", "logo", "watermark", "poster", "map of",
              "diagram", "chart", "infographic", "book", "page", "title", "typography", "inscription"]

_rare_cache: dict[str, float] = {}


def zipf(word: str) -> float:
    if word not in _rare_cache:
        from wordfreq import zipf_frequency

        _rare_cache[word] = zipf_frequency(word, "en")
    return _rare_cache[word]


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    bad_sentences: list[str] = field(default_factory=list)  # sentences that must be rewritten or dropped

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, problem: str, sentence: str | None = None) -> None:
        self.problems.append(problem)
        if sentence and sentence not in self.bad_sentences:
            self.bad_sentences.append(sentence)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!;…])\s+", text) if s.strip()]


class Sources:
    """The fact base for one episode: all words and all numeric values in its articles."""

    def __init__(self, texts: list[str]):
        joined = "\n".join(texts)
        self.text = joined
        self.lower = joined.lower()
        self.values = numbers.values(joined)
        self.words = set(re.findall(r"[a-zà-ÿ][a-zà-ÿ'\-]+", self.lower))

    def has_word(self, word: str) -> bool:
        """True if the word, or its singular/plural or Latin/English variant, occurs in the sources."""
        w = word.lower()
        forms = {w, w + "s", w + "es"}
        for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ians", "ia"), ("ids", "idae"), ("ia", "ium"),
                             ("a", "on"), ("i", "us"), ("ae", "a"), ("ines", "inae"), ("an", "a"), ("ans", "a")):
            if w.endswith(suffix):
                forms.add(w[: -len(suffix)] + repl)
        if forms & self.words:
            return True
        return bool(re.search(r"\b" + re.escape(w) + r"\b", self.lower))


ALLOW = {w.lower() for w in GEOLOGY}


def _word(token: str) -> str:
    w = token.strip("'’-")
    return re.sub(r"['’]s$", "", w)


def name_candidates(text: str) -> list[tuple[str, str]]:
    """(word, sentence) for words that look like names or taxa: a capitalised word that is not everyday English
    (stricter mid-sentence, where capitals mean a name), or any word that is very rare in everyday English."""
    out = []
    for sent in sentences(text):
        tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", sent)
        for i, tok in enumerate(tokens):
            word = _word(tok)
            if len(word) < 4 or word.lower() in ALLOW:
                continue
            freq = zipf(word.split("-")[0].lower())
            capital = word[0].isupper() and not word.isupper()
            if capital:
                limit = 3.6 if i > 0 else 2.4
            else:
                limit = 1.6
            if freq < limit:
                out.append((word, sent))
    return out


def check_facts(text: str, src: Sources, report: Report) -> None:
    for sent in sentences(text):
        for q in numbers.unsupported(sent, src.values):
            report.add(f"unsupported number '{q.text}'", sent)
    for word, sent in name_candidates(text):
        if not src.has_word(word):
            report.add(f"name not in sources '{word}'", sent)


def check_style(paragraphs: list[str], report: Report, opening: bool, hedges_needed: int,
                words_target: int) -> None:
    text = " ".join(paragraphs)
    low = text.lower()
    for sent in sentences(text):
        if "!" in sent:
            report.add("exclamation", sent)
        sl = sent.lower()
        for w in ALARM:
            if re.search(r"\b" + re.escape(w), sl):
                report.add(f"alarm word '{w}'", sent)
        for c in CALLS:
            if c in sl and not opening:
                report.add(f"call to action '{c}'", sent)
    if opening and sum(low.count(c) for c in CALLS) > 1:
        report.add("more than one quiet call to action in the opening")
    if re.search(r"[#*_`]|^\s*[-•]\s", text, re.M):
        report.add("markdown or list formatting")
    if low.count("?") > 3:
        report.add("too many questions (keep at most three gentle ones)")
    you = len(re.findall(r"\byou(r|rs|rself)?\b", low))
    if you < 3:
        report.add(f"too little second person ({you}); invite the listener in")
    hedges = sum(low.count(h) for h in HEDGES)
    if hedges < hedges_needed:
        report.add(f"too few hedges ({hedges}); frame uncertain science as uncertain")
    words = len(text.split())
    if words < 0.8 * words_target:
        report.add(f"too short ({words} words, want about {words_target})")
    if words > 1.35 * words_target:
        report.add(f"too long ({words} words, want about {words_target})")
    for p in paragraphs:
        n = len(p.split())
        if n > 190:
            report.add(f"paragraph too long ({n} words)")


def check_prompt(prompt: str) -> list[str]:
    low = " " + prompt.lower() + " "
    issues = [f"mentions '{w.strip()}'" for w in HUMAN_WORDS if re.search(r"\b" + re.escape(w.strip()) + r"\b", low)]
    issues += [f"mentions '{w}'" for w in TEXT_WORDS if re.search(r"\b" + re.escape(w) + r"s?\b", low)]
    if len(prompt.split()) < 18:
        issues.append("too short to describe a scene")
    return issues


def drop_sentences(paragraphs: list[str], bad: list[str]) -> list[str]:
    """Remove the offending sentences; paragraphs that empty out disappear."""
    out = []
    for p in paragraphs:
        kept = [s for s in sentences(p) if s not in bad]
        if kept:
            out.append(" ".join(kept))
    return out


def chapter_report(paragraphs: list[str], src: Sources, opening: bool, hedges_needed: int,
                   words_target: int) -> Report:
    report = Report()
    text = "\n".join(paragraphs)
    check_facts(text, src, report)
    check_style(paragraphs, report, opening, hedges_needed, words_target)
    return report
