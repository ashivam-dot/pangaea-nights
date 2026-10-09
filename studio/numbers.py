"""Find every quantity in a text (digits or words, with scale words and geological units) as a numeric value.

The fact gate compares the values in the narration with the values in the sources: '66 million', 'sixty-six
million' and '66 Ma' are all 66,000,000.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "dozen": 12,
}
SCALES = {"hundred": 100, "thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12}
UNIT_SCALES = {"ma": 10 ** 6, "mya": 10 ** 6, "myr": 10 ** 6, "my": 10 ** 6, "ga": 10 ** 9, "gya": 10 ** 9,
               "bya": 10 ** 9, "byr": 10 ** 9, "ka": 10 ** 3, "kya": 10 ** 3, "kyr": 10 ** 3}
EXEMPT_WORDS = {"zero", "one", "two", "three"}  # pronoun-like and counting words ('no one', 'two eyes')

_DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_WORD = "|".join(sorted(list(WORD_VALUES) + list(SCALES) + ["and", "a", "an", "half"], key=len, reverse=True))
_TOKEN = re.compile(rf"(?P<num>{_DIGITS})|(?P<word>\b(?:{_WORD})\b(?:[\s-]+\b(?:{_WORD})\b)*)", re.I)
_SCALE_AFTER = re.compile(r"^[\s-]*(hundred|thousand|million|billion|trillion|ma|mya|myr|ga|gya|bya|byr|ka|kya|kyr)\b",
                          re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    text: str
    start: int
    exempt: bool = False


def _words_value(words: list[str]) -> float | None:
    words = [w.lower() for w in words if w.lower() != "and"]
    if words and words[0] in ("a", "an"):
        if len(words) > 1 and words[1] in SCALES:
            words = ["one"] + words[1:]
        else:
            return None
    if not words:
        return None
    if words[0] == "half":
        rest = [w for w in words[1:] if w not in ("a", "an")]
        if rest and all(w in SCALES for w in rest):
            v = 0.5
            for w in rest:
                v *= SCALES[w]
            return v
        return None
    if any(w in ("a", "an", "half") for w in words):
        return None
    total, current, seen = 0.0, 0.0, False
    for w in words:
        if w in WORD_VALUES:
            current += WORD_VALUES[w]
            seen = True
        elif w == "hundred":
            current = (current or 1) * 100
            seen = True
        elif w in SCALES:
            total += (current or 1) * SCALES[w]
            current = 0.0
            seen = True
    return total + current if seen else None


def quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    for m in _TOKEN.finditer(text):
        if m.group("num"):
            raw = m.group("num")
            prev = text[max(0, m.start() - 1):m.start()]
            if prev.isalpha():  # part of a token like 'C4' or 'Q5'
                continue
            value = float(raw.replace(",", ""))
            after = _SCALE_AFTER.match(text[m.end():])
            label = raw
            if after:
                word = after.group(1).lower()
                value *= SCALES.get(word, UNIT_SCALES.get(word, 1))
                label = raw + " " + after.group(1)
                # 'five hundred million': a digit followed by 'hundred million'
                more = _SCALE_AFTER.match(text[m.end() + after.end():])
                if word == "hundred" and more and more.group(1).lower() in SCALES:
                    value *= SCALES[more.group(1).lower()]
                    label += " " + more.group(1)
            out.append(Quantity(value, label, m.start()))
        else:
            phrase = m.group("word")
            parts = re.split(r"[\s-]+", phrase.strip())
            # Split runs like 'one and two' into separate numbers only at 'and' between two non-scale numbers.
            value = _words_value(parts)
            if value is None:
                continue
            lowered = [p.lower() for p in parts if p.lower() not in ("and", "a", "an")]
            exempt = len(lowered) == 1 and lowered[0] in EXEMPT_WORDS
            after = _SCALE_AFTER.match(text[m.end():])
            if after and after.group(1).lower() in UNIT_SCALES:
                value *= UNIT_SCALES[after.group(1).lower()]
                exempt = False
            out.append(Quantity(value, phrase.strip(), m.start(), exempt))
    return out


def values(text: str) -> set[float]:
    """Every value a source supports, including both ends of ranges and bare mantissas of scaled numbers."""
    vals = set()
    for q in quantities(text):
        vals.add(round(q.value, 6))
    for m in re.finditer(rf"({_DIGITS})", text):
        vals.add(round(float(m.group(1).replace(",", "")), 6))
    return vals


def supported(q: Quantity, source_values: set[float]) -> bool:
    if q.exempt:
        return True
    v = round(q.value, 6)
    if v in source_values:
        return True
    # A source may give the same figure with a different unit scale ('66 Ma' vs '66 million years').
    return any(abs(v - s) <= 1e-9 * max(abs(v), 1) for s in source_values)


def unsupported(narration: str, source_values: set[float]) -> list[Quantity]:
    return [q for q in quantities(narration) if not supported(q, source_values)]
