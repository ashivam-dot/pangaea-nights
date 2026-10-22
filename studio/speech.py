"""Script text -> what the narrator says: numbers, units and symbols spelled out in calm British English."""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]


def _under_1000(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    rest = n % 100
    return _ONES[n // 100] + " hundred" + (" and " + _under_1000(rest) if rest else "")


def integer_words(n: int) -> str:
    if n < 0:
        return "minus " + integer_words(-n)
    if n < 1000:
        return _under_1000(n)
    parts = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(_under_1000(n // value) + " " + name if n // value < 1000 else
                         integer_words(n // value) + " " + name)
            n %= value
    if n:
        parts.append(("and " if n < 100 else "") + _under_1000(n))
    return " ".join(parts)


def number_words(text: str) -> str:
    """'252' -> 'two hundred and fifty-two'; '1.5' -> 'one point five'; '3,500' -> 'three thousand five hundred'."""
    t = text.replace(",", "")
    if "." in t:
        whole, frac = t.split(".", 1)
        return integer_words(int(whole or 0)) + " point " + " ".join(_ONES[int(d)] for d in frac)
    return integer_words(int(t))


def year_words(n: int) -> str:
    hi, lo = divmod(n, 100)
    if lo == 0:
        return integer_words(hi) + " hundred"
    return integer_words(hi) + " " + ("oh-" + _ONES[lo] if lo < 10 else _under_1000(lo))


UNITS = {
    "km/h": "kilometres per hour", "mph": "miles per hour", "km²": "square kilometres", "km2": "square kilometres", "m²": "square metres", "km": "kilometres",
    "cm": "centimetres", "mm": "millimetres", "m": "metres", "kg": "kilograms",
    "°C": "degrees Celsius", "°F": "degrees Fahrenheit", "Ma": "million years", "mya": "million years ago",
    "Mya": "million years ago", "Ga": "billion years", "bya": "billion years ago", "ka": "thousand years",
    "kya": "thousand years ago", "%": "percent", "ft": "feet", "lb": "pounds",
}
_UNIT_RE = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"


def speakable(text: str) -> str:
    """The sentence as Kokoro should read it. Hyphen ranges become 'to', years read as years."""
    s = text.replace("\u2009", " ").replace("\u00a0", " ")
    s = re.sub(r"\bc\.\s?(?=\d)", "about ", s)
    s = re.sub(r"~\s?(?=\d)", "about ", s)
    s = s.replace("±", " plus or minus ").replace("&", " and ")
    s = re.sub(rf"({_NUM})\s?[–—-]\s?({_NUM})", r"\1 to \2", s)
    s = re.sub(rf"({_NUM})\s?({_UNIT_RE})(?![A-Za-z])", lambda m: f"{m.group(1)} {UNITS[m.group(2)]}", s)

    def year_or_number(m: re.Match) -> str:
        raw = m.group(0)
        before = s[max(0, m.start() - 12):m.start()].lower()
        after = s[m.end():m.end() + 12].lower()
        n = raw.replace(",", "")
        if ("," not in raw and "." not in raw and len(n) == 4 and 1500 <= int(n) <= 2099
                and re.search(r"\b(in|since|by|of|from|until|around|year)\s*$", before)
                and not re.match(r"\s*(million|billion|thousand|years|metres|kilometres|species)", after)):
            return year_words(int(n))
        return number_words(raw)

    s = re.sub(_NUM, year_or_number, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.?;…])\s+(?=[A-Z\"'‘“(])", paragraph.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and len(p.split()) <= 2:
            out[-1] += " " + p
        else:
            out.append(p)
    return out
