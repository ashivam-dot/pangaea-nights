"""Gemini free-tier text models over REST, chosen at run time.

ListModels finds what the key can use; models are tried best first, and a 429 (quota) moves on to the next one.
The key travels only in the x-goog-api-key header: it is never put in a URL, logged or written to disk.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
SKIP = re.compile(r"tts|image|embedding|embed|live|audio|aqa|vision|computer-use|robotics|learnlm|veo|imagen|"
                  r"native|dialog|thinking-exp|nano|banana|lyria|gemma-3n|-e2b|-e4b", re.I)
FALLBACK = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.5-flash-lite",
            "models/gemini-2.0-flash", "models/gemma-3-27b-it"]


class NoModel(RuntimeError):
    pass


def _rank(name: str) -> tuple:
    """Lower sorts first: newest generation, then pro > flash > flash-lite > gemma; stable before preview."""
    n = name.lower()
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    gen = float(m.group(1)) if m else 0.0
    tier = 0 if "pro" in n else 2 if "lite" in n else 1 if "flash" in n else 4 if "gemma" in n else 3
    preview = 1 if ("preview" in n or "exp" in n) else 0
    return (tier >= 3, -gen if tier < 3 else 0, tier, preview, n)


def _thinking(model: str) -> dict | None:
    """Keep thinking modest so it cannot eat the output budget: a budget for 2.5 models, a level for 3.x."""
    n = model.lower()
    if "gemma" in n:
        return None
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    gen = float(m.group(1)) if m else 0.0
    if gen >= 3:
        return {"thinkingLevel": "low"}
    if gen >= 2.5:
        return {"thinkingBudget": 2048}
    return None


class Gemini:
    def __init__(self, key: str | None = None, prefer: str = "quality"):
        self.key = key or os.environ.get("GEMINI_API_KEY", "")
        if not self.key:
            raise NoModel("GEMINI_API_KEY is not set")
        self.session = requests.Session()
        self.session.headers.update({"x-goog-api-key": self.key, "Content-Type": "application/json"})
        self.exhausted: set[str] = set()
        self.models = self.discover()
        if prefer == "fast":
            self.models.sort(key=lambda n: ("pro" in n, _rank(n)))
        self.used: list[str] = []
        self._window: list[tuple[float, int]] = []

    def discover(self) -> list[str]:
        names = []
        try:
            token = ""
            for _ in range(10):
                r = self.session.get(f"{BASE}/models", params={"pageSize": 1000, **({"pageToken": token} if token else {})},
                                     timeout=30)
                r.raise_for_status()
                d = r.json()
                for m in d.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []) and not SKIP.search(m["name"]):
                        if "gemini" in m["name"] or m["name"].endswith("gemma-3-27b-it"):
                            names.append(m["name"])
                token = d.get("nextPageToken", "")
                if not token:
                    break
        except requests.RequestException as e:
            print(f"llm: ListModels failed ({type(e).__name__}); using the fallback list", flush=True)
        if not names:
            names = list(FALLBACK)
        names = sorted(set(names), key=_rank)
        print("llm: models in order: " + ", ".join(n.split("/")[-1] for n in names), flush=True)
        return names

    def _pace(self, tokens: int, per_minute: int = 200_000) -> None:
        now = time.monotonic()
        self._window = [(t, n) for t, n in self._window if now - t < 60]
        while self._window and sum(n for _, n in self._window) + tokens > per_minute:
            time.sleep(max(1.0, 60 - (now - self._window[0][0])))
            now = time.monotonic()
            self._window = [(t, n) for t, n in self._window if now - t < 60]
        self._window.append((now, tokens))

    def generate(self, system: str, prompt: str, schema: dict | None = None, temperature: float = 0.8,
                 max_tokens: int = 32768, models: list[str] | None = None) -> dict | str:
        """Text (or parsed JSON when a schema is given) from the best model that still has quota."""
        last = ""
        for model in models or self.models:
            if model in self.exhausted:
                continue
            gemma = "gemma" in model
            body = {"contents": [{"role": "user", "parts": [{"text": (system + "\n\n" + prompt) if gemma else prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
            thinking = _thinking(model)
            if thinking:
                body["generationConfig"]["thinkingConfig"] = thinking
            if not gemma:
                body["systemInstruction"] = {"parts": [{"text": system}]}
                if schema:
                    body["generationConfig"]["responseMimeType"] = "application/json"
                    body["generationConfig"]["responseSchema"] = schema
            elif schema:
                body["contents"][0]["parts"][0]["text"] += ("\n\nReply with JSON only, matching this schema:\n"
                                                            + json.dumps(schema))
            self._pace((len(system) + len(prompt)) // 4 + max_tokens // 4)
            truncated = 0
            for attempt in range(4):
                started = time.monotonic()
                try:
                    r = self.session.post(f"{BASE}/{model}:generateContent", data=json.dumps(body), timeout=420)
                except requests.RequestException as e:
                    last = f"{model}: {type(e).__name__}"
                    print(f"llm: {model.split('/')[-1]} {type(e).__name__} after {time.monotonic() - started:.0f}s",
                          flush=True)
                    time.sleep(10 * (attempt + 1))
                    continue
                if r.status_code == 429:
                    info = r.text[:2000]
                    if "PerMinute" in info and attempt < 3:
                        delay = re.search(r'"retryDelay":\s*"(\d+)s"', info)
                        time.sleep(min(90, int(delay.group(1)) + 2 if delay else 30))
                        continue
                    self.exhausted.add(model)
                    last = f"{model}: quota exhausted"
                    print(f"llm: {model.split('/')[-1]} quota exhausted, trying the next model", flush=True)
                    break
                if r.status_code in (500, 502, 503, 504):
                    last = f"{model}: HTTP {r.status_code}"
                    time.sleep(15 * (attempt + 1))
                    continue
                if r.status_code == 400 and "thinking" in r.text.lower() and \
                        "thinkingConfig" in body["generationConfig"]:
                    body["generationConfig"].pop("thinkingConfig")
                    continue
                if r.status_code in (400, 404):
                    last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                    self.exhausted.add(model)
                    break
                r.raise_for_status()
                d = r.json()
                cand = (d.get("candidates") or [{}])[0]
                parts = cand.get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                use = d.get("usageMetadata", {})
                finish = cand.get("finishReason")
                print(f"llm: {model.split('/')[-1]} {time.monotonic() - started:.0f}s finish={finish} "
                      f"in={use.get('promptTokenCount')} out={use.get('candidatesTokenCount')} "
                      f"thoughts={use.get('thoughtsTokenCount')}", flush=True)
                if finish == "MAX_TOKENS":
                    truncated += 1
                    last = f"{model}: truncated"
                    if truncated >= 2:
                        break
                    continue
                if not text.strip():
                    last = f"{model}: empty ({finish})"
                    continue
                try:
                    out = parse_json(text) if schema else text
                except ValueError:
                    last = f"{model}: unparseable JSON ({finish})"
                    continue
                if model not in self.used:
                    self.used.append(model)
                return out
        raise NoModel(f"no Gemini model could answer ({last})")


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("not JSON")
