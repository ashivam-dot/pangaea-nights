"""Keep automatic long episodes at the stable pilot length; defer the 2 h / 3 h experiment.

The plan job runs `python -m studio.length --choice auto` before it sizes the shards; the choice lands in
PANGAEA_LENGTH, which config.shape("full") resolves to the "full" (2 h) or "full3h" (3 h) shape. The episode keeps
that shape name, so every later stage builds the same size and the ledger records which length each episode was;
learn.lengths() reports historical estimates. Clear config.STABLE_LONG_LENGTH after reviewing four familiar-topic
episodes at 7 days to resume alternation. The loudness and sleep gates are the same for both.
"""
from __future__ import annotations

import argparse
import os

from . import config, ledger

NAMES = {v: k for k, v in config.LENGTHS.items()}  # shape name -> "2h" / "3h"


def length_of(shape_name: str) -> str | None:
    return NAMES.get(shape_name)


def experiment_episodes(data: dict) -> list[dict]:
    """Published long episodes created on or after config.LENGTH_EXPERIMENT_FROM, oldest first."""
    return [e for e in data["episodes"] if e.get("shape") in NAMES and e.get("video_id")
            and str(e.get("created", ""))[:10] >= config.LENGTH_EXPERIMENT_FROM]


def next_length(data: dict) -> str:
    """Stable pilot first; once released, alternate from the last published experiment episode."""
    if config.STABLE_LONG_LENGTH is not None:
        if config.STABLE_LONG_LENGTH not in config.LENGTHS:
            raise ValueError("STABLE_LONG_LENGTH must be a configured length or None")
        return config.STABLE_LONG_LENGTH
    done = experiment_episodes(data)
    if not done:
        return "2h"
    return "3h" if length_of(done[-1]["shape"]) == "2h" else "2h"


def choose(choice: str, data: dict) -> str:
    if choice in config.LENGTHS:
        return choice
    if choice != "auto":
        raise ValueError(f"length choice must be auto or one of {sorted(config.LENGTHS)}, not {choice!r}")
    return next_length(data)


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--choice", default="auto", help="auto (stable pilot, then deferred alternation), 2h or 3h")
    args = p.parse_args(argv)
    length = choose((args.choice or "auto").strip() or "auto", ledger.load())
    env = os.environ.get("GITHUB_ENV")
    if env:
        with open(env, "a", encoding="utf-8") as f:
            f.write(f"PANGAEA_LENGTH={length}\n")
    print(f"length: {length} (shape {config.LENGTHS[length]})")
    return length


if __name__ == "__main__":
    main()
