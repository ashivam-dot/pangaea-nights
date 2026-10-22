"""Keep the Shorts and the long episode inside the free plan's 20 concurrent jobs.

Each workflow can fill the runner pool on its own (episode: 18 image + 2 voice jobs; Shorts: 17 image + 2 voice), so
neither may fan out while the other is fanned out. The rule is asymmetric so the two can never wait on each other:

- the episode's plan job, before it emits its matrix, waits for every other active shorts.yml run to finish;
- the Shorts' plan job waits only for an episode run with matrix jobs (images, voice, render) active: a
  fanned-out episode.

An episode still in its plan job is one runner, and the Shorts' peak is 19, so that overlap stays at 20.

    python -m studio.fanout --wait-for shorts.yml --minutes 150
    python -m studio.fanout --wait-for episode.yml --fanned-out --minutes 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

LIMIT = 20
ACTIVE = ("queued", "in_progress", "waiting", "requested", "pending")


def active(runs: list[dict], self_id: str | None = None) -> list[dict]:
    """Runs (gh run list --json databaseId,status) that are not finished, other than this one."""
    return [r for r in runs if r.get("status") in ACTIVE and str(r.get("databaseId")) != str(self_id or "")]


FAN_OUT = ("images", "voice", "render")  # the matrix stages; audio, finish, publish, record hold one runner each


def fanned_out(jobs: list[dict]) -> bool:
    """A run is fanned out while any of its matrix jobs (images, voice, render: named "images (3)" and so on) is
    active (gh run view --json jobs). A late run still in a single-runner stage such as finish holds one runner,
    which the Shorts' peak of 19 leaves room for."""
    return any(j.get("name", "").split(" ")[0] in FAN_OUT and j.get("status") in ACTIVE for j in jobs)


def _gh(*args: str) -> list | dict:
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out or "[]")


def blocking(workflow: str, only_fanned_out: bool, self_id: str | None, repo: str | None) -> list[str]:
    where = ["-R", repo] if repo else []
    runs = active(_gh("run", "list", *where, "--workflow", workflow, "--limit", "20", "--json", "databaseId,status"),
                  self_id)
    if only_fanned_out:
        runs = [r for r in runs
                if fanned_out(_gh("run", "view", *where, str(r["databaseId"]), "--json", "jobs").get("jobs", []))]
    return [str(r["databaseId"]) for r in runs]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="studio.fanout")
    p.add_argument("--wait-for", required=True, help="the other workflow file, e.g. shorts.yml")
    p.add_argument("--fanned-out", action="store_true", help="only wait for a run past its plan job")
    p.add_argument("--minutes", type=float, default=60.0, help="give up (and fail) after this long")
    p.add_argument("--poll", type=float, default=60.0)
    args = p.parse_args(argv)
    deadline = time.monotonic() + args.minutes * 60
    while True:
        busy = blocking(args.wait_for, args.fanned_out, os.environ.get("GITHUB_RUN_ID"),
                        os.environ.get("GITHUB_REPOSITORY"))
        if not busy:
            print(f"no active {args.wait_for} run: fanning out")
            return
        if time.monotonic() >= deadline:
            sys.exit(f"{args.wait_for} runs {busy} still active after {args.minutes:g} min: not fanning out "
                     f"(that would exceed {LIMIT} concurrent jobs)")
        print(f"waiting for {args.wait_for} runs {busy}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
