"""Static checks on the workflow files: they parse, schedules honour the pause switches, dispatches default to a dry
run, and the Shorts and the episode stay inside the free plan's 20 concurrent jobs."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from studio import config, fanout, ladder  # noqa: E402

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def load(name):
    doc = yaml.safe_load((WF / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # YAML 1.1 reads the key `on` as True
    return doc


@pytest.mark.parametrize("path", sorted(WF.glob("*.yml")), ids=lambda p: p.name)
def test_workflow_parses(path):
    doc = load(path.name)
    assert doc["jobs"] and doc["on"]
    for name, job in doc["jobs"].items():
        assert ("runs-on" in job and "steps" in job) or "uses" in job, name  # or a reusable-workflow call


def first_jobs(doc):
    return [j for j in doc["jobs"].values() if "needs" not in j]


@pytest.mark.parametrize("name, gate", [
    ("episode.yml", "vars.PAUSED == 'false'"),
    ("shorts.yml", "vars.PAUSED == 'false' && vars.SHORTS_PAUSED != 'true'"),
    ("shorts-learn.yml", "vars.PAUSED == 'false' && vars.SHORTS_PAUSED != 'true'"),
    ("topics.yml", "vars.PAUSED == 'false' && vars.SHORTS_PAUSED != 'true'"),
])
def test_schedules_are_gated_on_pause(name, gate):
    doc = load(name)
    assert doc["on"]["schedule"]
    for job in first_jobs(doc):
        assert "github.event_name != 'schedule'" in job["if"] and gate in job["if"]


@pytest.mark.parametrize("name", ["shorts.yml", "shorts-learn.yml", "topics.yml"])
def test_dispatch_defaults_to_dry_run(name):
    inp = load(name)["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert inp["type"] == "boolean" and inp["default"] is True


def test_episode_dispatch_defaults_to_dry():
    assert load("episode.yml")["on"]["workflow_dispatch"]["inputs"]["mode"]["default"] == "dry"


def test_shorts_mode_expression_is_dry_unless_switched_off():
    mode = load("shorts.yml")["env"]["MODE"]
    assert "github.event_name == 'schedule' && 'publish'" in mode and "inputs.dry_run != false && 'dry'" in mode


def test_cadence():
    assert load("shorts.yml")["on"]["schedule"] == [{"cron": "0 8 * * *"}]
    assert load("episode.yml")["on"]["schedule"] == [{"cron": "10 14 * * 2,5"}]
    assert config.EPISODE_SLOT_UTC == (21, 0)
    assert load("topics.yml")["on"]["schedule"] == [{"cron": "0 5 * * 1"}]  # Mondays, before the learner


def test_topics_scan_saves_only_when_not_dry_and_alerts():
    doc = load("topics.yml")
    job = doc["jobs"]["scan"]
    assert job["environment"] == "youtube" and doc["concurrency"]["group"] == "topics"
    assert job["env"]["DRY"] == "${{ github.event_name != 'schedule' && inputs.dry_run != false }}"
    steps = {s.get("name"): s for s in job["steps"]}
    assert "--dry" in steps["Scan"]["run"] and {"YOUTUBE_OAUTH_JSON", "GEMINI_API_KEY"} <= set(steps["Scan"]["env"])
    for name in ("Save the ledger", "Low supply"):
        assert steps[name]["if"] == "env.DRY != 'true'"
    assert "alerts.txt" in steps["Low supply"]["run"] and "--label alert" in steps["Low supply"]["run"]
    assert "failure()" in steps["Alert"]["if"] and "alert.send" in steps["Alert"]["run"]
    assert "topics" in load("doctor.yml")["on"]["workflow_run"]["workflows"]


def test_concurrency_stays_within_twenty():
    shorts = load("shorts.yml")["jobs"]
    shorts_peak = shorts["images"]["strategy"]["max-parallel"] + len(ladder.SLOTS)  # images + one voice per slot
    assert shorts_peak <= fanout.LIMIT - 1  # room for an episode plan job
    ep = config.shape("full")
    assert ep.image_shards + ep.voice_shards <= fanout.LIMIT
    for length in config.LENGTHS:
        s = config.shape("full", length)
        assert s.image_shards + s.voice_shards <= fanout.LIMIT, length
    # each side waits for the other before fanning out, and only the Shorts look at the episode's fan-out
    ep_steps = load("episode.yml")["jobs"]["plan"]["steps"]
    assert "--wait-for shorts.yml" in ep_steps[-1]["run"]
    sh_steps = " ".join(s.get("run", "") for s in shorts["plan"]["steps"])
    assert "--wait-for episode.yml --fanned-out" in sh_steps


def test_shorts_publish_carries_the_alert_path_and_launch_date():
    doc = load("shorts.yml")
    assert doc["env"]["SHORTS_LAUNCH"] == "${{ vars.SHORTS_LAUNCH }}"
    step = next(s for s in doc["jobs"]["publish"]["steps"] if "ladder-publish" in s.get("run", ""))
    assert {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "YOUTUBE_OAUTH_JSON"} <= set(step["env"])
    rec = " ".join(s.get("run", "") for s in doc["jobs"]["record"]["steps"])
    assert "alerts.txt" in rec and "--label alert" in rec


def test_fanout_rules():
    runs = [{"databaseId": 1, "status": "in_progress"}, {"databaseId": 2, "status": "completed"},
            {"databaseId": 3, "status": "queued"}]
    assert [r["databaseId"] for r in fanout.active(runs, "3")] == [1]
    assert not fanout.fanned_out([{"name": "plan", "status": "in_progress"}])
    assert fanout.fanned_out([{"name": "plan", "status": "completed"}, {"name": "images (0)", "status": "queued"}])
    assert not fanout.fanned_out([{"name": "plan", "status": "completed"}, {"name": "images", "status": "completed"}])
