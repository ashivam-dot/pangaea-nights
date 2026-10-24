"""A durable batch reservation prevents scheduled producers from racing the batch's rotation and slots."""
import json
from datetime import datetime, timedelta, timezone


def production_hold(state, at=None):
    path = state / "batch-production.json"
    if not path.exists():
        return None
    batch = json.loads(path.read_text())
    if not batch.get("enabled"):
        return None
    if batch.get("schema") != 1:
        raise ValueError("unknown batch reservation schema; inspect before producing")
    times = [datetime.fromisoformat(j["target"].replace("Z", "+00:00")) for j in batch["jobs"]
             if j["target"] != "now-after-review"]
    horizon = max(times) + timedelta(hours=1) if times else datetime.min.replace(tzinfo=timezone.utc)
    if batch.get("complete") and (at or datetime.now(timezone.utc)) > horizon:
        return None
    return "the reviewed batch coordinator owns these premises and release slots"
