"""Benchmark one text-to-image candidate on this runner: seconds per image, peak memory, automatic review scores.

python bench/images.py <candidate> <out_dir> [n_prompts]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio import sdcpp  # noqa: E402

STYLE = ("Painterly photorealistic nature documentary still, cinematic wide composition, soft natural light, "
         "deep calm palette of midnight blue, teal and muted amber, gentle haze, fine detail, quiet atmosphere.")
PROMPTS = [
    "A Cambrian sea floor at dusk, about five hundred million years ago: an Anomalocaris, a long segmented "
    "swimming animal with a row of flexible lobes along each side, two spiny curved grasping appendages at the "
    "front of its head and large stalked eyes, glides above a pale sandy bottom scattered with sponges and small "
    "trilobites. Shafts of dim light come down through green water.",
    "A woolly mammoth with a long shaggy brown coat, a high domed head, small ears and long curving tusks stands on "
    "a frozen grassland steppe under a dark blue twilight sky, snow drifting, a low pale moon on the horizon.",
    "A deep-sea hydrothermal vent in total darkness lit only by faint bioluminescence: tall black mineral chimneys "
    "release shimmering plumes of hot water, dense clusters of giant tube worms with white tubes and bright red "
    "plumes sway at their base, small pale crabs on the rocks.",
    "A Carboniferous coal swamp forest at dawn: towering Lepidodendron trees with diamond-patterned bark and "
    "grass-like leaf tufts, giant horsetails, still black water reflecting mist; a large dragonfly-like Meganeura "
    "with a wingspan of about seventy centimetres rests on a fallen trunk.",
    "Close-up of rippled ancient sandstone on a tidal flat covered in a thin film of water, fossil ripple marks "
    "and the faint impression of a stromatolite mound, evening light low and golden, calm reflections.",
]


def main() -> None:
    name, out = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else len(PROMPTS)
    out.mkdir(parents=True, exist_ok=True)
    cand = sdcpp.CANDIDATES[name]
    t0 = time.monotonic()
    sdcpp.fetch(cand)
    download = time.monotonic() - t0
    rows = []
    with sdcpp.Server(cand) as server:
        for i, p in enumerate(PROMPTS[:n]):
            t = time.monotonic()
            data = server.generate(f"{p} {STYLE}", seed=1000 + i)
            sec = time.monotonic() - t
            (out / f"{name}-{i}.png").write_bytes(data)
            rows.append({"prompt": i, "seconds": round(sec, 1)})
            print(f"{name} prompt {i}: {sec:.1f}s", flush=True)
            (out / f"{name}.json").write_text(json.dumps({"candidate": name, "rows": rows}, indent=1))
        load = server.load_seconds
        rss = server.peak_rss_mb()
        log = server.tail(4000)
    (out / f"{name}-server.log").write_text(log)
    from studio import plates, qa

    for row in rows:
        img = sdcpp.to_pil((out / f"{name}-{row['prompt']}.png").read_bytes())
        row["size"] = list(img.size)
        row["qa"] = qa.review(img, PROMPTS[row["prompt"]]).as_dict()
    t = time.monotonic()
    img = sdcpp.to_pil((out / f"{name}-0.png").read_bytes())
    plate = plates.upscale(img, plates.PLATE)
    up = time.monotonic() - t
    plate.save(out / f"{name}-0-plate.jpg", quality=92)
    t = time.monotonic()
    plates.depth(plate)
    dep = time.monotonic() - t
    secs = [r["seconds"] for r in rows]
    summary = {"candidate": name, "license": cand.license, "steps": cand.steps, "size": [cand.width, cand.height],
               "download_s": round(download, 1), "load_s": round(load, 1), "peak_rss_mb": round(rss),
               "first_image_s": secs[0], "mean_image_s": round(sum(secs[1:] or secs) / len(secs[1:] or secs), 1),
               "upscale_s": round(up, 1), "depth_s": round(dep, 1),
               "mean_aesthetic": round(sum(r["qa"]["aesthetic"] for r in rows) / len(rows), 3),
               "mean_similarity": round(sum(r["qa"]["similarity"] for r in rows) / len(rows), 4),
               "rejected": sum(not r["qa"]["ok"] for r in rows), "rows": rows}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
