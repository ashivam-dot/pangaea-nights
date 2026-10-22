# Pangaea Nights studio

This is a fully automated, free studio for the **Pangaea Nights** YouTube channel: two- and three-hour sleep
documentaries about Earth's deep past, published on Tuesdays and Fridays at 21:00 UTC. Everything runs on the GitHub-hosted CPU runners
of this public repository; nothing runs on a personal machine. The binding editorial rules are in
[`CHANNEL.md`](CHANNEL.md).

## Pipeline

`.github/workflows/episode.yml` runs one job per stage. The image, voice and render stages fan out across
parallel runners.

| Stage | Job | What it does |
|---|---|---|
| Plan and write | `plan` (1) | The planner picks the next topic: never one already published, and never from either of the last two series. It fetches the topic's five Wikipedia articles at pinned revisions. Gemini then writes the outline, 14 chapters (about 14,500 words; 21 chapters for the 3-hour form). Trailers are off (`config.TRAILER_SHORTS`). Each chapter goes through the gates and up to three rewrites (`studio/writer.py`, `gates.py`, `numbers.py`). |
| Images | `images` (18) | Each runner generates its share of 141 stills (at most 8 per runner) with FLUX.2 klein 4B (Q8 GGUF, 768x432, 4 steps) in stable-diffusion.cpp on CPU. Every take is reviewed automatically (CLIP checks for people, text and broken anatomy, prompt similarity, an aesthetic score and pixel statistics), with up to three takes per image. A Real-ESRGAN upscale and a Depth Anything V2 depth map turn each still into a render plate (`sdcpp.py`, `qa.py`, `plates.py`, `images.py`). |
| Voice | `voice` (2) | Kokoro (`af_heart`) reads each chapter one sentence at a time, with calm pauses, at about 120 wpm. The first paragraph of each chapter sets its reading speed (never below 0.8), and the pauses are then scaled to land on the target. faster-whisper listens to every paragraph and re-reads any paragraph it cannot hear clearly (`voice.py`, `narrate.py`, `speech.py`). |
| Audio | `audio` (1) | Builds the timeline: chapter starts, title cards, shots, and the ambience tail. It then mixes the narration with the code-generated score and the CC0 ambience loop, using two-pass loudnorm to -20 LUFS / -3 dBTP (`timeline.py`, `score.py`, `ambience.py`, `mix.py`). |
| Render | `render` (15) | Each runner renders one contiguous run of shots at 1920x1080, 24 fps: depth-parallax camera moves, 4 s crossfades, grain and serif cards, encoded with x264 (`render.py`). |
| Finish | `finish` (1) | Joins the segments with a stream copy, muxes the audio and embeds chapter markers. It checks the format, duration, loudness, sample frames and swapped images. It also builds the thumbnail (with a contrast and OCR legibility test) and the Shorts (`finish.py`, `thumbnail.py`, `shorts.py`). Then **publish** runs, and the run is **recorded** in the ledger on the `state` branch. |
| Alert | `alert` | When a scheduled or publish run fails, it opens an issue labelled `alert`. |

`health.yml` runs every six hours and opens an `alert` issue when the channel is not paused and there has been no
upload for 108 hours. `tests.yml` runs the unit tests and a smoke test. The smoke test runs every media stage on
synthetic inputs in about five minutes.

## Publishing guard

`studio/youtube.py` refuses to upload unless **all** of these hold:

- the run mode is `publish`;
- the repository variable `PAUSED` is exactly `false` (so a missing variable also counts as paused);
- the script came from the real writer (`writer: gemini`, `publishable: true`, no stub chapters);
- media QA passed;
- `channels.list(mine=true)` returns exactly one channel, and its id is `UCtVG78IUTqvP9rH-LLies0A`.

Videos are uploaded private with `publishAt` set to the next 21:00 UTC (`youtube.episode_slot()`). Every upload uses category 27 (Education),
`selfDeclaredMadeForKids: false`, `containsSyntheticMedia: true`, and language `en`. Trailer Shorts cut from the
episode are off (`config.TRAILER_SHORTS`); the Shorts are the ladder below. Each upload carries a marker tag, so a
rerun finds the earlier upload instead of creating a duplicate.

## Ladder Shorts

`shorts.yml` (08:00 UTC daily) builds 30–45 s "pick a number 1–8" Shorts from the premise banks in `bible/ladder/`
(four series, 16 premises each; every fact is one sentence of a pinned Wikipedia revision). The narrator speaks the
verdict ("Real answer: number N.") with the sourced fact; there is no mascot. `studio/ladder_cli.py` runs one stage
per job: `plan` → `images` (9 stills per slot, at most 17 jobs at once) and `voice` → `render` (every gate has a floor
and a ceiling) → `publish` → `record`.

- Cadence: slot A releases 14:30 UTC. From launch day 8 (`SHORTS_LAUNCH`, YYYY-MM-DD) slot B (00:30 UTC next day)
  is added, but only while the previous 7 publish builds passed every gate. No launch date means no publishing.
- Pause: schedules need `PAUSED=false` and `SHORTS_PAUSED` not `true`. Dispatches default to `dry_run` (build and
  gate, upload nothing); with it off, `test` uploads private with no release and consumes no premise.
- Novelty: 4-gram overlap ≤35% against the last 30 scripts, first frame CLIP ≤0.93 against the last 30, every
  option pair CLIP ≤0.90.
- Every upload is read back with `videos.list` (madeForKids, selfDeclaredMadeForKids, privacy, publishAt). A
  mismatch goes to Telegram, to an `alert` issue and to `ladder.alerts` in the ledger.
- `shorts-learn.yml` (06:00 UTC) reads 24 h / 72 h numbers and updates the series weights and the pt.2 queue.
- Runner budget: the episode's plan job waits for any active Shorts run before fanning out (18 + 2 jobs), and the
  Shorts wait for an episode that has already fanned out (`studio/fanout.py`), so the two stay within 20 jobs.
- `python -m studio.cli ladder-check` validates every bank premise; fix failing data rather than the gates.

## Running it

| Want | Do |
|---|---|
| A 2-chapter test (~8–10 min) | Actions → episode → mode `short` |
| A full-length dry run (never uploads) | Actions → episode → mode `dry` |
| A real episode now | Actions → episode → mode `publish` (still refused while paused) |
| Publish a passed full dry run without rebuilding | Actions → promote → its run id; release `now` (public ~20 min after upload) or `slot` (next scheduled slot). Same guard; artifacts expire after 2–3 days |
| Scheduled episodes | Set `PAUSED=false`; the Tue/Fri 14:10 UTC schedule builds each one (alternating 2 h / 3 h) and schedules it for 21:00 UTC |
| A ladder Short test | Actions → shorts → `dry_run` on (default), or off with mode `test` for a private upload |

With writer `auto`, Gemini is used when `GEMINI_API_KEY` exists. Without the key, the run uses the stub writer, which
assembles the script from real Wikipedia sentences to prove the media stages. Stub output can never be published.

## What the coordinator must supply

Add these as secrets of the `youtube` environment:

- `GEMINI_API_KEY` is a Google AI Studio key from the channel's personal Google account. The writer finds the
  available models with ListModels and moves to the next model when it hits a 429.
- `YOUTUBE_OAUTH_JSON` is an authorized-user JSON (`client_id`, `client_secret`, `refresh_token`) for the
  Pangaea Nights channel. It needs the `youtube.upload` and `youtube` scopes, and comes from a Desktop OAuth client
  in a personal Google Cloud project.

Then set the repository variable `PAUSED` to `false`.

YouTube limits that still apply:

- Until the Cloud project passes YouTube's API audit, uploads from unverified projects are locked to private.
- Videos longer than 15 minutes and custom thumbnails need a phone-verified channel.
- The default quota of 10,000 units a day covers one episode (1,600 for the upload, 50 for the thumbnail) and two
  ladder Shorts (1,600 each, plus a first comment and the read-back).

## Repository layout

```
bible/        series.json (8 series), backlog.json (85 topics, 5 verified Wikipedia articles each), ladder/ (Shorts banks)
studio/       the pipeline (see the table above); cli.py runs each stage
assets/       fonts (Cormorant Garamond, OFL) and the CC0 ambience catalogue
bench/        the CPU image-model and voice benchmarks
tests/        unit tests (gates, numbers, planner, ledger, upload guard, timeline) and smoke.py
requirements/ pinned dependency sets per job
```

## Licences of the moving parts

| Part | Licence |
|---|---|
| stable-diffusion.cpp | MIT |
| FLUX.2 klein 4B (GGUF by leejet) and the FLUX.2 small decoder | Apache-2.0 |
| Qwen3-4B text encoder | Apache-2.0 |
| Real-ESRGAN realesr-general-x4v3 | BSD-3-Clause |
| spandrel | MIT |
| Depth Anything V2 Small | Apache-2.0 |
| open_clip ViT-B-32 (OpenAI weights) | MIT |
| LAION aesthetic predictor | MIT |
| Kokoro-82M | Apache-2.0 |
| misaki | Apache-2.0 |
| faster-whisper (base.en) | MIT |
| Cormorant Garamond | SIL OFL 1.1 |
| Ambience recordings | CC0, verified per recording at download time |
