# Pangaea Nights — channel bible

**Pangaea Nights** (`@PangaeaNights`, channel id `UCtVG78IUTqvP9rH-LLies0A`) publishes a long, calm sleep
documentary twice a week (Tuesday and Friday, premiering at 21:00 UTC) about Earth's deep past and its ancient and
deep oceans. Automatic episodes stay at two hours during the stable pilot below: slow narration over slowly moving
images and quiet nature sound, then about ten minutes of ambience alone. The old calm per-episode trailer Shorts
are switched off (`config.TRAILER_SHORTS = False`; the code in `studio/shorts.py` stays).

## Binding rules

These are enforced in code. A rule marked *(gate)* stops the pipeline or refuses the upload.

1. **Subject.** Earth's deep past, ancient life, ancient seas and the deep ocean. No space, no human history, no
   mythology, no real people and no real events involving people.
2. **No humans on screen.** Image prompts that mention people, divers, boats, camps or similar are rewritten or
   dropped *(gate: `gates.check_prompt`)*. Every generated image is also screened by a CLIP classifier, and an image
   that looks like a person is never used *(gate: `qa.review`)*.
3. **No text in images.** The same two checks apply to text, labels, signs and watermarks. The only on-screen
   text is the channel's own chapter cards and the thumbnail words.
4. **Every number and every name comes from the sources.** Each episode is written from five Wikipedia articles,
   pinned to their revisions. Every quantity in the narration must match a value in those articles; "66 Ma",
   "66 million" and "sixty-six million" count as the same value. Every rare or capitalised name must also appear in
   them *(gate: `gates.check_facts`)*. Sentences that still fail after three rewrites are removed. A chapter that
   loses too much text that way stops the run.
5. **Uncertain science is framed as uncertain.** Each chapter needs hedges such as "scientists think" or "may
   have" *(gate)*.
6. **Sleep style.** The narration is calm and in the second person. It has no exclamation marks, no alarm or
   clickbait words, no cliffhangers and at most three gentle questions *(gate)*. It can contain one quiet line
   about subscribing, and only in the opening chapter *(gate)*. The pace is about 120 words a minute with long
   pauses. The sound has no sudden loud moments: the loudest momentary level must stay within 8 LU of the episode
   average, which is set to -20 LUFS integrated *(gate: `finish.qa`)*.
7. **Every episode is distinct.** Each episode has its own topic, script, images, chord path, music key and
   ambience. A topic is never published twice *(gate: `ledger.record_episode`)*. The series rotate, and the next
   episode always comes from a series that was not used in the last two episodes. Titles follow one formula (see
   Packaging), rotate between its forms and are never close to an earlier title *(gate: `planner.title_problems`,
   tested on every backlog topic)*.
8. **Honest disclosure.** Every upload declares `containsSyntheticMedia: true` and is marked not made for kids.
   The description says which parts are AI-made, lists the sources, and credits each CC0 recording.
9. **The right channel only.** Uploads go only to `UCtVG78IUTqvP9rH-LLies0A`. Before any write, the signed-in
   account's channel id is checked against it, and on any mismatch every write is refused *(gate:
   `youtube.check_identity`)*.
10. **Stub scripts never publish.** The stub writer exists only to prove the media stages. Its output is marked
    `publishable: false`, and the publish guard refuses it *(gate: `youtube.refusals`)*.

## Series

| Series | Era | Ambience | Topics |
|---|---|---|---|
| Seas Before Fish | Precambrian to Silurian oceans | ocean | 11 |
| The Age of Giant Insects | Devonian to Permian forests and swamps | rain | 11 |
| Ice Age Nights | Pleistocene glaciations and megafauna | wind | 11 |
| Before the Continents Drifted | Earth's formation and supercontinents | wind | 10 |
| The Deep Sea Before Us | The deep ocean and its ancient lineages | underwater | 11 |
| Oceans of the Ancient Giants | Devonian to Miocene seas and their great swimmers | ocean | 10 |
| Forests of the Dinosaurs | Triassic, Jurassic and Cretaceous land | rain | 11 |
| Worlds Between the Ice | From the Great Dying to the Cenozoic | ocean | 12 |

The backlog holds 87 topics (`bible/backlog.json`), each with five Wikipedia articles, a `lead` and a `promise`
for its title. Every article title was checked against the Wikipedia API to confirm it exists and is not a
redirect (last checked 2026-10-22, all 417). Six topics are already published and can never run again. The
planner reads the backlog in order, so the famous topics at its top go first, in this order: the last day of the
dinosaurs, what if the asteroid had missed (science framing, hedged), Pangaea, megalodon, the Ice Age and woolly
mammoths, the world of T. rex, the Jurassic seas, giant insects, the Great Dying and Snowball Earth. At two
episodes a week the backlog lasts about nine months. Add topics before it runs out: the planner stops with
`BacklogExhausted` instead of repeating a topic.

**Proven topics.** Every week the channel looks at what is over-performing on the top AI channels
(`bible/competitors.json`) and takes only the appeal: the dilemma, the scale or the countdown. It never takes the
words. Only public metadata is read: the video id, title, view count, date, length and channel. Descriptions,
transcripts, captions, thumbnails and media are never downloaded, stored or sent to a model.
- A Shorts outlier becomes a new premise of our own. It is about deep-time Earth history only and grounded in a
  sentence of a pinned Wikipedia revision.
- It must pass every bank gate, and it must not reuse a competitor title, share any four-word run with one, or
  name a competitor's channel, character or brand *(gate: `topics.problems`)*.
- A long-form outlier never adds a topic. It can only move a matching backlog topic forward for 60 days *(gate:
  `planner.boosts`)*.

## Cadence and length

- **Reviewed batch release code (25 October 2026).** `ops/cloud_batch.py` and `studio/batch_release.py` can adopt
  exact finished dry previews, require a full AI audiovisual review bound to the MP4 and pinned brief, and recheck
  premise, novelty, rotation, opening-image similarity and the live upload marker before release. Original gates
  remain mandatory. Uploads are private while processing; owner-visible channel, audience and schedule readback
  must match before a result counts. Saved intent prevents uncertain requests from creating duplicate uploads.
  Recovery also requires the reviewed MP4 SHA-256 binding alongside the existing premise marker; unbound older
  uploads remain held. Processed upload status must be confirmed before publication. Provider/network outages
  retry with backoff, while a negative editorial verdict persists for unchanged media.
  A durable batch reservation prevents the ordinary Shorts producer from racing the same calendar. Sleep episodes
  are separate. `batch-release.yml` shares the existing publisher's lock and runs after completed builds and every
  fifteen minutes when `BATCH_RELEASE_ENABLED=true`; manual runs default to review only. The complete media-bound
  verdict is checkpointed before archiving, preserving refusals across archive outages and fresh runners. It uses the channel's
  existing environment credentials. Code and a green workflow are not proof of a public result.
At most two already reserved dry previews may produce concurrently (up to eighteen image shards),
with no overlap against an inherited manual/scheduled producer. Release and review remain in order.
A known editorial hold allows only bounded dry production ahead (four unconsumed previews maximum);
review and release retain their predecessor checks. Every pending dispatch is reconciled before another is sent.

- **Batch production (owner, 25 October 2026).** `strategy/batch-2026-10-25.json` reserves thirty distinct Shorts
  for 25 October–3 November UTC, plus the corrected opening preview for an immediate release after review.
  These are planned targets, not booked YouTube schedules. Dry previews retain the usual script, picture, voice
  and render gates. The release check must also compare earlier batch videos, then use fresh live deduplication,
  owner-visible metadata and complete audiovisual review. Sleep documentaries retain the stable cadence below.

- **Narration first.** Shorts check the actual voice report and measured timeline before generating images.
  Each image shard checks its own slot, so a failed sibling narration does not stop a passing slot. Failed voice
  diagnostics are retained; every final speech, picture, render and audiovisual gate remains mandatory.
- **Targeted image repairs.** A manual dry preview can use a manifest in `.github/replay` through `replay`.
  Original source, exact prompt/model, image hash, dimensions and original take metadata must match; every retained
  base/alternate receives current picture checks again. Changed prompts generate new art. Cached seeds are not
  repainted and counted as separate takes. Repairs have their own serialized dry-run group, so they do not cancel
  a pending ordinary preview. This input cannot directly publish. Deep-water scenes use neutral external fill
  against black water, without the shallow scenes' glowing focal-light style; exposure and readability gates stay in force.
  An explicit hash-bound `cached_only: true` repair checks only the supplied exact art and never generates spare
  takes. Missing inputs or failed current checks hold it; the full pairwise-image and rendered-video gates still apply.
  Bank entries with `production_hold` stay available for audit but cannot be selected, explicitly replayed or released.
  The unbooked sea-scorpion slot is deliberately replaced by the existing asteroid-hideout premise at the same time;
  original artifacts and unresolved scientific refusals remain quarantined.

- **When.** `episode.yml` builds on Tuesday and Friday at 14:10 UTC (`10 14 * * 2,5`). A 2-hour build takes about
  3.5 hours and a 3-hour build about 4.5, and the episode is scheduled for the next 21:00 UTC
  (`config.EPISODE_SLOT_UTC`, `youtube.episode_slot()`). The build only overlaps the end of the 08:00 UTC Shorts
  window, which is done by about 16:00. `health.yml` alerts after 108 hours without an upload (the Friday to
  Tuesday gap is 96).
- **Stable long-form pilot.** Keep the two-hour `full` shape (14 chapters of ~1,030 words), cinema, voice,
  sound rules and cadence steady for four familiar-topic episodes starting with the 23 October dinosaur episode.
  Review actual 72–84-hour and 168–192-hour observations, impressions, CTR, average seconds and watch hours;
  missing authenticated metrics remain unknown. Keep Shorts and older packaging separate. Review no earlier
  than seven days after the fourth release; publishing four alone does not trigger a duration change.
- **Deferred length experiment.** `config.STABLE_LONG_LENGTH = "2h"` holds `python -m studio.length --choice auto`
  at two hours. Set it to `None` only after the pilot review to resume alternation. Explicit `2h` / `3h` inputs
  remain available for private previews. The historical `learn.lengths()` output uses geometric means of
  estimated 72-hour views; it is not a measured comparison of the new pilot. Both shapes keep the same gates.
- **Shared compute.** Long-form image/voice and render stages use at most 16 runners, as does the three-slot
  Shorts matrix. This reserves four of the account's 20 for news, fiction and monitoring. Long image shards keep
  every planned scene (up to ten images per two-hour shard, eleven for a three-hour preview); the existing retry
  budget, image checks and workflow timeouts stay in force. The next full build must verify elapsed time.
  The two owned batch previews each cap image concurrency at eight; voice completes before images start.
  Completed previews still occupy the four-entry buffer while waiting for ordered review, but free a producer slot.

## Packaging

- **Titles.** `<lead>: <promise> | Documentary for Sleep` (or `| Sleep Documentary`, or `<lead> | Documentary for
  Sleep` when the full form is too long). The lead comes first and names something famous and searched for ("The
  Last Day of the Dinosaurs", "What If the Asteroid Had Missed", "Megalodon"). There is no leading jargon: period
  and eon names such as Ediacaran, Hadean or Carboniferous are refused, and the plain phrasings "before the
  dinosaurs", "coal forests" and "Earth's first oceans" are used instead. Rare words in the lead are refused unless
  they are the famous name. The channel never uses "Boring History For Sleep", "(As We Understand It)", "SELECTED
  SOURCES", "Fall Asleep in" or a competitor's exact title *(gate: `planner.title_problems`)*.
- **Thumbnails.** One bright, clearly lit focal creature, large and right of centre, over the episode's dark sleep
  palette. It has 2–4 large, bold words (Cormorant Garamond at weight 700, up to 190 px) naming the famous subject.
  Every thumbnail must pass all of these checks *(gate: `thumbnail.build`, enforced by `finish.qa`)*:
  - Tesseract must read every word back at phone-feed size.
  - The text must keep 4.5:1 contrast with the image behind it.
  - The composed thumbnail's mean luma (Rec. 601) must sit between a floor of 70 and a ceiling of 200, measured at
    feed size on the picture itself. A dark night frame is first lifted toward a mid-tone with a capped gamma
    curve.

## Look and sound

- **Images.** Dark, naturalistic, painterly-photorealistic nature-documentary frames. Each series has a palette and
  camera style, and each episode gets its own art direction that is applied to every prompt.
- **Motion.** Slow depth-parallax camera moves, using six gentle moves that never repeat back to back. Each image
  holds about 40–60 seconds, with four-second crossfades, fine film grain and a soft vignette.
- **Cards.** Cormorant Garamond (SIL OFL 1.1), centred, with a soft dark halo: a title card at the start and a
  "Chapter N" card at each chapter.
- **Score.** Pads and a low drone generated in code. There is one consonant chord per chapter, with
  twenty-second crossfades between chords, no melody and no rhythm, and an episode-specific key.
- **Ambience.** CC0 recordings from Freesound and Wikimedia Commons, layered over a synthetic bed of the same kind
  (surf, rain, wind or deep-water rumble). Each recording's licence is re-checked at download, and recordings with
  sudden loud events are rejected.

Producer metadata is refreshed independently of ordered review, including behind an editorial hold. A completed later preview still occupies the bounded buffer but never an active producer slot. This cannot clear a hold or approve media.

The batch preview image limits are shared across both channels: two Pangaea producers with four image jobs each plus two Kahaani producers with three painting jobs each use at most fourteen of twenty account runners. Six remain for news, release coordinators, narration, CI and monitoring. Earlier per-channel reservations did not account for the other channel and allowed the actual release coordinators to queue behind painting jobs. Existing producer, buffer and release-order limits stay in force.
