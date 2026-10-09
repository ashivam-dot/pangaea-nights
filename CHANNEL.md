# Pangaea Nights — channel bible

**Pangaea Nights** (`@PangaeaNights`, channel id `UCtVG78IUTqvP9rH-LLies0A`) publishes one long, calm sleep
documentary a day about Earth's deep past and its ancient and deep oceans. Each episode is about two hours: slow
narration over slowly moving images and quiet nature sound, then about ten minutes of ambience alone.
Short vertical trailers point to the full episodes.

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
   episode always comes from a series that was not used in the last two episodes. Titles rotate through each
   series' patterns and are never close to an earlier title.
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
| Before the Continents Drifted | Earth's formation and supercontinents | wind | 11 |
| The Deep Sea Before Us | The deep ocean and its ancient lineages | underwater | 11 |
| Oceans of the Ancient Giants | Devonian to Miocene seas and their great swimmers | ocean | 10 |
| Forests of the Dinosaurs | Triassic, Jurassic and Cretaceous land | rain | 10 |
| Worlds Between the Ice | From the Great Dying to the Cenozoic | ocean | 10 |

The backlog holds 85 topics (`bible/backlog.json`), each with five Wikipedia articles. Every article title was
checked against the Wikipedia API to confirm it exists and is not a redirect. At one episode a day this lasts
almost three months. Add topics before the backlog runs out: the planner stops with `BacklogExhausted` instead of
repeating a topic.

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
