# subseggo

![logo](https://raw.githubusercontent.com/frankier/subseggo/main/logo.png)

🤖 This repo is 100% vibe coded and the README is all LLM text 🤖 

YouTube → subtitles → hierarchical segmentation → cut videos with aligned subtitles

`pipeline.py` is a PEP 723 script (`uv run --script`) that:

1. **download** – grabs a YouTube video with `yt-dlp` → `work/<video_id>/raw.mp4`
2. **transcribe** – cuts the audio into chunks and sends each one to the selected
   provider's `POST /v1/audio/transcriptions` endpoint. The chunk boundaries come
   from the audio itself: `silencedetect` finds every stretch below
   `--silence-dbfs` (default −60), each chunk runs from the first non-silent sample
   to the next silence of at least `--chunk-silence-min-seconds` (default 30 s), and
   a speech region still longer than `--chunk-minutes` (55) is split into the
   fewest equal pieces that fit, overlapping by `--chunk-overlap-seconds`
   (default 60 s). Silences are never uploaded — 42 % of a typical conference day —
   and no chunk contains one, which is the point (see *Why the chunk boundaries
   follow silence*). Each chunk's transcript is then checked against its own audio
   (`transcript-coverage.txt`; `--no-transcript-coverage-check` disables). The
   default is **Voxtral-3-small** (`voxtral-small-transcribe-3`) with segment timestamps
   → `transcript.json`, `raw.srt`, `transcript.txt`
3. **segment** – pipes the timestamped transcript to **`pi` in oneshot mode**
   (`pi --print --session <workdir>/agent-scratch/pi-session.jsonl --model
   mistral/mistral-large-latest --thinking medium`, plus `--effort medium` when the
   pi build supports it; the literal Mistral-Large-3 id `mistral/mistral-large-2512`
   can be pinned with `--segment-model`). pi always runs with a **bare config**:
   `PI_CODING_AGENT_DIR` points at a fresh empty dir in `agent-scratch/`, so the
   user's `~/.pi/agent` (settings, extensions, custom models, auth) is never read —
   same as inside the Docker image, where no `~/.pi/agent` exists; the API key comes
   from the selected provider's environment variable. The command asks the model to
   segment hierarchically into **segments (icebreaker/episode) → sections
   (content/exercise/break)** → intermediate format `segments.json`. Days are never
   guessed: with `--config` each video is one day named by its `[[days]]` entry;
   a single-video run gets no day level at all. The pi run **persists its transcript**:
   `pi-session.jsonl` (pi's native JSONL session — inside the bwrap sandbox it is
   written to the writable `agent-scratch/` and copied out afterwards; pi appends
   incrementally, so it survives even a crashed run) and `pi-transcript.json` (a
   plain-JSON extraction: messages with text/tool calls/tool results, per-message
   usage, and summed token + cost totals).
   The prompt tells the model to first anchor every boundary to a transcript
   timestamp, then optionally fine-tune it visually: `ffprobe` keyframe listing +
   `ffmpeg` frame extraction, ImageMagick `compare -metric` between frames, and
   viewing the extracted frames directly (the model is multimodal). Cut points
   should land on video keyframes — seek **backwards** to the previous keyframe at
   the beginning of a segment (a mid-GOP start would make the stream-copied cut
   begin at the *next* keyframe, dropping the segment start), and **forwards** to
   the next keyframe at the end of one so the tail is not cut off mid-GOP.
4. **edit** – cuts each output's non-`break` section ranges straight out of
   `raw.mp4` with `-c copy`, rebases `raw.srt` onto the resulting timeline, and then
   muxes/burns the subtitle (`subs-` prefix when embedded). The cutting is done here
   rather than by a tool, because every detail of the timeline has to be controlled:
   - each range starts at the **exact PTS** of the nearest video keyframe (`--no-snap-keyframes`
     disables that). A stream copy can only begin on a keyframe, so a boundary that
     is not one has to move; snapping to the *nearest* keyframe rather than always
     forwards avoids the silent loss of up to a GOP of content per cut, and any move
     over 2 s is logged.
   - ffmpeg seeks with `-ss` as an **input** option, which lands on the keyframe at or
     before the requested time. An output-side `-ss` (what ffmpeg-editlist used) drops
     packets until the *next* keyframe and so loses content instead.
   - the parts are joined with the concat demuxer **declaring each part's duration
     explicitly**. Without that it advances by each part's container duration — the
     longer of its two tracks — so audio creeps ahead of video and of the .srt at
     every join (55 ms per join, unbounded in the number of cuts).
   - the `.srt` is rebased from the ranges *actually cut*, and a cue broken by a cut
     is clipped and merged back rather than emitted twice.
   `--reencode` re-encodes instead, which seeks frame-accurately and needs no
   keyframes at all.
5. **align** – measures the result and repairs it. At `--align-samples` points
   (default 8) it finds where the output's sound really came from by
   cross-correlating its loudness envelope against `raw.mp4` across
   `±--align-search-seconds` (default 3 s), and independently locates a 1 s burst of
   downscaled greyscale frames to do the same for the picture. Measurements that
   cannot be trusted (silence, or a static slide where every offset ties) are
   rejected rather than guessed. The sidecar `.srt` is then shifted by the
   interpolated lag, so whatever residual the cut and the transcript leave is
   cancelled and reported (`out/align-report.txt`) instead of shipped. Disable with
   `--no-align-check`, or measure without rewriting the `.srt` with `--no-align-repair`.

## Why the chunk boundaries follow silence

The subtitles were once badly out of sync, and the cut/mux/`.srt` stage was not the
cause: measured on rebuilt outputs of that day, audio, video and `.srt` agreed
within 50 ms. The transcription was. A provider's segment timestamps are produced per chunk, so
its idea of how quickly time passes inside a chunk is the only thing that can put
cues minutes away from the speech they belong to — and a long silence inside a
chunk is where that estimate goes wrong. On one day of a three-day event:

| chunk | real silence | what the provider emitted | effect on that chunk's cues |
| --- | --- | --- | --- |
| 3 | 347.7 s | 77.9 s | +92 s before it, −180 s after |
| 4 | 714 s | 925 s | +211 s, and a 484.6 s silence shrank to 36.9 s |

Chunk 4 also degenerated into a repetition loop around 16190–16700 s: 910 cues
carrying 56 distinct texts. Nothing downstream can repair that, because the error
is not a shift but a locally varying warp.

The silences themselves are real — 42–54 % of each day is constant-`-91 dBFS`
audio — so they are worth avoiding rather than tolerating: no chunk contains one,
none is uploaded (42 % less audio), and each chunk's transcript is checked against
its own audio afterwards so that a provider which still drifts is reported instead
of quietly shipped. The post-edit alignment pass then measures what actually
landed and cancels the residual.

## Outputs (in `work/<video_id>/out/`)

- `original-with-subs.mp4` – the original mp4 with subtitles applied
- `<NN>-<segment>.mp4` – one file per segment (icebreaker/episode),
  **break sections removed**, subtitles applied (`subs-` prefix when embedded)
- `full-nobreaks.mp4` – the whole video with all breaks removed (disable with `--no-combined`)
- `chapters.txt`, `sync-report.txt` (structural: does each output's video timeline
  match the timeline its `.srt` was rebased onto, plus the measured alignment),
  `align-report.txt` (per output, the lag measured at every sample point), plus
  intermediates in the workdir: `transcript.json`, `raw.srt`,
  `transcript-coverage.txt`, `segments.json`, `keyframes.json`, `silences.json`,
  `segment-prompt.md`, `segment-raw-output.txt`, `pi-session.jsonl`,
  `pi-transcript.json`

## API accounting

Every API request is recorded as one row in an **`api-usage.csv`** ledger, itemized
per day/step — columns: `timestamp, day, step, detail, model, requests, input_tokens,
output_tokens, cache_read_tokens, cache_write_tokens, total_tokens, audio_seconds,
cost_usd`.

- **config mode**: one CSV at the workdir root (`workdir_root/api-usage.csv`, e.g.
  `pyscicomp2025/api-usage.csv`), rows carry the `day` column.
- **single-video mode**: the CSV lives in the video's workdir (`work/<id>/api-usage.csv`).

Rows are appended when a request is actually made (cached transcription chunks are
not re-billed), so re-running a command never double-counts. Transcription usage
(`usage.prompt_audio_seconds` + tokens) is also cached per chunk in
`workdir/chunks/chunk-NNN.json` and aggregated into `transcript.json`; the segment
step's usage/cost is read from `pi-transcript.json`.

The `detail` field includes the provider on new rows. Transcription cost is
estimated from list prices and audio duration. The segmentation ledger uses pi's
catalog-derived cost from the session when available. Override transcription
prices with `OPENAI_TRANSCRIBE_USD_PER_AUDIO_HOUR` (Whisper default 0.36) or
`MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR` (default 0.24). Mistral token-price
fallbacks can be overridden with `MISTRAL_USD_PER_1M_INPUT`
(2.0), `MISTRAL_USD_PER_1M_OUTPUT` (6.0), `MISTRAL_USD_PER_1M_CACHE_READ` (0.2),
`MISTRAL_USD_PER_1M_CACHE_WRITE` (0.0).

After **any** command (even a failed or fully-cached one) the pipeline prints the API
usage of the run (per day/step) and the total recorded in the ledger.

## Run locally

```sh
export MISTRAL_API_KEY=...
uv run --script pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --workdir work/VIDEO_ID
```

## Multi-day events via a TOML config

Instead of one video, pass `--config event.toml` listing the **day videos** of an
event (see `config.example.toml`):

```toml
workdir_root = "work"
language = "en"
subs = "embed"

[[days]]
url = "https://www.youtube.com/watch?v=DAY1_VIDEO_ID"
title = "Day 1"

[[days]]
path = "/data/event/day2.mp4"     # local file works too
title = "Day 2"
```

```sh
uv run --script pipeline.py --config event.toml
```

Each `[[days]]` entry must have exactly one of `url=` (YouTube) or `path=` (local
file), plus an optional `title=` that names the day. Every video is processed in
its own workdir (`workdir_root/<slug(title)>`, or `workdir=` if given) and treated
as exactly that one day — the segment stage never guesses days; the day only
appears as context in the prompt and as a header in `chapters.txt`. Top-level TOML
keys set global options, `[[days]]` entries may override them per day, and
explicit CLI flags win over both (resolution: CLI > per-day TOML > global TOML >
built-in default).

Requirements on PATH: `uv`, `ffmpeg`, `ffprobe`, ImageMagick (`compare`, used only by
the *segment* model when it fine-tunes cut points visually), `pi`, `node` (yt-dlp JS
runtime). No ffmpeg-editlist: the edit stage cuts the video itself.
Options: `--language en`, `--subs embed|burn|sidecar`, `--reencode`,
`--transcribe-provider`, `--segment-provider`,
`--transcribe-model`, `--segment-model`, `--effort`, `--chunk-minutes`,
`--chunk-silence-min-seconds`, `--chunk-overlap-seconds`, `--silence-dbfs`,
`--silence-min-seconds`, `--no-transcript-coverage-check`,
`--skip-download/--skip-transcribe/--skip-segment/--skip-edit`, `--force-edit`,
`--no-snap-keyframes`, `--align-samples`, `--align-search-seconds`,
`--no-align-check`, `--no-align-repair`, `--sandbox auto|bwrap|none`,
`--mock-transcript`, `--mock-segments` (offline testing).
See `--help`.

## Providers

Transcription and segmentation each default to Mistral. Set either provider to
`"openai"` at the top level or in a `[[days]]` entry, or pass the corresponding
CLI flag. Supply `MISTRAL_API_KEY` and/or `OPENAI_API_KEY` in the environment for
the stages that make API calls. For Docker, pass the keys with `-e MISTRAL_API_KEY`
and/or `-e OPENAI_API_KEY`.

| Provider | Transcription model | Segmentation model | Segmentation effort |
| --- | --- | --- | --- |
| `mistral` | `voxtral-small-transcribe-3` | `mistral/mistral-large-latest` | `medium` |
| `openai` | `whisper-1` | `openai/gpt-6.1-terra` | `medium` |

For example, this event uses Voxtral for transcription and Terra for
segmentation. To switch transcription to OpenAI, set
`transcribe_provider = "openai"`.

```toml
transcribe_provider = "mistral"
segment_provider = "openai"

[providers.openai.transcribe]
model = "whisper-1"

[providers.openai.segment]
model = "gpt-6.1-terra"
effort = "medium"

[[days]]
path = "recordings/day-1.mp4"
title = "Day 1"
```

The provider tables are optional. Existing `transcribe_model`, `segment_model`
and `effort` overrides take precedence over the selected provider's settings.
If a day or CLI flag switches providers, model and effort overrides from TOML
apply only where the provider matches. OpenAI transcription model overrides must
support segment timestamps in `verbose_json`.

OpenAI transcription uses Whisper's timestamped response as a temporary bridge
until its [scheduled shutdown on February 26, 2027](https://developers.openai.com/api/docs/guides/speech-to-text#timestamps).
OpenAI audio uploads must be under 25 MB. The default 55-minute MP3 chunks fit
under this limit, and the silence-aware chunking above only makes them shorter.
Set a smaller `chunk_minutes` if needed.

## Retries and resuming a partial run

- Each transcription chunk is retried on HTTP 5xx/429/network errors with
  exponential backoff + jitter: `--max-retries 5 --retry-base-seconds 2
  --retry-max-seconds 120` (2s, 4s, 8s, … capped at 120s). Other 4xx errors are
  fatal immediately.
- Every finished chunk's API result is cached atomically in
  `workdir/chunks/chunk-NNN.json`, tagged with a fingerprint of the chunking that
  produced it (offsets, silences, settings). If a run dies (e.g. a persistent 520 on the last
  chunk), just **re-run the same command**: finished chunks are loaded from cache
  and only the missing ones are re-requested. Same for the other stages:
  `raw.mp4`, `transcript.json`, `segments.json` and the `.edit-done` marker make
  download/transcribe/segment/edit resumable. A provider, model or effort change
  refreshes the affected stage and its downstream outputs. Use distinct workdirs
  to keep outputs from both runs for comparison. `--skip-*` forces skipping a stage
  and `--force-edit` redoes the edit stage.
- `workdir/keyframes.json` (the video keyframe probe used for cut snapping) is
  probed once right after download and cached with a fingerprint of `raw.mp4`
  (size + sha256 of the first 1 MiB). The cache is written atomically and
  validated on read: a corrupt or foreign file is moved aside to
  `keyframes.json.bad-*` and re-probed, and a cache from a different `raw.mp4`
  is ignored -- neither crashes the pipeline.

## Sandboxing the segment agent

The `pi` oneshot that segments the video is run inside a bubblewrap sandbox
(`--sandbox auto`, the default; `bwrap` must be on PATH, `none` disables): the
whole filesystem is mounted read-only except `workdir/agent-scratch/` and an
ephemeral `/tmp`, so the agent cannot modify pipeline files (`raw.mp4`,
`keyframes.json`, `segments.json`, ...) even via shell redirects. It also gets
only the `read` and `bash` tools (`--tools read,bash`). Network stays up for the
model API call. With `--sandbox auto` and no `bwrap`, the pipeline warns and runs
unsandboxed.

Independently of the sandbox, pi never uses the user's `~/.pi/agent`: every pi
invocation (including the `--help` capability probes) gets `PI_CODING_AGENT_DIR`
pointing at a freshly emptied `workdir/agent-scratch/pi-agent/`, so pi runs on its
built-in defaults — the same bare configuration as inside the Docker image. Pi
reads the selected provider's API key from the environment and resolves the
model from its built-in catalog via `--model`. The bare
dir lives in `agent-scratch/`, so it is writable both inside the bwrap sandbox
(the only writable bind) and outside. Unit tests:
`uv run --script tests/test_pipeline_units.py`.
- Test the retry/resume logic offline against a mock API:
  `tests/mock_mistral_server.py` (env `FAIL_FIRST=N` fails the first N requests
  with 520, `FAIL_FILE` fails while the file exists), then run with
  `MISTRAL_API_BASE=http://127.0.0.1:8799/v1 MISTRAL_API_KEY=dummy ...`.

## Run in Docker

```sh
docker build -t yt-segment .
docker run --rm \
    -e MISTRAL_API_KEY="$MISTRAL_API_KEY" \
    -v "$PWD/work:/work/work" \
    yt-segment "https://www.youtube.com/watch?v=VIDEO_ID"
```

With rootless podman on an SELinux system the bind mount needs a relabel
(`-v "$PWD/work:/work/work:Z"`); otherwise the container cannot write into it.
The pipeline was tested end-to-end in this image (download → silence-aware
chunking → Voxtral transcription → pi segmentation → keyframe cutting → alignment
→ mp4 + mov_text outputs).

The image is `debian:bookworm-slim` + ffmpeg + ImageMagick + Node 22 (official
tarball; pi needs ≥ 22.19, yt-dlp uses it as JS runtime) + `uv` + `pi`
(`npm i -g @earendil-works/pi-coding-agent@1.5.2`). Entrypoint is
`uv run --script /app/pipeline.py`, so all script args pass straight through.

## Intermediate format (`segments.json`)

```json
{
  "days": [
    {
      "title": "Day 1", "start": 0, "end": 12345,
      "segments": [
        {
          "kind": "icebreaker | episode", "title": "...", "start": 0, "end": 5400,
          "sections": [
            {"kind": "content | exercise | break", "title": "...", "start": 0, "end": 1800}
          ]
        }
      ]
    }
  ]
}
```

All times are seconds in the original video; sections partition their segment.
`build_output_ranges()` turns this into one output per segment, each being the
list of that segment's non-`break` section ranges, and `render_ranges()` cuts them
out of `raw.mp4` and rebases `raw.srt` onto the result; `--no-combined` drops the
extra output that concatenates the episodes.
