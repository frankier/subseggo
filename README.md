# YouTube → subtitles → hierarchical segmentation → editlist pipeline

`pipeline.py` is a PEP 723 script (`uv run --script`) that:

1. **download** – grabs a YouTube video with `yt-dlp` → `work/<video_id>/raw.mp4`
2. **transcribe** – chunks the audio (≤ 55 min per request, the Mistral API limit is
   60 min / 500 MB) and sends each chunk to `POST /v1/audio/transcriptions` with
   **Voxtral-3-small** (`voxtral-small-transcribe-3`), `timestamp_granularities=segment`
   → `transcript.json`, `raw.srt`, `transcript.txt`
3. **segment** – pipes the timestamped transcript to **`pi` in oneshot mode**
   (`pi --print --session <workdir>/agent-scratch/pi-session.jsonl --model
   mistral/mistral-large-latest --thinking medium`, plus `--effort medium` when the
   pi build supports it; the literal Mistral-Large-3 id `mistral/mistral-large-2512`
   can be pinned with `--segment-model`) asking the Mistral-Large model to
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
4. **edit** – converts `segments.json` to an `editlist.yaml` for
   [`ffmpeg-editlist`](https://pypi.org/project/ffmpeg-editlist/) and runs it with
   `--srt --no-mkv-props`, then muxes the cut `.srt` into each `.mp4` as `mov_text`.
   Because the cuts are stream-copied, each cut actually starts at the first video
   keyframe with `dts >= start` (probed once from `raw.mp4`, cached in
   `workdir/keyframes.json`), so editlist start times are snapped to those keyframes
   first — otherwise the video timeline gets a hole at every cut point and the
   nominal-time subtitles drift against it (up to a GOP per cut, accumulating in
   multi-range outputs). After cutting, a sync report (`out/sync-report.txt`)
   verifies per output that actual video range starts match the nominal editlist
   times. Disable snapping with `--no-snap-keyframes`.

## Outputs (in `work/<video_id>/out/`)

- `original-with-subs.mp4` – the original mp4 with subtitles applied
- `<NN>-<segment>.mp4` – one file per segment (icebreaker/episode),
  **break sections removed**, subtitles applied (`subs-` prefix when embedded)
- `full-nobreaks.mp4` – the whole video with all breaks removed (disable with `--no-combined`)
- `chapters.txt`, `sync-report.txt` (per-output subtitle/video drift measured after
  cutting), plus intermediates in the workdir: `transcript.json`, `raw.srt`,
  `segments.json`, `editlist.yaml`, `keyframes.json`, `segment-prompt.md`,
  `segment-raw-output.txt`, `pi-session.jsonl`, `pi-transcript.json`

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

Costs: the APIs return usage, not cost, so USD is computed from list prices —
transcription is billed per audio hour, the pi run uses pi's own catalog-derived cost
from the session. Override the defaults via env if pricing changes:
`MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR` (default 0.24), `MISTRAL_USD_PER_1M_INPUT`
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

Requirements on PATH: `uv`, `ffmpeg`, `ffprobe`, ImageMagick (`compare`, used when the
segment model fine-tunes cut points visually), `pi`, `node` (yt-dlp JS runtime).
Options: `--language en`, `--subs embed|burn|sidecar`, `--reencode`,
`--transcribe-model`, `--segment-model`, `--effort`, `--chunk-minutes`,
`--skip-download/--skip-transcribe/--skip-segment/--skip-edit`, `--force-edit`,
`--no-snap-keyframes`, `--sandbox auto|bwrap|none`, `--mock-transcript`,
`--mock-segments` (offline testing).
See `--help`.

## Retries and resuming a partial run

- Each transcription chunk is retried on HTTP 5xx/429/network errors with
  exponential backoff + jitter: `--max-retries 5 --retry-base-seconds 2
  --retry-max-seconds 120` (2s, 4s, 8s, … capped at 120s). Other 4xx errors are
  fatal immediately.
- Every finished chunk's API result is cached atomically in
  `workdir/chunks/chunk-NNN.json`. If a run dies (e.g. a persistent 520 on the last
  chunk), just **re-run the same command**: finished chunks are loaded from cache
  and only the missing ones are re-requested. Same for the other stages:
  `raw.mp4`, `transcript.json`, `segments.json` and the `.edit-done` marker make
  download/transcribe/segment/edit resumable; `--skip-*` forces skipping a stage
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
unsandboxed. Unit tests: `uv run --script tests/test_pipeline_units.py`.
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
The pipeline was tested end-to-end in this image (download → Voxtral
transcription → pi segmentation → editlist → mp4 + mov_text outputs).

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
`build_editlist_yaml()` converts this to ffmpeg-editlist YAML: one output per
segment, editlist = the non-`break` section ranges (start/stop pairs in
`HH:MM:SS.mmm`, start times snapped to keyframes when available), plus an optional
combined output.
