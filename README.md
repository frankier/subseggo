# YouTube → subtitles → hierarchical segmentation → editlist pipeline

`pipeline.py` is a PEP 723 script (`uv run --script`) that:

1. **download** – grabs a YouTube video with `yt-dlp` → `work/<video_id>/raw.mp4`
2. **transcribe** – chunks the audio (55 minutes by default) and sends each chunk
   to the selected provider's `POST /v1/audio/transcriptions` endpoint. The
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

Requirements on PATH: `uv`, `ffmpeg`, `ffprobe`, ImageMagick (`compare`, used when the
segment model fine-tunes cut points visually), `pi`, `node` (yt-dlp JS runtime).
Options: `--language en`, `--subs embed|burn|sidecar`, `--reencode`,
`--transcribe-provider`, `--segment-provider`,
`--transcribe-model`, `--segment-model`, `--effort`, `--chunk-minutes`,
`--skip-download/--skip-transcribe/--skip-segment/--skip-edit`, `--force-edit`,
`--no-snap-keyframes`, `--sandbox auto|bwrap|none`, `--mock-transcript`,
`--mock-segments` (offline testing).
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
under this limit. Set a smaller `chunk_minutes` if needed.

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
