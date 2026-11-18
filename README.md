# YouTube → subtitles → hierarchical segmentation → editlist pipeline

`pipeline.py` is a PEP 723 script (`uv run --script`) that:

1. **download** – grabs a YouTube video with `yt-dlp` → `work/<video_id>/raw.mp4`
2. **transcribe** – chunks the audio (≤ 55 min per request, the Mistral API limit is
   60 min / 500 MB) and sends each chunk to `POST /v1/audio/transcriptions` with
   **Voxtral-3-small** (`voxtral-small-transcribe-3`), `timestamp_granularities=segment`
   → `transcript.json`, `raw.srt`, `transcript.txt`
3. **segment** – pipes the timestamped transcript to **`pi` in oneshot mode**
   (`pi --print --no-session --model mistral/mistral-large-latest --thinking medium`,
   plus `--effort medium` when the pi build supports it; the literal Mistral-Large-3 id
   `mistral/mistral-large-2512` can be pinned with `--segment-model`) asking the
   Mistral-Large model to
   segment hierarchically into **days → segments (icebreaker/episode) → sections
   (content/exercise/break)** → intermediate format `segments.json`
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
- `day<N>-<NN>-<segment>.mp4` – one file per level-2 segment (icebreaker/episode),
  **break sections removed**, subtitles applied (`subs-` prefix when embedded)
- `full-nobreaks.mp4` – the whole video with all breaks removed (disable with `--no-combined`)
- `chapters.txt`, `sync-report.txt` (per-output subtitle/video drift measured after
  cutting), plus intermediates in the workdir: `transcript.json`, `raw.srt`,
  `segments.json`, `editlist.yaml`, `keyframes.json`, `segment-prompt.md`,
  `segment-raw-output.txt`

## Run locally

```sh
export MISTRAL_API_KEY=...
uv run --script pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --workdir work/VIDEO_ID
```

Requirements on PATH: `uv`, `ffmpeg`, `ffprobe`, `pi`, `node` (yt-dlp JS runtime).
Options: `--language en`, `--subs embed|burn|sidecar`, `--reencode`,
`--transcribe-model`, `--segment-model`, `--effort`, `--chunk-minutes`,
`--skip-download/--skip-transcribe/--skip-segment/--skip-edit`, `--force-edit`,
`--no-snap-keyframes`, `--mock-transcript`, `--mock-segments` (offline testing).
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

The image is `debian:bookworm-slim` + ffmpeg + Node 22 (official tarball; pi needs
≥ 22.19, yt-dlp uses it as JS runtime) + `uv` + `pi` (`npm i -g
@earendil-works/pi-coding-agent@1.5.2`). Entrypoint is
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
