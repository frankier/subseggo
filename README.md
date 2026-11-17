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

## Outputs (in `work/<video_id>/out/`)

- `original-with-subs.mp4` – the original mp4 with subtitles applied
- `day<N>-<NN>-<segment>.mp4` – one file per level-2 segment (icebreaker/episode),
  **break sections removed**, subtitles applied (`subs-` prefix when embedded)
- `full-nobreaks.mp4` – the whole video with all breaks removed (disable with `--no-combined`)
- `chapters.txt`, plus intermediates in the workdir: `transcript.json`, `raw.srt`,
  `segments.json`, `editlist.yaml`, `segment-prompt.md`, `segment-raw-output.txt`

## Run locally

```sh
export MISTRAL_API_KEY=...
uv run --script pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --workdir work/VIDEO_ID
```

Requirements on PATH: `uv`, `ffmpeg`, `ffprobe`, `pi`, `node` (yt-dlp JS runtime).
Options: `--language en`, `--subs embed|burn|sidecar`, `--reencode`,
`--transcribe-model`, `--segment-model`, `--effort`, `--chunk-minutes`,
`--skip-download/--skip-transcribe/--skip-segment` (resume), `--mock-transcript`,
`--mock-segments` (offline testing). See `--help`.

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
`HH:MM:SS`), plus an optional combined output.
