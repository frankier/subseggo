#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "yt-dlp",
#   "requests",
#   "pyyaml",
#   "ffmpeg-editlist[srt]",
# ]
# ///
"""
YouTube -> subtitles -> hierarchical segmentation -> editlist -> mp4 outputs.

Pipeline stages (resumable: re-run the same command after a failure — finished
downloads, per-chunk transcription results (workdir/chunks/), segmentation and
edit stages are all cached/skipped automatically):
  - transcription chunks retry 5xx/429/network errors with exponential backoff
    (--max-retries, --retry-base-seconds, --retry-max-seconds); stages can also be
    skipped explicitly via --skip-download/--skip-transcribe/--skip-segment/--skip-edit.
  1. download    : yt-dlp grabs the YouTube video as raw.mp4 (+ raw.info.json)
  2. transcribe  : audio chunks -> Mistral /v1/audio/transcriptions (Voxtral-3-small)
                   -> transcript.json (segments) + raw.srt
  3. segment     : `pi` oneshot (Mistral-Large-3, medium effort) over the transcript
                   -> segments.json (intermediate format: days > segments > sections)
  4. edit        : segments.json -> editlist.yaml -> ffmpeg-editlist (--srt)
                   -> per-segment mp4 outputs with break sections removed,
                      plus the original mp4 with subtitles applied.

Usage:
  uv run --script pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" [options]

Requires on PATH: ffmpeg, ffprobe, pi (and node as yt-dlp JS runtime, optional).
Requires env: MISTRAL_API_KEY (for the transcription API; `pi` uses its own auth).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

MISTRAL_API_BASE = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
DEFAULT_TRANSCRIBE_MODEL = "voxtral-small-transcribe-3"   # Voxtral Small Transcribe 3 ("Voxtral-3-small")
DEFAULT_SEGMENT_MODEL = "mistral/mistral-large-latest"     # Mistral Large line (Large 3 = mistral-large-2512; pin via --segment-model)
DEFAULT_EFFORT = "medium"
API_MAX_CHUNK_MINUTES = 55  # API limit is 60 min / 500 MB per request; stay under


def log(msg: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(shutil.which(c) and c or c for c in cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    return subprocess.run(cmd, check=True, **kw)


def which_or_die(name: str) -> str:
    p = shutil.which(name)
    if not p:
        sys.exit(f"ERROR: '{name}' not found on PATH (install it or use the Docker image)")
    return p


# --------------------------------------------------------------------------- #
# Stage 1: download
# --------------------------------------------------------------------------- #

def video_id_from_url(url: str) -> str:
    m = re.search(r"(?:youtu\.be/|v=|/shorts/|/embed/)([\w-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    return re.sub(r"[^\w-]", "_", url)[:40]


def step_download(url: str, workdir: Path, height: int | None) -> dict:
    from yt_dlp import YoutubeDL

    raw_mp4 = workdir / "raw.mp4"
    if raw_mp4.exists() and (workdir / "raw.info.json").exists():
        log(f"download: raw.mp4 already exists in {workdir}, skipping")
        return json.loads((workdir / "raw.info.json").read_text())

    if height:
        fmt = f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/b[height<={height}][ext=mp4]/b[ext=mp4]/b"
    else:
        fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
    ydl = YoutubeDL({
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "raw.%(ext)s"),
        "noplaylist": True,
        "writeinfojson": True,
        "quiet": False,
        "no_warnings": True,
        # YouTube needs a JS runtime for some formats; node/deno/bun all work.
        "js_runtimes": {"node": {}, "deno": {}, "bun": {}},
        "ffmpeg_location": shutil.which("ffmpeg") or None,
    })
    log(f"download: {url}")
    info = ydl.extract_info(url, download=True)
    if info.get("_type") == "playlist":
        info = info["entries"][0]
    if not raw_mp4.exists():
        cand = workdir / ydl.prepare_filename(info)
        if cand.exists() and cand != raw_mp4:
            shutil.move(str(cand), str(raw_mp4))
    if not raw_mp4.exists():
        sys.exit("ERROR: download finished but raw.mp4 not found")
    log(f"download: {raw_mp4} ({raw_mp4.stat().st_size/1e6:.1f} MB), "
        f"title={info.get('title')!r}, duration={info.get('duration')}s")
    return info


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [which_or_die("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


# --------------------------------------------------------------------------- #
# Stage 2: transcribe (Voxtral-3-small via Mistral API, chunked)
# --------------------------------------------------------------------------- #

def extract_audio_chunk(raw_mp4: Path, start: float, dur: float, out: Path) -> None:
    run([which_or_die("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(raw_mp4),
         "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-c:a", "libmp3lame", str(out)])


def transcribe_chunk(mp3: Path, model: str, language: str | None, api_key: str,
                     max_retries: int = 5, retry_base: float = 2.0,
                     retry_max: float = 120.0, timeout: int = 3600) -> list[dict]:
    """POST one audio chunk to /v1/audio/transcriptions, retrying 5xx/429/network
    errors with exponential backoff + jitter. 4xx (other than 429) is fatal."""
    data = {"model": model, "timestamp_granularities": "segment", "stream": "false"}
    if language:
        data["language"] = language
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            with open(mp3, "rb") as fh:
                files = {"file": (mp3.name, fh, "audio/mpeg")}
                r = requests.post(f"{MISTRAL_API_BASE}/audio/transcriptions",
                                  headers={"Authorization": f"Bearer {api_key}"},
                                  files=files, data=data, timeout=timeout)
            if r.status_code == 200:
                body = r.json()
                segs = body.get("segments") or []
                out = [{"start": float(s["start"]), "end": float(s["end"]),
                        "text": s.get("text", ""), "speaker_id": s.get("speaker_id")}
                       for s in segs]
                log(f"  chunk {mp3.name}: {len(out)} segments, usage={body.get('usage')}")
                return out
            last_err = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code < 500 and r.status_code != 429:
                sys.exit(f"ERROR: transcription failed, not retrying ({last_err})")
        except requests.RequestException as e:
            last_err = f"network error: {e}"
        if attempt < max_retries:
            delay = min(retry_max, retry_base * (2 ** (attempt - 1))) + random.uniform(0, 1.0)
            log(f"  chunk {mp3.name}: attempt {attempt}/{max_retries} failed ({last_err}); "
                f"retrying in {delay:.1f}s")
            time.sleep(delay)
    sys.exit(f"ERROR: transcription failed after {max_retries} attempts for {mp3.name}: "
             f"{last_err}\nAlready-completed chunks are cached in {mp3.parent / 'chunks'}; "
             f"re-run the same command to resume.")


def srt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, s in enumerate(segments, 1):
        text = re.sub(r"\s+", " ", s["text"]).strip()
        lines.append(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{text}\n")
    path.write_text("\n".join(lines))


def step_transcribe(workdir: Path, model: str, language: str | None,
                    chunk_minutes: int, mock_transcript: Path | None,
                    max_retries: int = 5, retry_base: float = 2.0,
                    retry_max: float = 120.0) -> list[dict]:
    transcript_json = workdir / "transcript.json"
    if transcript_json.exists():
        log("transcribe: transcript.json exists, skipping")
        return json.loads(transcript_json.read_text())["segments"]
    if mock_transcript:
        log(f"transcribe: using mock transcript {mock_transcript}")
        segments = json.loads(mock_transcript.read_text())["segments"]
    else:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            sys.exit("ERROR: MISTRAL_API_KEY is not set (needed for /v1/audio/transcriptions)")
        raw_mp4 = workdir / "raw.mp4"
        duration = probe_duration(raw_mp4)
        chunk_seconds = chunk_minutes * 60
        overlap = 15.0  # seconds of overlap between chunks to avoid cutting mid-segment
        cache_dir = workdir / "chunks"   # per-chunk API results -> resumable
        cache_dir.mkdir(exist_ok=True)
        # chunk boundaries (deterministic, so cache indexes are stable across runs)
        boundaries = []
        start = 0.0
        while start < duration - 1:
            boundaries.append((start, min(chunk_seconds, duration - start)))
            start += chunk_seconds - overlap
        all_segments: list[dict] = []
        for idx, (start, dur) in enumerate(boundaries):
            cache_file = cache_dir / f"chunk-{idx:03d}.json"
            if cache_file.exists():
                log(f"transcribe: chunk {idx} [{start:.0f}s +{dur:.0f}s] cached -> {cache_file.name}")
                raw_segs = json.loads(cache_file.read_text())["segments"]
            else:
                chunk = workdir / f"chunk-{idx:03d}.mp3"
                log(f"transcribe: chunk {idx} [{start:.0f}s +{dur:.0f}s] of {duration:.0f}s")
                extract_audio_chunk(raw_mp4, start, dur, chunk)
                raw_segs = transcribe_chunk(chunk, model, language, api_key,
                                            max_retries=max_retries, retry_base=retry_base,
                                            retry_max=retry_max)
                chunk.unlink(missing_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text(json.dumps({"model": model, "segments": raw_segs}))
                tmp.rename(cache_file)  # atomic: only complete results are cached
            for s in raw_segs:
                s = dict(s)
                s["start"] += start
                s["end"] += start
                # drop segments that lie fully inside the overlap tail of this chunk
                if idx > 0 and s["end"] <= start + overlap:
                    continue
                all_segments.append(s)
        all_segments.sort(key=lambda s: s["start"])
        segments = all_segments
    (workdir / "transcript.json").write_text(json.dumps(
        {"model": model, "segments": segments}, indent=2))
    write_srt(segments, workdir / "raw.srt")
    (workdir / "transcript.txt").write_text(
        "\n".join(f"[{srt_ts(s['start'])[:8]}] {s['text'].strip()}" for s in segments))
    log(f"transcribe: {len(segments)} segments -> transcript.json, raw.srt, transcript.txt")
    return segments


# --------------------------------------------------------------------------- #
# Stage 3: hierarchical segmentation via pi oneshot (Mistral-Large-3, medium effort)
# --------------------------------------------------------------------------- #

SEGMENT_PROMPT = """\
You are segmenting a recorded workshop/training video into a hierarchy using ONLY the
transcript below. The transcript lines are prefixed with the timestamp in the ORIGINAL
video, format [hh:mm:ss].

VIDEO METADATA:
Title: {title}
Uploader: {uploader}
Duration: {duration_hms} ({duration:.0f} seconds)
Description (may contain the real chapter list):
{description}

TRANSCRIPT (timestamps are in the original video):
{transcript}

TASK:
Segment the video hierarchically into exactly three levels:
  1. days      - top-level blocks of the recording (e.g. "Day 1"); usually 1 day for a
                 single recording, more only if the video clearly spans multiple days.
  2. segments  - level-2 parts inside a day, of kind "icebreaker" or "episode"
                 (an episode = one main session/talk of the day).
  3. sections  - level-3 parts inside a segment: the parts of an episode before/after an
                 exercise, the exercise itself, and the break time itself.
                 Each section has kind: "content" | "exercise" | "break".
                 "break" sections are the break/lunch times that should be REMOVED.

RULES:
- All times are in SECONDS in the original video (numbers, not strings).
- days/segments/sections must be contiguous and non-overlapping; sections must exactly
  partition their parent segment (section.start == previous section.end).
- Anchor every boundary to a transcript timestamp (use the timestamp of the line where
  the topic changes). Round to whole seconds.
- Keep titles short and descriptive.
- Do NOT invent content that is not in the transcript.

OUTPUT FORMAT: reply with exactly one fenced ```json code block and nothing else,
conforming to this schema:
```json
{{
  "days": [
    {{
      "title": "Day 1",
      "start": 0,
      "end": 12345,
      "segments": [
        {{
          "kind": "icebreaker" | "episode",
          "title": "...",
          "start": 0,
          "end": 5400,
          "sections": [
            {{"kind": "content" | "exercise" | "break", "title": "...", "start": 0, "end": 1800}}
          ]
        }}
      ]
    }}
  ]
}}
```
"""


def fmt_hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_segment_prompt(workdir: Path, segments: list[dict]) -> str:
    info = json.loads((workdir / "raw.info.json").read_text())
    duration = probe_duration(workdir / "raw.mp4")
    # compact transcript: [hh:mm:ss] text
    lines = []
    for s in segments:
        text = re.sub(r"\s+", " ", s["text"]).strip()
        if text:
            lines.append(f"[{fmt_hms(s['start'])}] {text}")
    transcript = "\n".join(lines)
    return SEGMENT_PROMPT.format(
        title=info.get("title", ""), uploader=info.get("uploader", ""),
        duration=duration, duration_hms=fmt_hms(duration),
        description=(info.get("description") or "")[:8000],
        transcript=transcript)


def pi_supports_effort(pi_bin: str) -> bool:
    try:
        help_out = subprocess.run([pi_bin, "--help"], capture_output=True, text=True).stdout
        return "--effort" in help_out
    except Exception:
        return False


def step_segment(workdir: Path, model: str, effort: str, pi_bin: str,
                 mock_segments: Path | None) -> dict:
    segments_json = workdir / "segments.json"
    if segments_json.exists():
        log("segment: segments.json exists, skipping")
        return json.loads(segments_json.read_text())
    if mock_segments:
        log(f"segment: using mock segments {mock_segments}")
        data = json.loads(mock_segments.read_text())
    else:
        transcript = json.loads((workdir / "transcript.json").read_text())["segments"]
        prompt = build_segment_prompt(workdir, transcript)
        prompt_file = workdir / "segment-prompt.md"
        prompt_file.write_text(prompt)
        log(f"segment: prompt is {len(prompt)} chars -> {prompt_file}")
        cmd = [pi_bin, "--print", "--no-session", "--model", model,
               "--thinking", effort, "@" + str(prompt_file.resolve())]
        if pi_supports_effort(pi_bin):
            cmd += ["--effort", effort]
        log(f"segment: running pi oneshot: {' '.join(cmd[:8])} ...")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
        if proc.returncode != 0:
            sys.exit(f"ERROR: pi failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
        (workdir / "segment-raw-output.txt").write_text(proc.stdout)
        data = extract_json_block(proc.stdout)
    data = validate_segments(data, probe_duration(workdir / "raw.mp4"))
    segments_json.write_text(json.dumps(data, indent=2))
    n_days = len(data["days"])
    n_seg = sum(len(d["segments"]) for d in data["days"])
    n_sec = sum(len(s["sections"]) for d in data["days"] for s in d["segments"])
    n_break = sum(1 for d in data["days"] for s in d["segments"]
                  for sec in s["sections"] if sec["kind"] == "break")
    log(f"segment: {n_days} days, {n_seg} segments, {n_sec} sections ({n_break} breaks) -> segments.json")
    return data


def extract_json_block(text: str) -> dict:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not blocks:
        # fall back: first {...} spanning to last }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            sys.exit(f"ERROR: no JSON block found in pi output:\n{text[-2000:]}")
        blocks = [m.group(0)]
    return json.loads(blocks[-1])


def validate_segments(data: dict, duration: float) -> dict:
    """Normalize + sanity-check the intermediate format; clamp to video duration."""
    if "days" not in data:
        sys.exit(f"ERROR: segments JSON missing 'days': {list(data)}")
    for day in data["days"]:
        day["start"] = max(0.0, min(float(day.get("start", 0)), duration))
        day["end"] = min(float(day.get("end", duration)), duration)
        for seg in day.get("segments", []):
            seg.setdefault("kind", "episode")
            seg["start"] = max(0.0, min(float(seg.get("start", 0)), duration))
            seg["end"] = min(float(seg.get("end", duration)), duration)
            if seg["end"] <= seg["start"]:
                seg["end"] = min(seg["start"] + 1, duration)
            sections = []
            for sec in seg.get("sections", []):
                sec.setdefault("kind", "content")
                sec["start"] = max(seg["start"], min(float(sec.get("start", seg["start"])), seg["end"]))
                sec["end"] = min(seg["end"], max(float(sec.get("end", sec["start"])), sec["start"]))
                if sec["end"] > sec["start"]:
                    sections.append(sec)
            if not sections:
                sections = [{"kind": "content", "title": seg.get("title", "segment"),
                             "start": seg["start"], "end": seg["end"]}]
            seg["sections"] = sections
    return data


# --------------------------------------------------------------------------- #
# Stage 4: intermediate format -> ffmpeg-editlist YAML -> apply
# --------------------------------------------------------------------------- #

def slugify(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\- ]+", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:maxlen].strip("-") or "segment"


def hms(t: float) -> str:
    return fmt_hms(float(t))


def build_editlist_yaml(data: dict, combined: bool) -> list[dict]:
    """Convert segments.json (days>segments>sections) to an ffmpeg-editlist YAML list.

    One output per level-2 segment (icebreaker/episode), containing only non-break
    sections; TOC entries are emitted for every section. Optionally one combined
    output with all breaks removed.
    """
    # NOTE: ffmpeg-editlist calls mkvpropedit whenever an entry has title/description
    # or TOC entries, and mkvpropedit cannot edit .mp4 files. Since our outputs are
    # .mp4, we emit bare start/stop editlists here and write chapters ourselves.
    doc: list[dict] = [{"input": "raw.mp4"}]

    def ranges_for(seg: dict) -> list[tuple[float, float, str]]:
        out = []
        for sec in seg["sections"]:
            if sec["kind"] == "break":
                continue
            out.append((sec["start"], sec["end"], sec.get("title", "")))
        if not out:
            out = [(seg["start"], seg["end"], seg.get("title", ""))]
        return out

    day_idx = 0
    for day in data["days"]:
        day_idx += 1
        for seg_idx, seg in enumerate(day["segments"], 1):
            name = f"day{day_idx}-{seg_idx:02d}-{slugify(seg.get('title', seg['kind']))}.mp4"
            entry = {"output": name, "editlist": []}
            for (s, e, _title) in ranges_for(seg):
                entry["editlist"].append({"start": hms(s)})
                entry["editlist"].append({"stop": hms(e)})
            doc.append(entry)

    if combined:
        entry = {"output": "full-nobreaks.mp4", "editlist": []}
        all_ranges = []
        for day in data["days"]:
            for seg in day["segments"]:
                all_ranges.extend(ranges_for(seg))
        all_ranges.sort()
        merged: list[list] = []
        for s, e, _title in all_ranges:
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            entry["editlist"].append({"start": hms(s)})
            entry["editlist"].append({"stop": hms(e)})
        doc.append(entry)
    return doc


def write_chapters(data: dict, outdir: Path) -> None:
    """Human/ffmpeg-readable chapter lists derived from the intermediate format."""
    lines = []
    for day_idx, day in enumerate(data["days"], 1):
        lines.append(f"# {day.get('title', f'Day {day_idx}')} [{hms(day['start'])}-{hms(day['end'])}]")
        for seg_idx, seg in enumerate(day["segments"], 1):
            lines.append(f"  segment {seg_idx} ({seg.get('kind')}): {seg.get('title','')} "
                         f"[{hms(seg['start'])}-{hms(seg['end'])}]")
            for sec in seg["sections"]:
                marker = " (REMOVED)" if sec["kind"] == "break" else ""
                lines.append(f"    - {sec.get('kind')}: {sec.get('title','')} "
                             f"[{hms(sec['start'])}-{hms(sec['end'])}]{marker}")
    (outdir / "chapters.txt").write_text("\n".join(lines) + "\n")


def apply_subs(mp4: Path, srt: Path, out: Path, mode: str) -> None:
    """mode: embed (mov_text, stream copy) | burn (re-encode) | sidecar (just copy srt)"""
    if mode == "embed":
        run([which_or_die("ffmpeg"), "-y", "-loglevel", "error",
             "-i", str(mp4), "-i", str(srt),
             "-map", "0:v", "-map", "0:a?", "-map", "1:s",
             "-c", "copy", "-c:s", "mov_text",
             "-metadata:s:s:0", "language=eng",
             str(out)])
    elif mode == "burn":
        run([which_or_die("ffmpeg"), "-y", "-loglevel", "error",
             "-i", str(mp4), "-vf", f"subtitles={srt}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-c:a", "copy", str(out)])


def step_edit(workdir: Path, data: dict, subs: str, reencode: bool, combined: bool) -> None:
    workdir = workdir.resolve()
    outdir = (workdir / "out").resolve()
    outdir.mkdir(exist_ok=True)
    doc = build_editlist_yaml(data, combined=combined)
    editlist_yaml = workdir / "editlist.yaml"
    editlist_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    log(f"edit: wrote {editlist_yaml} ({len(doc)-1} outputs)")

    fel = [sys.executable, "-m", "ffmpeg_editlist"]
    cmd = fel + [str(editlist_yaml), str(workdir), "-o", str(outdir),
                 "--srt", "--no-mkv-props", "-f"]
    if reencode:
        cmd.append("--reencode")
    run(cmd, cwd=workdir)
    write_chapters(data, outdir)

    # ffmpeg-editlist leaves .srt sidecars (we passed --no-mkv-props because the
    # outputs are .mp4); mux them into the mp4s now.
    for entry in doc[1:]:
        mp4 = outdir / entry["output"]
        srt = mp4.with_suffix(".srt")
        if not mp4.exists():
            log(f"edit: WARNING output missing: {mp4}")
            continue
        if subs == "sidecar":
            continue
        final = outdir / ("subs-" + entry["output"]) if subs == "embed" else outdir / entry["output"]
        if srt.exists():
            tmp_out = final.with_suffix(".tmp.mp4")
            apply_subs(mp4, srt, tmp_out, subs)
            if subs == "embed":
                shutil.move(str(tmp_out), str(final))
            else:
                mp4.unlink()
                shutil.move(str(tmp_out), str(final))
            log(f"edit: subtitles {subs} -> {final.name}")
        elif subs != "sidecar":
            log(f"edit: WARNING no srt for {mp4.name}, copying as-is")

    # original mp4 with subtitles applied
    raw_mp4 = workdir / "raw.mp4"
    raw_srt = workdir / "raw.srt"
    if subs == "embed":
        apply_subs(raw_mp4, raw_srt, outdir / "original-with-subs.mp4", "embed")
    elif subs == "burn":
        apply_subs(raw_mp4, raw_srt, outdir / "original-with-subs.mp4", "burn")
    else:
        shutil.copy(raw_mp4, outdir / "original-with-subs.mp4")
        shutil.copy(raw_srt, outdir / "original-with-subs.srt")
    log(f"edit: original-with-subs.mp4 written to {outdir}")
    (workdir / ".edit-done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="YouTube URL or 11-char video ID")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="working/output dir (default: ./work/<video_id>)")
    ap.add_argument("--height", type=int, default=1080,
                    help="max video height for download (0 = best available)")
    ap.add_argument("--language", default=None,
                    help="ISO 639-1 language code for transcription (default: auto-detect)")
    ap.add_argument("--transcribe-model", default=DEFAULT_TRANSCRIBE_MODEL)
    ap.add_argument("--segment-model", default=DEFAULT_SEGMENT_MODEL,
                    help="pi model for segmentation (Mistral Large line; literal Large 3 id: mistral/mistral-large-2512)")
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    help="pi thinking/effort level for segmentation")
    ap.add_argument("--chunk-minutes", type=int, default=API_MAX_CHUNK_MINUTES,
                    help="audio chunk size for transcription (API max is 60)")
    ap.add_argument("--subs", choices=["embed", "burn", "sidecar"], default="embed",
                    help="how to apply subtitles to outputs (default: embed as mov_text)")
    ap.add_argument("--reencode", action="store_true",
                    help="re-encode segments in ffmpeg-editlist (seamless cuts, slower)")
    ap.add_argument("--no-combined", action="store_true",
                    help="do not emit the combined full-nobreaks.mp4 output")
    ap.add_argument("--pi-bin", default=os.environ.get("PI_BIN", "pi"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-transcribe", action="store_true")
    ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--skip-edit", action="store_true",
                    help="skip the edit stage even if not done yet")
    ap.add_argument("--force-edit", action="store_true",
                    help="redo the edit stage even if the .edit-done marker exists")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="per-chunk transcription retry limit (5xx/429/network errors)")
    ap.add_argument("--retry-base-seconds", type=float, default=2.0,
                    help="exponential backoff base for transcription retries")
    ap.add_argument("--retry-max-seconds", type=float, default=120.0,
                    help="cap for the backoff delay between retries")
    ap.add_argument("--mock-transcript", type=Path, default=None,
                    help="testing: JSON file with {'segments': [...]} instead of API call")
    ap.add_argument("--mock-segments", type=Path, default=None,
                    help="testing: JSON file with the intermediate segments format")
    args = ap.parse_args()

    vid = video_id_from_url(args.url)
    workdir = args.workdir or Path("work") / vid
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"workdir: {workdir.resolve()}")

    if not args.skip_download:
        step_download(args.url, workdir, args.height or None)
    else:
        log("download: skipped")

    if not args.skip_transcribe:
        step_transcribe(workdir, args.transcribe_model, args.language,
                        args.chunk_minutes, args.mock_transcript,
                        max_retries=args.max_retries, retry_base=args.retry_base_seconds,
                        retry_max=args.retry_max_seconds)
    else:
        log("transcribe: skipped")

    if not args.skip_segment:
        data = step_segment(workdir, args.segment_model, args.effort,
                            args.pi_bin, args.mock_segments)
    else:
        log("segment: skipped")
        data = json.loads((workdir / "segments.json").read_text())

    if args.skip_edit:
        log("edit: skipped (--skip-edit)")
    elif (workdir / ".edit-done").exists() and not args.force_edit:
        log("edit: .edit-done marker exists, skipping (use --force-edit to redo)")
    else:
        step_edit(workdir, data, args.subs, args.reencode, combined=not args.no_combined)
    log(f"DONE. Outputs in {(workdir / 'out').resolve()}")


if __name__ == "__main__":
    main()
