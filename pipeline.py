#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "yt-dlp",
#   "requests",
# ]
# ///
"""
YouTube -> subtitles -> hierarchical segmentation -> keyframe cuts -> aligned mp4.

Pipeline stages (resumable: re-run the same command after a failure — finished
downloads, per-chunk transcription results (workdir/chunks/), segmentation and
edit stages are all cached/skipped automatically):
  - transcription chunks retry 5xx/429/network errors with exponential backoff
    (--max-retries, --retry-base-seconds, --retry-max-seconds); stages can also be
    skipped explicitly via --skip-download/--skip-transcribe/--skip-segment/--skip-edit.
  - the segment pi oneshot catches common pi failures and classifies them from the
    process output + its session transcript: fatal causes (out of API money/402,
    auth errors, prompt too large, unknown model, bad request) exit immediately with
    an informative message, anything else (API 5xx, 429, network errors, the run
    being terminated/killed, unknown random failures) is retried up to 3 total
    attempts (PI_MAX_ATTEMPTS) before dying with the diagnostics.
  1. download    : yt-dlp grabs the YouTube video as raw.mp4 (+ raw.info.json)
  2. transcribe  : silence-aware audio chunks -> Mistral Voxtral by default or
                   OpenAI Whisper -> transcript.json (segments) + raw.srt.
                   Chunks run from the first non-silent sample to the next silence of
                   at least --chunk-silence-min-seconds (30 by default), so no chunk
                   contains a long silence: a provider compresses those (a real 347.7 s
                   silence came back as 77.9 s) and its timestamps then drift by
                   minutes. Regions still over --chunk-minutes are split into the
                   fewest equal pieces that fit, overlapping by
                   --chunk-overlap-seconds. Each chunk's transcript is compared with
                   its own audio (workdir/transcript-coverage.txt,
                   --no-transcript-coverage-check) so a drifting provider is reported.
  3. segment     : `pi` oneshot (Mistral Large by default or OpenAI GPT-6.1 Terra,
                   medium effort) over the transcript,
                   run in a bubblewrap read-only sandbox when available (--sandbox;
                   the workdir is read-only except agent-scratch/, so the agent cannot
                   clobber pipeline files). pi always runs with a BARE config:
                   PI_CODING_AGENT_DIR points at a fresh empty dir in agent-scratch,
                   so the user's ~/.pi/agent (settings, extensions, custom models,
                   auth) is never read -- same as inside the Docker image; the API
                   key comes from MISTRAL_API_KEY or OPENAI_API_KEY
                   (transient failures are retried up to 3 attempts; fatal ones like
                   an out-of-balance API exit immediately with an informative error)
                   -> segments.json (intermediate format: segments > sections; days are
                   never guessed -- with --config each video is one day given by the
                   config entry's title, single-video runs get no day level)
  4. edit        : segments.json -> keyframe-exact cuts -> mp4 + rebased .srt
                   -> per-segment mp4 outputs with break sections removed,
                      plus the original mp4 with subtitles applied.
                   Each range is cut with -c copy from the exact PTS of the nearest
                   keyframe (raw.mp4 is probed once, right after download, cached in
                   workdir/keyframes.json; the cache is fingerprint-bound to raw.mp4,
                   validated on read and written atomically, so a stale/corrupt cache
                   is re-probed, not fatal). Snapping to the NEAREST keyframe rather
                   than always forwards avoids losing up to a GOP per cut
                   (--no-snap-keyframes disables snapping). ffmpeg seeks with -ss as an
                   INPUT option, which lands on the keyframe at or before the request;
                   an output-side -ss lands on the NEXT keyframe and loses content. The
                   parts are joined with the concat demuxer declaring each part's
                   duration explicitly -- without that it advances by the container
                   duration (the longer track), so audio creeps ahead of video and of
                   the .srt at every join. The .srt is rebased from the ranges actually
                   cut, and a cue broken by a cut is clipped and merged, not emitted
                   twice. A sync report (out/sync-report.txt) verifies per output that
                   the video timeline matches the one the .srt was rebased onto.
  5. align       : post-edit verification and repair. At --align-samples points
                   (8 by default) the output's loudness envelope is cross-correlated
                   against raw.mp4 over +/-align-search-seconds (3 by default) to find
                   where its sound really came from, and a 1 s burst of downscaled
                   greyscale frames is located the same way for the picture.
                   Untrustworthy measurements (silence, a static slide where every
                   offset ties) are rejected, not guessed. The sidecar .srt is shifted
                   by the interpolated lag, cancelling whatever residual the cut and the
                   transcript leave, and the measurements go to out/align-report.txt
                   (--no-align-check disables the pass, --no-align-repair measures
                   without rewriting the .srt).

Transcripts & API accounting:
  - the pi oneshot of stage 3 runs with --session, so pi's transcript is persisted:
    workdir/pi-session.jsonl (pi's native JSONL session, written incrementally, so it
    survives even a crashed run; inside the bwrap sandbox it is written to the writable
    agent-scratch/ and copied out afterwards) and workdir/pi-transcript.json (a plain-JSON
    extraction of it: messages with text/tool calls/tool results, per-message usage, and
    summed token+cost totals).
  - every API request is recorded as one row in an api-usage.csv ledger, itemized per
    day/step: config mode keeps one CSV at the workdir root (workdir_root/api-usage.csv,
    rows carry a day column); single-video mode keeps it in the video's workdir
    (workdir/api-usage.csv). Columns: timestamp, day, step, detail, model, requests,
    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_tokens,
    audio_seconds, cost_usd. Costs come from the API usage (transcription: billed per
    audio hour; pi segment run: pi's own catalog-derived cost in the session) with
    env-overridable list prices as fallback (MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR,
    MISTRAL_USD_PER_1M_INPUT/OUTPUT/CACHE_READ/CACHE_WRITE). After any command the
    pipeline prints the API usage of the run and the ledger total.

Usage:
  uv run --script pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" [options]
  uv run --script pipeline.py --config event.toml [options]

With --config, the TOML file lists the day videos of an event as [[days]] entries
(each with url= or path=, plus an optional title= that names the day); each video
is processed in its own workdir under workdir_root and treated as that one day.
See config.example.toml.

Requires on PATH: ffmpeg, ffprobe, pi, imagemagick (compare; lets the segment model
fine-tune cut points visually) (and node as yt-dlp JS runtime, optional).
Requires env: MISTRAL_API_KEY and/or OPENAI_API_KEY for the providers in use.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import requests

MISTRAL_API_BASE = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
OPENAI_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
DEFAULT_TRANSCRIBE_MODEL = "voxtral-small-transcribe-3"   # Voxtral Small Transcribe 3 ("Voxtral-3-small")
DEFAULT_SEGMENT_MODEL = "mistral/mistral-large-latest"     # Mistral Large line (Large 3 = mistral-large-2512; pin via --segment-model)
DEFAULT_EFFORT = "medium"
API_MAX_CHUNK_MINUTES = 55  # API limit is 60 min / 500 MB per request; stay under
OPENAI_AUDIO_MAX_BYTES = 25_000_000
OPENAI_TRANSCRIBE_MODEL = "whisper-1"
OPENAI_SEGMENT_MODEL = "openai/gpt-6.1-terra"
# Silence analysis (ffmpeg silencedetect) shared by silence-aware chunking and the
# post-transcription check. -60 dBFS is well below speech, so a detected silence is
# genuinely empty rather than a quiet passage.
DEFAULT_SILENCE_DBFS = -60.0
DEFAULT_SILENCE_MIN_SECONDS = 0.5   # finest granularity we care about
DEFAULT_CHUNK_SILENCE_MIN_SECONDS = 30.0  # long enough to hold no speech at all
DEFAULT_CHUNK_OVERLAP_SECONDS = 60.0
PROVIDER_API_KEYS = {"mistral": "MISTRAL_API_KEY", "openai": "OPENAI_API_KEY"}
PROVIDER_DEFAULTS = {
    "mistral": {
        "transcribe": {"model": DEFAULT_TRANSCRIBE_MODEL},
        "segment": {"model": DEFAULT_SEGMENT_MODEL, "effort": DEFAULT_EFFORT},
    },
    "openai": {
        "transcribe": {"model": OPENAI_TRANSCRIBE_MODEL},
        "segment": {"model": OPENAI_SEGMENT_MODEL, "effort": DEFAULT_EFFORT},
    },
}

# List prices used to turn API usage into USD (the APIs return usage, not cost).
# pi reports its own catalog-derived cost in the session transcript; these are the
# fallback for chat tokens and the primary source for transcription (billed per audio
# hour). Override via env if Mistral changes pricing.
DEFAULT_TRANSCRIBE_USD_PER_AUDIO_HOUR = 0.24   # voxtral-small-transcribe-3
DEFAULT_CHAT_USD_PER_1M = {"input": 2.0, "output": 6.0, "cache_read": 0.2, "cache_write": 0.0}  # Mistral Large line

PI_SESSION_NAME = "pi-session.jsonl"      # pi's native session transcript (JSONL) in the workdir
PI_TRANSCRIPT_NAME = "pi-transcript.json"  # plain-JSON extraction of it (messages + usage totals)
PI_AGENT_DIR_NAME = "pi-agent"             # bare pi config dir inside agent-scratch (PI_CODING_AGENT_DIR)
PI_MAX_ATTEMPTS = 3                        # total attempts of the segment pi oneshot before dying
USAGE_CSV_NAME = "api-usage.csv"
USAGE_CSV_COLUMNS = ["timestamp", "day", "step", "detail", "model", "requests",
                     "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                     "total_tokens", "audio_seconds", "cost_usd"]


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
# API accounting (usage ledger CSV) and pi transcript extraction
# --------------------------------------------------------------------------- #

def chat_price_per_1m(kind: str) -> float:
    """USD per 1M tokens for a chat-usage kind (input/output/cache_read/cache_write),
    env-overridable via MISTRAL_USD_PER_1M_<KIND>."""
    return float(os.environ.get(f"MISTRAL_USD_PER_1M_{kind.upper()}",
                                DEFAULT_CHAT_USD_PER_1M[kind]))


def transcribe_usage_fields(usage: dict, provider: str = "mistral") -> dict:
    """Map provider transcription usage to the ledger's counters."""
    if provider == "openai":
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "audio_seconds": float(usage.get("seconds") or 0),
        }
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "audio_seconds": float(usage.get("prompt_audio_seconds") or 0.0),
    }


def accumulate_usage(total: dict, usage: dict | None) -> None:
    """Add the scalar counters of one API usage dict into a running total.

    The API may nest objects inside usage (e.g. token details); only numbers are
    summed, so nested objects are skipped instead of breaking the addition."""
    for k, v in (usage or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        total[k] = total.get(k, 0) + v


def transcribe_cost_usd(usage: dict, provider: str = "mistral") -> float:
    """Estimate transcription cost from audio duration, with a Mistral token fallback."""
    if provider == "openai":
        per_hour = float(os.environ.get("OPENAI_TRANSCRIBE_USD_PER_AUDIO_HOUR", "0.36"))
        return float(usage.get("seconds") or 0) / 3600.0 * per_hour
    per_hour = float(os.environ.get("MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR",
                                    DEFAULT_TRANSCRIBE_USD_PER_AUDIO_HOUR))
    secs = usage.get("prompt_audio_seconds")
    if secs:
        return float(secs) / 3600.0 * per_hour
    return ((usage.get("prompt_tokens") or 0) * chat_price_per_1m("input")
            + (usage.get("completion_tokens") or 0) * chat_price_per_1m("output")) / 1e6


def estimate_chat_cost_usd(totals: dict, provider: str = "mistral",
                           model: str = DEFAULT_SEGMENT_MODEL) -> float:
    """Fallback USD cost of pi usage totals (pi usually reports cost itself in the
    session; this is for sessions without it)."""
    if provider == "openai":
        if model.removeprefix("openai/") != "gpt-6.1-terra":
            return 0.0  # other models use the cost reported by pi
        return (totals.get("input", 0) + totals.get("output", 0) * 5
                + totals.get("cacheRead", 0) * 0.05
                + totals.get("cacheWrite", 0) * 1.25) / 1e6
    if provider != "mistral":
        return 0.0
    return (totals.get("input", 0) * chat_price_per_1m("input")
            + totals.get("output", 0) * chat_price_per_1m("output")
            + totals.get("cacheRead", 0) * chat_price_per_1m("cache_read")
            + totals.get("cacheWrite", 0) * chat_price_per_1m("cache_write")) / 1e6


def pi_usage_fields(totals: dict) -> dict:
    """pi session usage totals -> accounting fields."""
    return {
        "input_tokens": int(totals.get("input") or 0),
        "output_tokens": int(totals.get("output") or 0),
        "cache_read_tokens": int(totals.get("cacheRead") or 0),
        "cache_write_tokens": int(totals.get("cacheWrite") or 0),
        "total_tokens": int(totals.get("totalTokens") or 0),
    }


def _sum_rows(rows: list[dict]) -> dict:
    keys = ["requests", "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "total_tokens", "audio_seconds", "cost_usd"]
    out = {k: 0.0 for k in keys}
    for r in rows:
        for k in keys:
            out[k] += float(r.get(k) or 0)
    return out


def _fmt_totals(t: dict) -> str:
    s = (f"{int(t['requests'])} req, tokens in/out/total "
         f"{int(t['input_tokens'])}/{int(t['output_tokens'])}/{int(t['total_tokens'])}")
    if t["cache_read_tokens"] or t["cache_write_tokens"]:
        s += f" (cache r/w {int(t['cache_read_tokens'])}/{int(t['cache_write_tokens'])})"
    if t["audio_seconds"]:
        s += f", audio {t['audio_seconds']:.0f}s"
    s += f", ${t['cost_usd']:.4f}"
    return s


class Accounting:
    """Append-only ledger of API spend: one CSV row per API request, itemized per
    day/step. Also keeps this run's rows in memory for the end-of-command report.

    In config mode the CSV lives at the workdir root (one per event, rows carry the
    day); in single-video mode in the video's workdir."""

    def __init__(self, csv_path: Path, day: str):
        self.csv_path = Path(csv_path)
        self.day = day
        self.rows: list[dict] = []

    def record(self, *, step: str, detail: str, model: str, requests: int = 1,
               input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0,
               cache_write_tokens: int = 0, total_tokens: int = 0, audio_seconds: float = 0.0,
               cost_usd: float = 0.0) -> dict:
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "day": self.day, "step": step,
               "detail": detail, "model": model, "requests": int(requests),
               "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
               "cache_read_tokens": int(cache_read_tokens), "cache_write_tokens": int(cache_write_tokens),
               "total_tokens": int(total_tokens), "audio_seconds": round(float(audio_seconds), 3),
               "cost_usd": round(float(cost_usd), 6)}
        self.rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=USAGE_CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"api: [{self.day}] {step} {detail} model={model} {_fmt_totals(_sum_rows([row]))} "
            f"-> {self.csv_path}")
        return row

    def run_totals(self) -> dict:
        return _sum_rows(self.rows)


def ledger_rows(csv_path: Path) -> list[dict]:
    """All rows recorded so far in a usage CSV (empty if there is none yet)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("step")]


def report_api_usage(accts: list["Accounting"]) -> None:
    """Print the API usage of this run plus the ledger totals of each usage CSV.
    Called after any command (even a failed one) from main()'s finally block."""
    if not accts:
        return
    log("=== API usage this run ===")
    for acct in accts:
        if not acct.rows:
            log(f"  [{acct.day}] no new API calls (stages cached/skipped, or mocked)")
            continue
        groups: dict[str, list[dict]] = {}
        for r in acct.rows:
            groups.setdefault(r["step"], []).append(r)
        for step, rows in groups.items():
            log(f"  [{acct.day}] {step}: {_fmt_totals(_sum_rows(rows))}")
        log(f"  [{acct.day}] TOTAL: {_fmt_totals(acct.run_totals())}")
    log("=== total API usage (ledger) ===")
    by_csv: dict[Path, list[dict]] = {}
    for csv_path in dict.fromkeys(acct.csv_path for acct in accts):
        if csv_path.exists():
            by_csv[csv_path] = ledger_rows(csv_path)
    if not by_csv:
        log("  (no api-usage.csv written yet -- no API calls were made)")
    for csv_path, rows in by_csv.items():
        t = _sum_rows(rows)
        days = sorted({r.get("day", "") for r in rows if r.get("day")})
        log(f"  TOTAL {csv_path}: {_fmt_totals(t)}"
            + (f" across day(s): {', '.join(days)}" if days else ""))


def extract_pi_transcript(session_path: Path) -> dict:
    """Parse a pi session file (JSONL, one entry per line) into a plain-JSON transcript:
    session metadata, provider/model, the message list (text, thinking, tool calls and
    tool results; assistant messages carry their usage), standalone usage entries, and
    summed usage totals (tokens + cost)."""
    entries = []
    for line in Path(session_path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    header = next((e for e in entries if e.get("type") == "session"), {})
    provider = model = None
    messages: list[dict] = []
    other_usage: list[dict] = []
    totals: dict = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                     "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0,
                              "cacheWrite": 0.0, "total": 0.0}}
    for e in entries:
        etype = e.get("type")
        if etype == "model_change":
            provider = e.get("provider", provider)
            model = e.get("modelId", model)
        elif etype == "usage":
            other_usage.append({"kind": e.get("kind"), "provider": e.get("provider"),
                                "model": e.get("model"), "usage": e.get("usage")})
        elif etype == "message":
            m = e.get("message") or {}
            role = m.get("role")
            msg: dict = {"role": role, "timestamp": e.get("timestamp")}
            content = m.get("content")
            blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}]
                                                                if isinstance(content, str) else [])
            texts, thinking, tool_calls = [], [], []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    texts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinking.append(b.get("thinking", ""))
                elif bt == "toolCall":
                    tool_calls.append({"id": b.get("id"), "name": b.get("name"),
                                       "arguments": b.get("arguments")})
            if texts:
                msg["text"] = "\n".join(texts)
            if thinking:
                msg["thinking"] = "\n".join(thinking)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if role == "toolResult":
                msg["tool_call_id"] = m.get("toolCallId")
                msg["tool_name"] = m.get("toolName")
                msg["is_error"] = bool(m.get("isError"))
            if role == "assistant":
                msg["model"] = m.get("model")
                msg["provider"] = m.get("provider")
                msg["stop_reason"] = m.get("stopReason")
                if m.get("errorMessage"):
                    msg["error"] = m["errorMessage"]
                u = m.get("usage") or {}
                msg["usage"] = u
                for k in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
                    totals[k] += u.get(k) or 0
                c = u.get("cost") or {}
                for k in totals["cost"]:
                    totals["cost"][k] += c.get(k) or 0.0
                model = model or m.get("model")
                provider = provider or m.get("provider")
            messages.append(msg)
    return {"session_id": header.get("id"), "session_version": header.get("version"),
            "started": header.get("timestamp"), "cwd": header.get("cwd"),
            "provider": provider, "model": model,
            "messages": messages, "other_usage": other_usage, "usage": totals}


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


def step_download_local(src: Path, workdir: Path, title: str | None = None) -> dict:
    """Config mode: use a local video file as the raw input (one pre-cut day)."""
    raw_mp4 = workdir / "raw.mp4"
    info_json = workdir / "raw.info.json"
    if raw_mp4.exists() and info_json.exists():
        log(f"download: raw.mp4 already exists in {workdir}, skipping")
        return json.loads(info_json.read_text())
    if not src.exists():
        sys.exit(f"ERROR: local video not found: {src}")
    log(f"download: local file {src} -> {raw_mp4}")
    shutil.copyfile(src, raw_mp4)
    info = {
        "id": slugify(src.stem),
        "title": title or src.stem,
        "uploader": "",
        "description": "",
        "duration": probe_duration(raw_mp4),
    }
    info_json.write_text(json.dumps(info, indent=2))
    log(f"download: {raw_mp4} ({raw_mp4.stat().st_size/1e6:.1f} MB), title={info['title']!r}, "
        f"duration={info['duration']}s")
    return info


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [which_or_die("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


def probe_keyframes(mp4: Path) -> list[list[float]]:
    """Video keyframes of an mp4 as [pts, dts] pairs (seconds), sorted by pts."""
    out = subprocess.run(
        [which_or_die("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,dts_time,flags", "-of", "csv=p=0", str(mp4)],
        check=True, capture_output=True, text=True).stdout
    kfs = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) == 3 and "K" in parts[2]:
            try:
                kfs.append([float(parts[0]), float(parts[1])])
            except ValueError:  # pts/dts can be "N/A"
                continue
    return sorted(kfs)


KF_CACHE_VERSION = 1
KF_FINGERPRINT_BYTES = 1 << 20  # first 1 MiB


def file_fingerprint(mp4: Path) -> dict:
    """Cheap identity of the source video a keyframe cache belongs to: size plus
    sha256 of the first 1 MiB. A re-download of a *different* video virtually
    always differs here; the same video re-downloaded byte-identical stays valid
    (and correctly so -- same keyframes)."""
    st = mp4.stat()
    h = hashlib.sha256()
    with mp4.open("rb") as f:
        h.update(f.read(KF_FINGERPRINT_BYTES))
    return {"size": st.st_size, "head_sha256": h.hexdigest()}


def write_keyframe_cache(path: Path, mp4: Path, kfs: list[list[float]]) -> None:
    """Write the keyframe cache atomically (tmp file + rename), so an interrupted
    run can never leave a half-written cache behind."""
    payload = {"version": KF_CACHE_VERSION, "source": mp4.name,
               "fingerprint": file_fingerprint(mp4), "keyframes": kfs}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def load_keyframe_cache(path: Path, mp4: Path) -> list[list[float]] | None:
    """Cached keyframes if `path` is a valid cache for the current `mp4`, else None.

    A corrupt or foreign cache file (e.g. raw ffprobe CSV that the segment agent
    redirected into it) is moved aside to <name>.bad-<timestamp> and None is
    returned, so the caller re-probes raw.mp4 instead of crashing. A well-formed
    cache whose fingerprint does not match the current raw.mp4 is stale and also
    ignored (it gets overwritten by the next write)."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        raw = payload["keyframes"]
        if payload.get("version") != KF_CACHE_VERSION:
            raise ValueError(f"unsupported cache version {payload.get('version')!r}")
        if payload.get("fingerprint") != file_fingerprint(mp4):
            log(f"cache: {path.name} was probed from a different raw.mp4 "
                f"(fingerprint mismatch) -- ignoring it")
            return None
        if not isinstance(raw, list) or not raw:
            raise ValueError("'keyframes' is not a non-empty list")
        kfs = []
        for kf in raw:
            # tolerate an older cache that stored bare pts values instead of [pts, dts]
            pair = kf if isinstance(kf, list) else [kf, kf]
            if (len(pair) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               for v in pair)):
                raise ValueError(f"bad keyframe entry {kf!r}")
            kfs.append([float(pair[0]), float(pair[1])])
        return kfs
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        bad = path.with_name(f"{path.name}.bad-{int(time.time())}")
        log(f"WARNING: {path.name} is not a valid keyframe cache ({e}); "
            f"moving it aside to {bad.name} and re-probing {mp4.name}")
        try:
            os.replace(path, bad)
        except OSError:
            pass
        return None


def ensure_keyframe_cache(workdir: Path) -> None:
    """Probe raw.mp4 keyframes once, right after download, so a valid
    keyframes.json already exists before the segment agent runs (it only needs
    to READ it). The edit stage reuses the same cache."""
    raw_mp4 = workdir / "raw.mp4"
    kf_cache = workdir / "keyframes.json"
    if not raw_mp4.exists():
        return
    kfs = load_keyframe_cache(kf_cache, raw_mp4)
    if kfs is not None:
        log(f"keyframes: loaded {len(kfs)} keyframes from {kf_cache.name}")
        return
    log("keyframes: probing video keyframes of raw.mp4 (cached in keyframes.json)")
    kfs = probe_keyframes(raw_mp4)
    write_keyframe_cache(kf_cache, raw_mp4, kfs)
    log(f"keyframes: {len(kfs)} keyframes -> {kf_cache.name}")


def probe_audio_silences(mp4: Path, noise_dbfs: float = DEFAULT_SILENCE_DBFS,
                         min_seconds: float = DEFAULT_SILENCE_MIN_SECONDS) -> list[list[float]]:
    """Silent stretches of the audio as [start, end] pairs (seconds).

    Uses ffmpeg's `silencedetect`, which reports a stretch once the decoded level
    stays below `noise_dbfs` for `min_seconds`; shorter gaps are not reported.
    A silence running into either end of the file is clamped to [0, duration].
    """
    try:
        proc = subprocess.run(
            [which_or_die("ffmpeg"), "-nostats", "-hide_banner", "-i", str(mp4),
             "-vn", "-af", f"silencedetect=noise={noise_dbfs}dB:d={min_seconds}",
             "-f", "null", "-"],
            capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"WARNING: could not analyse audio silences of {mp4.name} ({e})")
        return []
    if proc.returncode != 0:
        log(f"WARNING: ffmpeg silencedetect failed on {mp4.name} "
            f"(no audio stream?); treating the whole file as speech")
        return []
    sil: list[list[float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and start is not None:
            sil.append([max(0.0, start), float(m.group(1))])
            start = None
    if start is not None:  # trailing silence that never got its silence_end
        sil.append([max(0.0, start), probe_duration(mp4)])
    return sil


SILENCE_CACHE_VERSION = 1


def write_silence_cache(path: Path, mp4: Path, sil: list[list[float]],
                        params: dict) -> None:
    """Write the silence cache atomically (tmp file + rename), like the keyframe one."""
    payload = {"version": SILENCE_CACHE_VERSION, "source": mp4.name,
               "fingerprint": file_fingerprint(mp4), "params": params, "silences": sil}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def load_silence_cache(path: Path, mp4: Path, params: dict) -> list[list[float]] | None:
    """Cached silences if `path` is a valid cache for this `mp4` and these analysis
    parameters, else None. A corrupt or foreign file is moved aside to
    `<name>.bad-<timestamp>` and None is returned, so the caller re-probes instead
    of crashing."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("version") != SILENCE_CACHE_VERSION:
            raise ValueError(f"unsupported cache version {payload.get('version')!r}")
        if payload.get("fingerprint") != file_fingerprint(mp4):
            log(f"cache: {path.name} was probed from a different raw.mp4 "
                f"(fingerprint mismatch) -- ignoring it")
            return None
        if payload.get("params") != params:
            log(f"cache: {path.name} was probed with different silence settings "
                f"-- ignoring it")
            return None
        raw = payload["silences"]
        if not isinstance(raw, list):
            raise ValueError("'silences' is not a list")
        out = []
        for pair in raw:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               for v in pair)):
                raise ValueError(f"bad silence entry {pair!r}")
            out.append([float(pair[0]), float(pair[1])])
        return out
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        bad = path.with_name(f"{path.name}.bad-{int(time.time())}")
        log(f"WARNING: {path.name} is not a valid silence cache ({e}); "
            f"moving it aside to {bad.name} and re-probing {mp4.name}")
        try:
            os.replace(path, bad)
        except OSError:
            pass
        return None


def silence_params(noise_dbfs: float, min_seconds: float) -> dict:
    return {"noise_dbfs": float(noise_dbfs), "min_seconds": float(min_seconds)}


def ensure_silence_cache(workdir: Path, noise_dbfs: float,
                         min_seconds: float) -> list[list[float]]:
    """Probe raw.mp4's audio silences once, cached in workdir/silences.json. Used by
    silence-aware chunking and by the post-transcription coverage check."""
    raw_mp4 = workdir / "raw.mp4"
    cache = workdir / "silences.json"
    if not raw_mp4.exists():
        return []
    params = silence_params(noise_dbfs, min_seconds)
    sil = load_silence_cache(cache, raw_mp4, params)
    if sil is not None:
        log(f"silences: loaded {len(sil)} from {cache.name}")
        return sil
    log(f"silences: analysing audio of raw.mp4 (noise {noise_dbfs} dBFS, "
        f"min {min_seconds}s; cached in silences.json)")
    sil = probe_audio_silences(raw_mp4, noise_dbfs, min_seconds)
    write_silence_cache(cache, raw_mp4, sil, params)
    log(f"silences: {len(sil)} silent stretches -> {cache.name}")
    return sil


def probe_output_timeline(mp4: Path) -> tuple[dict[str, float], float]:
    """Per-codec start_time of an output mp4 + video stream duration."""
    out = subprocess.run(
        [which_or_die("ffprobe"), "-v", "error",
         "-show_entries", "stream=codec_type,start_time,duration",
         "-of", "default=nw=1", str(mp4)],
        check=True, capture_output=True, text=True).stdout
    starts: dict[str, float] = {}
    video_dur = 0.0
    codec: str | None = None
    for line in out.splitlines():
        if line.startswith("codec_type="):
            codec = line.split("=", 1)[1]
        elif line.startswith("start_time=") and codec is not None:
            starts[codec] = float(line.split("=", 1)[1])
        elif line.startswith("duration=") and codec == "video":
            video_dur = max(video_dur, float(line.split("=", 1)[1]))
    return starts, video_dur


def probe_video_range_starts(mp4: Path, nominal_cums: list[float], gap: float = 0.25) -> list[float]:
    """Actual start pts of each edit range in the output video timeline: the first
    video pts plus every pts jump > `gap` (a stream-copied cut leaves a hole where the
    next range starts at a later keyframe).  Matched positionally to the nominal
    cumulative range starts `nominal_cums`."""
    out = subprocess.run(
        [which_or_die("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(mp4)],
        check=True, capture_output=True, text=True).stdout
    pts = sorted({float(x) for x in out.split() if x.strip()})
    if not pts:
        return list(nominal_cums)
    cands = [pts[0]]
    for prev, cur in zip(pts, pts[1:]):
        if cur - prev > gap:
            cands.append(cur)
    if len(cands) == len(nominal_cums):
        return cands
    # fallback: walk both lists in order; a nominal start with no detected jump
    # nearby (sub-threshold hole) maps to itself
    starts = []
    p = 0
    for c in nominal_cums:
        while p < len(cands) and cands[p] < c - 0.5:
            p += 1
        if p < len(cands) and cands[p] <= c + 30:
            starts.append(cands[p])
        else:
            starts.append(c)
    return starts


def srt_first_cue(path: Path) -> float | None:
    m = re.search(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", path.read_text())
    if not m:
        return None
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


# --------------------------------------------------------------------------- #
# Stage 2: transcribe (Voxtral-3-small via Mistral API, chunked)
# --------------------------------------------------------------------------- #

def extract_audio_chunk(raw_mp4: Path, start: float, dur: float, out: Path) -> None:
    run([which_or_die("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(raw_mp4),
         "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-c:a", "libmp3lame", str(out)])


def transcribe_chunk(mp3: Path, model: str, language: str | None, api_key: str,
                     max_retries: int = 5, retry_base: float = 2.0,
                     retry_max: float = 120.0, timeout: int = 3600, *,
                     provider: str = "mistral") -> tuple[list[dict], dict]:
    """POST one audio chunk to /v1/audio/transcriptions, retrying 5xx/429/network
    errors with exponential backoff + jitter. 4xx (other than 429) is fatal.
    Returns (segments, usage) -- the API's usage dict (audio seconds + tokens)."""
    if provider == "openai":
        if mp3.stat().st_size >= OPENAI_AUDIO_MAX_BYTES:
            sys.exit(f"ERROR: OpenAI audio upload {mp3.name} must be under 25 MB. "
                     "Reduce chunk_minutes or --chunk-minutes.")
        api_base = OPENAI_API_BASE
        data = {"model": model, "response_format": "verbose_json",
                "timestamp_granularities[]": "segment"}
    else:
        api_base = MISTRAL_API_BASE
        data = {"model": model, "timestamp_granularities": "segment", "stream": "false"}
    if language:
        data["language"] = language
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            with open(mp3, "rb") as fh:
                files = {"file": (mp3.name, fh, "audio/mpeg")}
                r = requests.post(f"{api_base}/audio/transcriptions",
                                  headers={"Authorization": f"Bearer {api_key}"},
                                  files=files, data=data, timeout=timeout)
            if r.status_code == 200:
                body = r.json()
                usage = body.get("usage") or {}
                if provider == "openai":
                    if not isinstance(body.get("segments"), list):
                        sys.exit(f"ERROR: OpenAI model {model!r} did not return timestamped segments.")
                    if not usage.get("seconds") and body.get("duration"):
                        usage["seconds"] = body["duration"]
                segs = body.get("segments") or []
                out = [{"start": float(s["start"]), "end": float(s["end"]),
                        "text": s.get("text", ""), "speaker_id": s.get("speaker_id")}
                       for s in segs]
                log(f"  chunk {mp3.name}: {len(out)} segments, usage={usage}")
                return out, usage
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


_SRT_TIME = re.compile(r"(-?\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_SRT_CUE = re.compile(
    r"(-?\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(-?\d+:\d{2}:\d{2}[,.]\d{1,3})")


def srt_seconds(stamp: str) -> float:
    m = _SRT_TIME.search(stamp)
    if not m:
        raise ValueError(f"not an SRT timestamp: {stamp!r}")
    h, mi, s, ms = m.groups()
    sign = -1.0 if h.startswith("-") else 1.0
    return sign * (abs(int(h)) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0)


def parse_srt(path: Path) -> list[dict]:
    """Cues of an SRT file as [{start, end, text}] (seconds), tolerant of BOM, CRLF
    and blank lines. Cues whose timing line cannot be parsed are skipped."""
    cues: list[dict] = []
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _SRT_CUE.search(line)
            if not m:
                continue
            try:
                start, end = srt_seconds(m.group(1)), srt_seconds(m.group(2))
            except ValueError:
                break
            body = " ".join(part.strip() for part in lines[i + 1:]).strip()
            cues.append({"start": start, "end": end, "text": body})
            break
    return cues


def compose_srt(cues: list[dict]) -> str:
    """SRT text for [{start, end, text}]; identical formatting to write_srt()."""
    blocks = []
    for i, c in enumerate(cues, 1):
        text = re.sub(r"\s+", " ", c["text"]).strip()
        blocks.append(f"{i}\n{srt_ts(c['start'])} --> {srt_ts(c['end'])}\n{text}\n")
    return "\n".join(blocks)


def write_srt(segments: list[dict], path: Path) -> None:
    path.write_text(compose_srt(segments))


def write_stage_json(path: Path, data: dict) -> None:
    """Publish a completed stage or chunk only after its JSON is written."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class AudioChunk:
    """One audio range sent to the transcription API.

    `drop_before` is the absolute time before which returned segments belong to the
    previous chunk's overlap tail; step_transcribe discards them so a phrase spoken
    inside the overlap is not emitted twice. It equals `start` when this chunk does
    not overlap its predecessor (which is always the case across a long silence).
    """

    __slots__ = ("start", "duration", "drop_before")

    def __init__(self, start: float, duration: float, drop_before: float):
        self.start = float(start)
        self.duration = float(duration)
        self.drop_before = float(drop_before)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def __repr__(self) -> str:  # keeps log lines short
        return f"AudioChunk({self.start:.1f}, +{self.duration:.1f})"


def _split_region(start: float, end: float, max_seconds: float,
                  overlap_seconds: float) -> list[AudioChunk]:
    """Split [start, end] into the fewest equal pieces that fit under `max_seconds`,
    each overlapping the next by `overlap_seconds`."""
    length = end - start
    if length <= max_seconds:
        return [AudioChunk(start, length, start)]
    span = max_seconds - overlap_seconds
    n = max(2, math.ceil((length - overlap_seconds) / span - 1e-9))
    while (length + (n - 1) * overlap_seconds) / n > max_seconds + 1e-9:
        n += 1
    piece = (length + (n - 1) * overlap_seconds) / n
    step = piece - overlap_seconds
    chunks = []
    for i in range(n):
        s = start + i * step
        if i == n - 1:      # anchor the tail so rounding cannot overshoot the end
            s = max(start, end - piece)
        chunks.append(AudioChunk(s, piece, s + overlap_seconds if i else s))
    return chunks


def build_audio_chunks(duration: float, silences: list[list[float]],
                       max_seconds: float,
                       long_silence_seconds: float = DEFAULT_CHUNK_SILENCE_MIN_SECONDS,
                       overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS
                       ) -> list[AudioChunk]:
    """Transcription chunks that contain no long silence.

    A transcription provider's segment timestamps are only reliable while the audio
    is contiguous speech: given a long stretch of silence inside one request they
    compress or stretch the timeline instead of tracking it, which silently shifts
    every subtitle after that point. So a chunk starts at the first non-silent
    moment and stops where the next silence of at least `long_silence_seconds`
    begins; the following chunk resumes at the end of that silence. The silence is
    therefore never uploaded, and no chunk straddles one. Shorter pauses stay
    inside their chunk.

    A contiguous speech region that is still longer than `max_seconds` is split into
    equal overlapping pieces, since there is no silence to split on.
    """
    if duration <= 0:
        return []
    long_silences = sorted((max(0.0, s), min(duration, e)) for s, e in silences
                           if e - s >= long_silence_seconds and e > 0 and s < duration)
    regions: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in long_silences:
        if s > pos:
            regions.append((pos, s))
        pos = max(pos, e)
    if pos < duration:
        regions.append((pos, duration))
    chunks: list[AudioChunk] = []
    for s, e in regions:
        if e - s <= 0:
            continue
        chunks.extend(_split_region(s, e, max_seconds, overlap_seconds))
    if not chunks:
        # everything is silence: still send one chunk so we emit an empty transcript
        # rather than failing, and let the coverage check report the oddity
        chunks = [AudioChunk(0.0, duration, 0.0)]
    return chunks


def _intervals_union(intervals: list[list[float]]) -> list[list[float]]:
    """Merge overlapping/adjacent [start, end] intervals, dropping empty ones."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _intervals_minus(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """a minus b, both lists of disjoint [start, end] intervals."""
    out: list[list[float]] = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur:
                continue
            if bs >= e:
                break
            if bs > cur:
                out.append([cur, min(bs, e)])
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            out.append([cur, e])
    return out


def _intervals_intersect(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in a:
        for bs, be in b:
            lo, hi = max(s, bs), min(e, be)
            if hi > lo:
                out.append([lo, hi])
    return _intervals_union(out)


def _total_length(intervals: list[list[float]]) -> float:
    return sum(e - s for s, e in intervals)


# Coverage tolerances, calibrated against day-2 (the run whose subtitles were wrong):
# chunks with a correct transcript scored 0.0-2.3% of cue time on silence, the two
# drifted chunks 20% and 27%. `spurious` is the signal that discriminates, and it has a
# clean physical meaning: cue text on audio that is measurably empty. `missed` is
# reported but not gated on, because a provider rightly emits no cue for non-speech
# sound (typing, laughter, room tone above the threshold), which would otherwise look
# like drift. An outright empty transcript is caught separately.
COVERAGE_SILENCE_MIN_SECONDS = 3.0   # a cue sitting on 3s+ of real silence is an error
COVERAGE_SPURIOUS_MIN_SECONDS = 5.0
COVERAGE_SPURIOUS_FRACTION = 0.05
COVERAGE_EMPTY_SPEECH_SECONDS = 60.0  # this much speech with nothing cued is a failure


def chunk_coverage_report(idx: int, part: AudioChunk, raw_segs: list[dict],
                          silences: list[list[float]]) -> dict:
    """Compare what one chunk's audio contains with what the provider says is in it.

    Silence-aware chunking stops a provider from drifting across a long silence, but
    nothing forces it not to drift anyway. This is the cheap check that would have
    caught the day-2 failure without any forensics: a chunk is contiguous speech by
    construction, so cue text landing on a detected silence means the timestamps are
    wrong, and every subtitle after that point is shifted.
    """
    lo, hi = part.start, part.end
    cue = _intervals_union([[max(lo, lo + s["start"]), min(hi, lo + s["end"])]
                            for s in raw_segs if float(s["end"]) > float(s["start"])])
    silence = _intervals_union([[max(s, lo), min(e, hi)] for s, e in silences
                                if e - s >= COVERAGE_SILENCE_MIN_SECONDS])
    speech = _intervals_minus(
        [[lo, hi]],
        _intervals_union([[max(s, lo), min(e, hi)] for s, e in silences]))
    spurious = _total_length(_intervals_intersect(cue, silence))
    cue_seconds = _total_length(cue)
    speech_seconds = _total_length(speech)
    missed = _total_length(_intervals_minus(speech, cue))
    missed_fraction = missed / speech_seconds if speech_seconds > 0 else 0.0
    spurious_fraction = spurious / cue_seconds if cue_seconds > 0 else 0.0
    too_much_silence = spurious > max(COVERAGE_SPURIOUS_MIN_SECONDS,
                                      COVERAGE_SPURIOUS_FRACTION * cue_seconds)
    empty = speech_seconds >= COVERAGE_EMPTY_SPEECH_SECONDS and not cue_seconds
    return {"chunk": idx, "start": lo, "duration": part.duration,
            "segments": len(raw_segs), "speech": speech_seconds, "cue": cue_seconds,
            "spurious": spurious, "spurious_fraction": spurious_fraction,
            "missed": missed, "missed_fraction": missed_fraction,
            "ok": not too_much_silence and not empty}


def report_chunk_coverage(workdir: Path, coverage: list[dict]) -> None:
    """Write workdir/transcript-coverage.txt and warn about suspicious chunks."""
    lines = ["# per-chunk transcript coverage",
             "# spurious = seconds of cue text sitting on audio detected as silence",
             f"#            (silences shorter than {COVERAGE_SILENCE_MIN_SECONDS:.0f}s are ignored)",
             "# missed   = seconds of speech (chunk audio minus detected silence) with no",
             "#            cue; informational only, since non-speech sound is not cued"]
    for c in coverage:
        lines.append(
            f"chunk {c['chunk']:03d} [{c['start']:.1f}s +{c['duration']:.1f}s]: "
            f"{c['segments']} segments, speech {c['speech']:.0f}s, cue {c['cue']:.0f}s, "
            f"spurious {c['spurious']:.0f}s ({c['spurious_fraction'] * 100:.1f}%), "
            f"missed {c['missed']:.0f}s ({c['missed_fraction'] * 100:.0f}%)"
            + ("" if c["ok"] else "  <-- SUSPICIOUS"))
    path = workdir / "transcript-coverage.txt"
    path.write_text("\n".join(lines) + "\n")
    bad = [c for c in coverage if not c["ok"]]
    if bad:
        log(f"transcribe: WARNING {len(bad)} of {len(coverage)} chunk(s) look misaligned "
            f"against their own audio -- the provider's segment timestamps do not match "
            f"the timeline, so subtitles after them will be shifted "
            f"(see {path.name})")
        for c in bad:
            log(f"transcribe:   chunk {c['chunk']:03d} [{c['start']:.1f}s "
                f"+{c['duration']:.1f}s]: {c['spurious']:.0f}s of cue on silence "
                f"({c['spurious_fraction'] * 100:.1f}% of cue time), "
                f"{c['segments']} segment(s) for {c['speech']:.0f}s of speech")


def step_transcribe(workdir: Path, model: str, language: str | None,
                    chunk_minutes: int, mock_transcript: Path | None,
                    max_retries: int = 5, retry_base: float = 2.0,
                    retry_max: float = 120.0,
                    acct: Accounting | None = None, *,
                    provider: str = "mistral",
                    silence_dbfs: float = DEFAULT_SILENCE_DBFS,
                    silence_min_seconds: float = DEFAULT_SILENCE_MIN_SECONDS,
                    chunk_silence_min_seconds: float = DEFAULT_CHUNK_SILENCE_MIN_SECONDS,
                    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
                    coverage_check: bool = True) -> list[dict]:
    transcript_json = workdir / "transcript.json"
    raw_mp4 = workdir / "raw.mp4"
    source = mock_transcript if mock_transcript else raw_mp4
    chunking = {"max_seconds": chunk_minutes * 60,
                "long_silence_seconds": float(chunk_silence_min_seconds),
                "overlap_seconds": float(chunk_overlap_seconds),
                "silence_dbfs": float(silence_dbfs),
                "silence_min_seconds": float(silence_min_seconds)}
    provenance = {"provider": provider, "model": model, "language": language,
                  "mock": bool(mock_transcript), "chunking": chunking,
                  "source": file_fingerprint(source) if source.exists() else None}
    if transcript_json.exists():
        cached = json.loads(transcript_json.read_text())
        # No legacy fallback: a transcript cached before silence-aware chunking was
        # cut differently and its timestamps may be minutes out, so it must be redone.
        if cached.get("_cache") == provenance:
            log("transcribe: transcript.json matches, skipping")
            return cached["segments"]
        log("transcribe: transcript.json was produced with different settings "
            "(or predates silence-aware chunking)")
    total_usage: dict = {}
    if mock_transcript:
        log(f"transcribe: using mock transcript {mock_transcript}")
        segments = json.loads(mock_transcript.read_text())["segments"]
    else:
        api_key_env = PROVIDER_API_KEYS[provider]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            sys.exit(f"ERROR: {api_key_env} is not set (needed for /v1/audio/transcriptions)")
        duration = probe_duration(raw_mp4)
        silences = ensure_silence_cache(workdir, silence_dbfs, silence_min_seconds)
        chunks = build_audio_chunks(duration, silences, chunking["max_seconds"],
                                    chunking["long_silence_seconds"],
                                    chunking["overlap_seconds"])
        silent_total = sum(e - s for s, e in silences
                           if e - s >= chunking["long_silence_seconds"])
        log(f"transcribe: {len(chunks)} silence-aware chunk(s), "
            f"{duration - silent_total:.0f}s of {duration:.0f}s audio, "
            f"{silent_total:.0f}s of long silence skipped")
        cache_dir = workdir / "chunks"   # per-chunk API results -> resumable
        cache_dir.mkdir(exist_ok=True)
        all_segments: list[dict] = []
        coverage: list[dict] = []
        for idx, part in enumerate(chunks):
            start, dur = part.start, part.duration
            cache_file = cache_dir / f"chunk-{idx:03d}.json"
            cache_key = {**provenance, "start": start, "duration": dur}
            payload = json.loads(cache_file.read_text()) if cache_file.exists() else {}
            if payload.get("_cache") == cache_key:
                log(f"transcribe: chunk {idx} [{start:.0f}s +{dur:.0f}s] cached -> {cache_file.name}")
                raw_segs = payload["segments"]
                # usage of a cached chunk was already recorded in the ledger when the
                # API call was made; only accumulate it into the transcript totals
                accumulate_usage(total_usage, payload.get("usage"))
            else:
                chunk = workdir / f"chunk-{idx:03d}.mp3"
                log(f"transcribe: chunk {idx} [{start:.0f}s +{dur:.0f}s] of {duration:.0f}s")
                extract_audio_chunk(raw_mp4, start, dur, chunk)
                try:
                    raw_segs, usage = transcribe_chunk(
                        chunk, model, language, api_key, max_retries=max_retries,
                        retry_base=retry_base, retry_max=retry_max, provider=provider)
                finally:
                    chunk.unlink(missing_ok=True)
                if provider == "openai" and not usage.get("seconds"):
                    usage["seconds"] = dur
                if acct is not None:
                    acct.record(step="transcribe", model=model,
                                detail=f"chunk-{idx:03d} [{start:.0f}s +{dur:.0f}s] provider={provider}",
                                **transcribe_usage_fields(usage, provider),
                                cost_usd=transcribe_cost_usd(usage, provider))
                accumulate_usage(total_usage, usage)
                write_stage_json(cache_file, {"_cache": cache_key, "model": model,
                                              "usage": usage, "segments": raw_segs})
            if coverage_check:
                coverage.append(chunk_coverage_report(idx, part, raw_segs, silences))
            for s in raw_segs:
                s = dict(s)
                s["start"] += start
                s["end"] += start
                # drop segments that lie fully inside this chunk's overlap tail
                if s["end"] <= part.drop_before:
                    continue
                all_segments.append(s)
        if coverage:
            report_chunk_coverage(workdir, coverage)
        all_segments.sort(key=lambda s: s["start"])
        segments = all_segments
    # Drop degenerate segments (end <= start): the transcription API occasionally
    # returns zero-duration duplicates, which are invalid SRT cues and make the
    # srt parser log a skip per cue on every edit run.
    kept = [s for s in segments if float(s["end"]) > float(s["start"])]
    if len(kept) != len(segments):
        log(f"transcribe: dropped {len(segments) - len(kept)} degenerate segment(s) with end <= start")
    segments = kept
    doc = {"model": model, "provider": provider, "segments": segments, "_cache": provenance}
    if total_usage:
        doc["usage"] = total_usage
        doc["cost_usd"] = round(transcribe_cost_usd(total_usage, provider), 6)
    (workdir / ".edit-done").unlink(missing_ok=True)
    write_srt(segments, workdir / "raw.srt")
    (workdir / "transcript.txt").write_text(
        "\n".join(f"[{srt_ts(s['start'])[:8]}] {s['text'].strip()}" for s in segments))
    write_stage_json(transcript_json, doc)
    log(f"transcribe: {len(segments)} segments -> transcript.json, raw.srt, transcript.txt"
        + (f" (usage {total_usage}, cost ${doc['cost_usd']:.4f})" if total_usage else ""))
    return segments


# --------------------------------------------------------------------------- #
# Stage 3: hierarchical segmentation via pi oneshot (Mistral-Large-3, medium effort)
# --------------------------------------------------------------------------- #

SEGMENT_PROMPT = """\
You are segmenting a recorded workshop/training video into a hierarchy. Find the
initial cut points from the transcript below, then fine-tune them visually (see
VISUAL FINE-TUNING and KEYFRAME CUTS at the end). The transcript lines are prefixed
with the timestamp in the ORIGINAL video, format [hh:mm:ss].

VIDEO METADATA:
Title: {title}
Uploader: {uploader}
Duration: {duration_hms} ({duration:.0f} seconds)
Description (may contain the real chapter list):
{description}

TRANSCRIPT (timestamps are in the original video):
{transcript}

TASK:
{day_mode}
Segment the video hierarchically into exactly two levels:
  1. segments  - top-level parts of the video, of kind "icebreaker" or "episode"
                 (an episode = one main session/talk).
  2. sections  - parts inside a segment: the parts of an episode before/after an
                 exercise, the exercise itself, and the break time itself.
                 Each section has kind: "content" | "exercise" | "break".
                 "break" sections are the break/lunch times that should be REMOVED.

RULES:
- All times are in SECONDS in the original video (numbers, not strings).
- segments/sections must be contiguous and non-overlapping; sections must exactly
  partition their parent segment (section.start == previous section.end).
- Anchor every boundary first to a transcript timestamp (use the timestamp of the
  line where the topic changes). Round to whole seconds. After visual fine-tuning,
  boundary times should sit on video keyframes where possible (see KEYFRAME CUTS).
- Keep titles short and descriptive.
- Do NOT invent content that is not in the transcript.

VISUAL FINE-TUNING (optional, but recommended for precise boundaries):
The original video is available as raw.mp4 in the directory containing this prompt
file, and ffmpeg/ffprobe and ImageMagick are installed on PATH. After picking a cut
point from the transcript, refine it using the visuals:
{kf_note}
- Extract the frame at time t as a PNG (into the scratch dir):
    ffmpeg -y -ss <t> -i raw.mp4 -frames:v 1 agent-scratch/frame.png
- Measure the visual difference between two frames with ImageMagick:
    compare -metric AE agent-scratch/frame_a.png agent-scratch/frame_b.png null:
  (also try -metric RMSE; a large value means the picture changed -- a scene/slide/
  camera cut; `compare` can also write a highlighted diff image instead of null:)
- You are a multimodal model: read the extracted PNGs and look at them directly to
  see where the picture actually changes (slide transitions, camera cuts,
  screen-share start/stop, black or break frames).
{sandbox_note}
KEYFRAME CUTS:
The edit stage cuts with stream copy, so every cut point must be a video keyframe
(use the keyframe list above). When fine-tuning a boundary, snap it to a
keyframe:
- At the BEGINNING of a segment, seek BACKWARDS to the previous keyframe: a start
  time inside a GOP makes the stream-copied cut begin at the NEXT keyframe instead,
  which drops the beginning of the segment.
- At the END of a segment, seeking FORWARD to the next keyframe may be needed so the
  tail of the segment is not cut off mid-GOP.

OUTPUT FORMAT: reply with exactly one fenced ```json code block and nothing else,
conforming to this schema:
```json
{{
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
```
"""


def fmt_hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_segment_prompt(workdir: Path, segments: list[dict],
                         day_title: str | None = None, sandboxed: bool = False) -> str:
    info = json.loads((workdir / "raw.info.json").read_text())
    duration = probe_duration(workdir / "raw.mp4")
    kf_ok = load_keyframe_cache(workdir / "keyframes.json", workdir / "raw.mp4") is not None
    if kf_ok:
        kf_note = (
            "- The video keyframes (packets whose flags contain K) have ALREADY been probed\n"
            "  by the pipeline and stored as JSON in keyframes.json in this directory:\n"
            "      {\"version\": 1, \"source\": \"raw.mp4\", \"fingerprint\": {...}, \"keyframes\": [[pts, dts], ...]}\n"
            "  (times in seconds, sorted by pts). READ that file, e.g.:\n"
            "      python3 -c \"import json; print(json.load(open('keyframes.json'))['keyframes'])\"\n"
            "  Do NOT run ffprobe again, and NEVER create, overwrite or delete keyframes.json\n"
            "  or any other pipeline file."
        )
    else:
        kf_note = (
            "- List the video keyframes (packets whose flags contain K) with their timestamps,\n"
            "  writing the raw listing into the scratch dir only:\n"
            "      ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,dts_time,flags -of csv=p=0 raw.mp4 > agent-scratch/packets.csv\n"
            "  (each CSV line is pts,dts,flags; a flags field containing K = keyframe)"
        )
    if sandboxed:
        sandbox_note = (
            "FILE ACCESS: this directory is mounted READ-ONLY for you (bubblewrap sandbox).\n"
            "Write ALL scratch files -- frames, packet CSVs, helper scripts, notes, your\n"
            "session file -- into the agent-scratch/ subdirectory only; writes anywhere else\n"
            "will fail. keyframes.json, segments.json, raw.mp4, raw.srt, transcript.*,\n"
            "segment-prompt.md, pi-session.jsonl and pi-transcript.json are pipeline-owned\n"
            "and protected."
        )
    else:
        sandbox_note = (
            "FILE ACCESS: never create, overwrite or delete pipeline files (keyframes.json,\n"
            "segments.json, raw.mp4, raw.srt, transcript.*, segment-prompt.md,\n"
            "pi-session.jsonl, pi-transcript.json). Write ALL scratch files (frames, packet\n"
            "CSVs, helper scripts, your session file) into agent-scratch/ only."
        )
    # compact transcript: [hh:mm:ss] text
    lines = []
    for s in segments:
        text = re.sub(r"\s+", " ", s["text"]).strip()
        if text:
            lines.append(f"[{fmt_hms(s['start'])}] {text}")
    transcript = "\n".join(lines)
    if day_title is None:
        day_mode = ""
    else:
        # config mode: this video is one pre-cut day of a multi-day event. The day
        # is given by the caller -- do not create, guess or label day boundaries.
        day_mode = (
            f'CONTEXT: this video is "{day_title}" of a multi-day event. The day is '
            f'already determined -- do NOT create, guess or label day boundaries; '
            f'segment only the segments/sections within this video.\n'
        )
    return SEGMENT_PROMPT.format(
        day_mode=day_mode, kf_note=kf_note, sandbox_note=sandbox_note,
        title=info.get("title", ""), uploader=info.get("uploader", ""),
        duration=duration, duration_hms=fmt_hms(duration),
        description=(info.get("description") or "")[:8000],
        transcript=transcript)


def pi_bare_env(scratch: Path) -> dict[str, str]:
    """Environment for every pi invocation: PI_CODING_AGENT_DIR points at a fresh,
    empty dir inside agent-scratch, so pi runs with its built-in defaults (a bare
    config) and never reads the user's ~/.pi/agent (settings, extensions, custom
    models, auth) -- exactly like inside the Docker image, where no ~/.pi/agent
    exists. Auth comes from MISTRAL_API_KEY in the inherited environment (pi's
    env-var API key fallback), as in the container. The dir lives inside
    agent-scratch so it is writable both inside the bwrap sandbox (the only
    writable bind there) and outside; it is recreated empty on every call so no
    state from a previous run can leak in. The capability probes below must run
    with this same env so the flags they detect match the ones the oneshot gets."""
    agent_dir = (scratch / PI_AGENT_DIR_NAME).resolve()
    shutil.rmtree(agent_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    return env


def pi_supports_effort(pi_bin: str, env: dict | None = None) -> bool:
    try:
        help_out = subprocess.run([pi_bin, "--help"], capture_output=True, text=True,
                                  env=env).stdout
        return "--effort" in help_out
    except Exception:
        return False


def pi_supports_tools(pi_bin: str, env: dict | None = None) -> bool:
    try:
        help_out = subprocess.run([pi_bin, "--help"], capture_output=True, text=True,
                                  env=env).stdout
        return "--tools" in help_out
    except Exception:
        return False


def resolve_sandbox(mode: str) -> bool:
    """Whether the pi run will be wrapped in a bwrap read-only sandbox.
    mode "auto": bwrap if on PATH, else unsandboxed with a warning;
    mode "bwrap": require bwrap; mode "none": never sandbox."""
    if mode == "none":
        return False
    if shutil.which("bwrap"):
        return True
    if mode == "bwrap":
        sys.exit("ERROR: --sandbox bwrap but bwrap is not on PATH")
    log("segment: WARNING bwrap not found; running pi WITHOUT a filesystem sandbox")
    return False


def bwrap_command(cmd: list[str], workdir: Path) -> list[str]:
    """Wrap `cmd` in a bubblewrap sandbox: the whole filesystem is mounted
    read-only except workdir/agent-scratch/ (created here) and an ephemeral
    /tmp, so the agent cannot modify pipeline files (raw.mp4, keyframes.json,
    segments.json, ...) even via shell redirects. HOME/TMPDIR point into the
    tmpfs so pi itself writes nothing persistent. Network stays up (the model
    API call needs it) and the environment (e.g. MISTRAL_API_KEY, and
    PI_CODING_AGENT_DIR from pi_bare_env, which lands on the writable scratch
    bind) passes through."""
    scratch = (workdir / "agent-scratch").resolve()
    scratch.mkdir(exist_ok=True)
    return [shutil.which("bwrap"), "--ro-bind", "/", "/",
            "--bind", str(scratch), str(scratch),
            "--dev-bind", "/dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--setenv", "HOME", "/tmp/pi-home",
            "--setenv", "TMPDIR", "/tmp",
            "--dir", "/tmp/pi-home",
            "--"] + cmd


def build_pi_oneshot_cmd(pi_bin: str, model: str, effort: str,
                         prompt_file: Path, session_file: Path,
                         env: dict | None = None) -> list[str]:
    """pi oneshot command for the segment stage. The session is persisted to
    `session_file` (instead of --no-session) so the transcript lands in the workdir.
    `env` (from pi_bare_env) is used for the capability probes so the flags probed
    are the ones available under the bare pi config the oneshot itself runs with."""
    cmd = [pi_bin, "--print", "--session", str(session_file),
           "--model", model, "--thinking", effort, "@" + str(prompt_file.resolve())]
    if pi_supports_effort(pi_bin, env):
        cmd += ["--effort", effort]
    if pi_supports_tools(pi_bin, env):
        # the segment task only needs read (prompt/frames/keyframes.json) and
        # bash (ffprobe/ffmpeg/compare); never the write/edit tools
        cmd += ["--tools", "read,bash"]
    return cmd


def save_pi_transcript(workdir: Path, session_scratch: Path, acct: Accounting | None,
                       model_fallback: str, provider: str = "mistral") -> dict | None:
    """Copy the pi session file out of agent-scratch into the workdir and extract a
    plain-JSON transcript next to it; record the run's API usage in the ledger.
    Returns the extracted transcript (None if pi wrote no session file)."""
    session_out = workdir / PI_SESSION_NAME
    if not session_scratch.exists():
        return None
    shutil.copyfile(session_scratch, session_out)
    transcript = extract_pi_transcript(session_out)
    (workdir / PI_TRANSCRIPT_NAME).write_text(json.dumps(transcript, indent=2))
    totals = transcript["usage"]
    model = transcript.get("model") or model_fallback
    provider = transcript.get("provider") or provider
    cost = totals["cost"]["total"] or estimate_chat_cost_usd(totals, provider, model)
    n_req = sum(1 for m in transcript["messages"]
                if m["role"] == "assistant" and m.get("usage"))
    if acct is not None:
        acct.record(step="segment", model=model,
                    detail=f"pi oneshot (session {str(transcript.get('session_id'))[:8]}) provider={provider}",
                    requests=max(n_req, 1), **pi_usage_fields(totals), cost_usd=cost)
    log(f"segment: pi transcript -> {session_out.name} + {PI_TRANSCRIPT_NAME} "
        f"({n_req} api response(s), {totals['totalTokens']} tokens, ${cost:.4f})")
    return transcript


def pi_server_error_detail(diag: str) -> str:
    """The actual status code and message behind a 5xx match, e.g.
    '503 Service unavailable', so the retry log and the final error name the real
    server error instead of just '5xx'. pi records the provider error in its session
    ('Mistral API error (503): {"object":"error","message":"Service unavailable",...}');
    when it records only 'Provider stopped with: error (server error)' there is no
    code to report and this returns ''."""
    m = re.search(r"\b(5\d\d)\b", diag)
    if not m:
        return ""
    msg = _error_message_after(diag, m.end())
    return f"{m.group(1)} {msg}" if msg else m.group(1)


def _error_message_after(diag: str, pos: int) -> str:
    """The human message that follows a status code in the diagnostics: the
    'message' field of a JSON error body, else the reason phrase on the same line
    ('503 Service Unavailable')."""
    rest = diag[pos:pos + 2000]
    brace = rest.find("{")
    if brace != -1:
        try:
            body, _ = json.JSONDecoder().raw_decode(rest[brace:])
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("message"), str):
            return body["message"][:200]
    return rest.split("\n", 1)[0].lstrip(" \t:)-").strip()[:200]


# Common `pi` oneshot failure causes, matched case-insensitively against the pi
# process output AND the error messages pi records in its session transcript.
# Fatal causes (out of money, auth, bad request, ...) are never retried -- the
# identical request cannot succeed on a retry; everything else is treated as a
# transient/random failure and retried up to PI_MAX_ATTEMPTS times. A retryable
# entry may carry a third element: a function that pulls the concrete detail
# (the actual status code and message) out of the diagnostics.
PI_FATAL_CAUSES = [
    (r"payment required|insufficient (balance|credits|funds)|out of (credits|money)|"
     r"credit balance|top[- ]?up your|\b402\b|quota exceeded|insufficient_quota|billing",
     "API balance/billing problem -- the provider account looks out of money/credits"),
    (r"\b401\b|\b403\b|unauthorized|forbidden|invalid( or missing)? api key|"
     r"authentication (error|failed)|api key not (valid|found)",
     "auth problem -- bad or missing provider API key"),
    (r"context length|too many tokens|maximum context|prompt (is )?too long|"
     r"exceeds the model'?s (context|limit)",
     "prompt too large for the model"),
    (r"model (not found|does not exist|doesn'?t exist)|no such model|invalid model|\b404\b",
     "unknown/invalid model"),
    (r"\b400\b|\b422\b|bad request|invalid request",
     "bad request (deterministic -- a retry would send the identical request)"),
]
PI_RETRYABLE_CAUSES = [
    (r"\b429\b|rate ?limit|too many requests|overloaded",
     "rate limited / overloaded", None),
    (r"\b5\d\d\b|server error|internal (server )?error|bad gateway|service unavailable|"
     r"gateway timeout|upstream",
     "API server error (5xx)", pi_server_error_detail),
    (r"network|econnreset|econnrefused|etimedout|timed? ?out|timeout|fetch failed|"
     r"socket|dns|eai_again|connection (reset|refused|closed)",
     "network error", None),
    (r"terminated|aborted|interrupted|killed|sigterm|sigint|broken pipe",
     "run terminated/aborted (killed or cancelled mid-request)", None),
]


def classify_pi_failure(rc: int, diag: str) -> tuple[str, str]:
    """Classify a failed pi oneshot run -> ("fatal" | "retryable", cause).
    Unknown causes and a process killed by a signal count as random failures
    (retryable); the caller gives up after PI_MAX_ATTEMPTS of those."""
    low = diag.lower()
    for pat, cause in PI_FATAL_CAUSES:
        if re.search(pat, low):
            return "fatal", cause
    for pat, cause, detail in PI_RETRYABLE_CAUSES:
        if re.search(pat, low):
            extra = detail(diag) if detail else ""
            return "retryable", f"{cause}: {extra}" if extra else cause
    if rc < 0:
        return "retryable", f"pi process killed by signal {-rc}"
    return "retryable", "unknown/random failure"


def pi_session_errors(transcript: dict | None) -> list[str]:
    """The distinct errorMessage strings pi recorded in its session (assistant
    messages carry them, e.g. 'Provider stopped with: error (server error)')."""
    errs: list[str] = []
    for m in (transcript or {}).get("messages", []):
        e = m.get("error")
        if e and e not in errs:
            errs.append(e)
    return errs


def pi_failure_diagnostics(proc: subprocess.CompletedProcess,
                           transcript: dict | None) -> str:
    """Human-readable excerpt of a failed pi run. The session error messages are the
    reliable source (on a provider error pi often prints only a terse 'terminated'
    on stderr); the process output tails are added for anything else."""
    parts = []
    errs = pi_session_errors(transcript)
    if errs:
        parts.append("pi session error(s): " + " | ".join(errs[-4:]))
    for name, out in (("stderr", proc.stderr), ("stdout", proc.stdout)):
        out = (out or "").strip()
        if out:
            parts.append(f"{name} tail: {out[-1500:]}")
    return "\n".join(parts) if parts else "(no output captured)"


def step_segment(workdir: Path, model: str, effort: str, pi_bin: str,
                 mock_segments: Path | None, day_title: str | None = None,
                 sandbox: str = "auto", acct: Accounting | None = None, *,
                 provider: str = "mistral") -> dict:
    """Segment one video into segments > sections. day_title (config mode only) is
    the day this video belongs to, given by the caller -- passed to the prompt as
    context; the model never guesses days."""
    segments_json = workdir / "segments.json"
    transcript_json = workdir / "transcript.json"
    source = mock_segments if mock_segments else transcript_json
    source_bytes = source.read_bytes() if source.exists() else None
    provenance = {"provider": provider, "model": model, "effort": effort,
                  "day_title": day_title, "mock": bool(mock_segments),
                  "source_sha256": hashlib.sha256(source_bytes).hexdigest()
                  if source_bytes is not None else None}
    if segments_json.exists():
        cached = json.loads(segments_json.read_text())
        if cached.get("_cache") == provenance or (
                "_cache" not in cached and not mock_segments and provider == "mistral"
                and model == DEFAULT_SEGMENT_MODEL and effort == DEFAULT_EFFORT
                and (source_bytes is None or "_cache" not in json.loads(source_bytes))):
            log("segment: segments.json matches, skipping")
            return cached
        log("segment: segments.json was produced with different settings or input")
    duration = probe_duration(workdir / "raw.mp4")
    if mock_segments:
        log(f"segment: using mock segments {mock_segments}")
        data = json.loads(mock_segments.read_text())
    else:
        transcript = json.loads(source_bytes if source_bytes is not None
                                else transcript_json.read_bytes())["segments"]
        will_sandbox = resolve_sandbox(sandbox)
        prompt = build_segment_prompt(workdir, transcript, day_title=day_title,
                                      sandboxed=will_sandbox)
        prompt_file = workdir / "segment-prompt.md"
        prompt_file.write_text(prompt)
        log(f"segment: prompt is {len(prompt)} chars -> {prompt_file}")
        # the session file goes into agent-scratch: inside the bwrap sandbox that is
        # the only writable place, and pi writes it incrementally (survives a crash);
        # it is copied to the workdir as pi-session.jsonl right after the run.
        # Absolute path: pi resolves a relative --session path against its own cwd.
        if shutil.which(pi_bin) is None:
            sys.exit(f"ERROR: pi binary not found: {pi_bin!r} (install pi, or set --pi-bin / PI_BIN)")
        api_key_env = PROVIDER_API_KEYS.get(provider)
        if api_key_env and not os.environ.get(api_key_env):
            log(f"segment: WARNING {api_key_env} is not set -- pi uses it for {provider} auth")
        scratch = (workdir / "agent-scratch").resolve()
        scratch.mkdir(exist_ok=True)
        session_scratch = scratch / PI_SESSION_NAME
        # bare pi config: PI_CODING_AGENT_DIR -> empty dir in agent-scratch, so pi
        # never picks up the user's ~/.pi/agent (sandboxed or not -- inside bwrap
        # the env passes through and the dir is on the writable scratch bind)
        env = pi_bare_env(scratch)
        cmd = build_pi_oneshot_cmd(pi_bin, model, effort, prompt_file, session_scratch,
                                   env=env)
        if will_sandbox:
            cmd = bwrap_command(cmd, workdir)
        log(f"segment: running pi oneshot{' in bwrap sandbox' if will_sandbox else ''} "
            f"with bare config (PI_CODING_AGENT_DIR={env['PI_CODING_AGENT_DIR']}): "
            f"{' '.join(cmd[:8])} ...")
        # Retry loop: common fatal pi problems (out of API money, bad/missing API key,
        # prompt too large, unknown model, bad request) exit immediately with an
        # informative message; anything else (API 5xx, 429, network errors, the run
        # being terminated/killed, unknown random failures) is retried up to
        # PI_MAX_ATTEMPTS total attempts, then dies with the diagnostics.
        for attempt in range(1, PI_MAX_ATTEMPTS + 1):
            session_scratch.unlink(missing_ok=True)  # fresh session per attempt
            (workdir / PI_SESSION_NAME).unlink(missing_ok=True)
            if attempt > 1:
                log(f"segment: pi attempt {attempt}/{PI_MAX_ATTEMPTS}")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, env=env)
            except FileNotFoundError as e:
                sys.exit(f"ERROR: cannot run pi: {cmd[0]!r} not found/executable ({e}); "
                         f"install pi or set --pi-bin / PI_BIN")
            # save the transcript + record the spend even of a failed attempt
            transcript = save_pi_transcript(workdir, session_scratch, acct, model, provider)
            if proc.returncode == 0:
                break
            diag = pi_failure_diagnostics(proc, transcript)
            kind, cause = classify_pi_failure(proc.returncode, diag)
            if kind == "fatal":
                sys.exit(f"ERROR: pi failed (rc={proc.returncode}) on attempt "
                         f"{attempt}/{PI_MAX_ATTEMPTS}: {cause} -- NOT retried, a retry "
                         f"would send the identical request.\n{diag}\n"
                         f"pi session: {workdir / PI_SESSION_NAME} (spend already in the usage "
                         f"ledger). Fix the cause -- e.g. top up the {provider} API balance, set "
                         f"{api_key_env or 'the provider API key'}, shrink the prompt, pick an existing model -- and "
                         f"re-run the same command.")
            if attempt == PI_MAX_ATTEMPTS:
                sys.exit(f"ERROR: pi failed after {PI_MAX_ATTEMPTS} attempts "
                         f"(last: rc={proc.returncode}, {cause}).\n{diag}\n"
                         f"pi session: {workdir / PI_SESSION_NAME} (spend already in the usage "
                         f"ledger). Re-run the same command to retry the segment stage -- it "
                         f"is only cached once it succeeds.")
            delay = min(120.0, 15.0 * 2 ** (attempt - 1)) + random.uniform(0, 2.0)
            log(f"segment: pi attempt {attempt}/{PI_MAX_ATTEMPTS} failed "
                f"({cause}, rc={proc.returncode}); retrying in {delay:.1f}s")
            time.sleep(delay)
        (workdir / "segment-raw-output.txt").write_text(proc.stdout)
        data = extract_json_block(proc.stdout)
    data = validate_segments(data, duration)
    data["_cache"] = provenance
    (workdir / ".edit-done").unlink(missing_ok=True)
    write_stage_json(segments_json, data)
    segs = segments_of(data)
    n_sec = sum(len(s["sections"]) for s in segs)
    n_break = sum(1 for s in segs for sec in s["sections"] if sec["kind"] == "break")
    log(f"segment: {len(segs)} segments, {n_sec} sections ({n_break} breaks) -> segments.json"
        + (f' [day: {day_title}]' if day_title else ""))
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


def segments_of(data: dict) -> list[dict]:
    """The segment list of the intermediate format; tolerates the legacy
    {"days": [{... "segments": [...]}]} shape by flattening it."""
    if "segments" in data:
        return data["segments"]
    return [s for d in data.get("days", []) for s in d.get("segments", [])]


def validate_segments(data: dict, duration: float) -> dict:
    """Normalize + sanity-check the intermediate format ({"segments": [...]});
    clamps to the video duration. Accepts the legacy days>segments shape too."""
    segs: list[dict] = []
    for seg in segments_of(data):
        seg = dict(seg)
        seg.setdefault("kind", "episode")
        seg["start"] = max(0.0, min(float(seg.get("start", 0)), duration))
        seg["end"] = min(float(seg.get("end", duration)), duration)
        if seg["end"] <= seg["start"]:
            seg["end"] = min(seg["start"] + 1, duration)
        sections = []
        for sec in seg.get("sections", []):
            sec = dict(sec)
            sec.setdefault("kind", "content")
            sec["start"] = max(seg["start"], min(float(sec.get("start", seg["start"])), seg["end"]))
            sec["end"] = min(seg["end"], max(float(sec.get("end", sec["start"])), sec["start"]))
            if sec["end"] > sec["start"]:
                sections.append(sec)
        if not sections:
            sections = [{"kind": "content", "title": seg.get("title", "segment"),
                         "start": seg["start"], "end": seg["end"]}]
        seg["sections"] = sections
        segs.append(seg)
    if not segs:
        segs = [{"kind": "episode", "title": "video", "start": 0.0, "end": duration,
                 "sections": [{"kind": "content", "title": "video",
                               "start": 0.0, "end": duration}]}]
    return {"segments": segs}


# --------------------------------------------------------------------------- #
# Stage 4: intermediate format -> keyframe-aligned cuts -> apply
# --------------------------------------------------------------------------- #

def slugify(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\- ]+", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:maxlen].strip("-") or "segment"


def hms(t: float) -> str:
    return fmt_hms(float(t))


def ranges_for_segment(seg: dict, keyframes: list[list[float]] | None = None
                       ) -> list[tuple[float, float]]:
    """Non-break section ranges of one segment, as (start, stop) in raw-video seconds.

    With `keyframes`, each start is moved to the nearest keyframe that is still before
    the range's stop -- a stream-copied cut can only begin on a keyframe, so the
    request has to move somewhere, and the distance is logged so a large move (a
    boundary the segmenter placed badly) is visible.
    """
    out: list[tuple[float, float]] = []
    for sec in seg["sections"]:
        if sec["kind"] == "break":
            continue
        s, e = snap_range_start(float(sec["start"]), float(sec["end"]), keyframes)
        if e > s:
            out.append((s, e))
    if not out:
        s, e = snap_range_start(float(seg["start"]), float(seg["end"]), keyframes)
        if e > s:
            out.append((s, e))
    return out


def snap_range_start(s: float, e: float, keyframes: list[list[float]] | None
                     ) -> tuple[float, float]:
    """Move a range start onto a keyframe that still leaves a non-empty range."""
    if not keyframes:
        return s, e
    usable = [kf[0] for kf in keyframes if 0 <= kf[0] < e]
    if not usable:
        return s, e
    snapped = min(usable, key=lambda pts: abs(pts - s))
    if abs(snapped - s) > 2.0:
        log(f"edit: range start {s:.1f}s moved to the nearest keyframe {snapped:.1f}s "
            f"({snapped - s:+.1f}s)")
    return snapped, e


def build_output_ranges(data: dict, combined: bool,
                        keyframes: list[list[float]] | None = None) -> list[dict]:
    """One output per segment (only its non-break sections), plus optionally one
    combined output with every break removed."""
    outputs = []
    every_range: list[tuple[float, float]] = []
    for seg_idx, seg in enumerate(segments_of(data), 1):
        ranges = ranges_for_segment(seg, keyframes)
        outputs.append({
            "output": f"{seg_idx:02d}-{slugify(seg.get('title', seg['kind']))}.mp4",
            "ranges": ranges,
        })
        every_range.extend(ranges)
    if combined:
        merged: list[list[float]] = []
        for s, e in sorted(every_range):
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        outputs.append({"output": "full-nobreaks.mp4",
                        "ranges": [(s, e) for s, e in merged]})
    return outputs


def render_ranges(workdir: Path, out: Path, ranges: list[tuple[float, float]],
                  reencode: bool) -> list[tuple[float, float]]:
    """Write `out` as the concatenation of `ranges` cut from raw.mp4.

    Returns the ranges actually cut: with stream copy each start is rounded up to the
    millisecond of its keyframe PTS, which is what the .srt rebasing must use. Nothing
    else is shifted, so the output timeline is exactly the running sum of the cut
    durations and does not accumulate drift from range to range.
    """
    raw_mp4 = workdir / "raw.mp4"
    tmpdir = workdir / "edit-tmp"
    tmpdir.mkdir(exist_ok=True)
    parts: list[Path] = []
    cut: list[tuple[float, float]] = []
    for i, (start, stop) in enumerate(ranges):
        # Round up so `-ss` can never fall a hair below the keyframe PTS, which would
        # make ffmpeg seek to the *previous* keyframe and include a whole GOP.
        s = math.ceil(start * 1000) / 1000
        e = math.ceil(stop * 1000) / 1000
        part = tmpdir / f"{out.stem}-{i:03d}.mp4"
        cut_range(raw_mp4, s, e, part, reencode)
        parts.append(part)
        cut.append((s, e))
    if len(parts) == 1:
        shutil.move(str(parts[0]), str(out))
    else:
        concat_mp4(parts, [e - s for s, e in cut], out, workdir, out.stem)
    for part in parts:
        part.unlink(missing_ok=True)
    return cut


def cut_range(raw_mp4: Path, start: float, stop: float, out: Path,
              reencode: bool) -> None:
    """Cut one range to its own file.

    Stream copy seeks with `-ss` as an *input* option, which lands exactly on the
    keyframe at or before `start` (ffmpeg-editlist used an output `-ss` instead, which
    drops packets until the *next* keyframe and so loses up to a GOP of content at
    every cut). Re-encoding seeks frame-accurately, so no keyframe is needed at all.
    """
    cmd = [which_or_die("ffmpeg"), "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-to", f"{stop:.3f}", "-i", str(raw_mp4),
           "-map", "0:v:0", "-map", "0:a:0?"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    run(cmd + [str(out)])


def concat_mp4(parts: list[Path], durations: list[float], out: Path,
               workdir: Path, tag: str) -> None:
    """Concatenate cut parts without re-encoding, at exactly the given offsets.

    The concat demuxer otherwise advances by each file's *container* duration, which
    is the longer of its audio and video tracks. Because a stream-copied part's audio
    and video end up to a frame apart, every join then pushed the audio ahead of the
    video and ahead of the timeline the .srt was rebased onto (55ms per join on
    day-2, 260ms over four joins, and it grows with the number of cuts). Declaring
    `duration` explicitly makes the demuxer advance by exactly the cut length and
    offsets both tracks by the same amount, so picture, sound and subtitles stay on
    one timeline.
    """
    listfile = workdir / f"concat-{slugify(tag)}.txt"
    listfile.write_text("".join(
        f"file '{p.resolve()}'\nduration {d:.3f}\n" for p, d in zip(parts, durations)))
    run([which_or_die("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat",
         "-safe", "0", "-i", str(listfile), "-fflags", "+igndts",
         "-c", "copy", str(out)])
    listfile.unlink(missing_ok=True)


def rebase_srt(raw_cues: list[dict], ranges: list[tuple[float, float]],
               path: Path) -> None:
    """Map cues from raw-video time onto the timeline `ranges` produces.

    Cues are clipped to each range, so a cue crossing a cut is split and then merged
    back together on the output timeline (rather than being emitted twice, as
    ffmpeg-editlist did). Anything inside a removed break disappears.
    """
    cums: list[float] = []
    c = 0.0
    for s, e in ranges:
        cums.append(c)
        c += e - s
    pieces: list[dict] = []
    for idx, (s, e) in enumerate(ranges):
        off = s - cums[idx]
        for ci, cue in enumerate(raw_cues):
            lo, hi = max(cue["start"], s), min(cue["end"], e)
            if hi <= lo:
                continue
            pieces.append({"start": lo - off, "end": hi - off,
                           "text": cue["text"], "_src": ci})
    merged: list[dict] = []
    for piece in pieces:
        prev = merged[-1] if merged else None
        if (prev is not None and prev["_src"] == piece["_src"]
                and piece["start"] - prev["end"] <= 0.002):
            prev["end"] = max(prev["end"], piece["end"])
            continue
        merged.append(piece)
    for cue in merged:
        del cue["_src"]
    write_srt(merged, path)


ALIGN_AUDIO_BLOCK = 0.01        # seconds per envelope block (~100 Hz correlation)
ALIGN_AUDIO_WINDOW = 20.0       # audio compared per sample point
ALIGN_FRAME_WIDTH, ALIGN_FRAME_HEIGHT = 64, 36
ALIGN_FRAME_FPS = 4.0           # video search grid
ALIGN_BURST_SECONDS = 1.0       # consecutive frames compared at each candidate
ALIGN_AUDIO_MIN_PEAK = 0.5      # normalised correlation needed to trust an audio lag
ALIGN_VIDEO_MAX_RMSE = 24.0     # mean |delta| in grey levels needed to trust a frame


def _read_wav_mono(mp4: Path, start: float, dur: float, tmp: Path) -> array.array:
    """Decode a short window of an mp4's audio to 8 kHz mono int16 samples."""
    run([which_or_die("ffmpeg"), "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(mp4),
         "-vn", "-ac", "1", "-ar", "8000", "-f", "wav", str(tmp)])
    with wave.open(str(tmp)) as w:
        return array.array("h", w.readframes(w.getnframes()))


def _envelope(samples: array.array, block_seconds: float = ALIGN_AUDIO_BLOCK,
              rate: int = 8000) -> list[float]:
    """Block RMS of the signal, i.e. its loudness envelope."""
    n = max(1, int(block_seconds * rate))
    out = []
    for i in range(0, len(samples) - n + 1, n):
        chunk = samples[i:i + n]
        out.append(math.sqrt(sum(v * v for v in chunk) / n))
    return out


def normalized_correlation(a: list[float], b: list[float], max_offset: int) -> tuple[int, float]:
    """Best offset of window `a` inside (the longer) window `b`, with its normalised
    correlation. Equivalent to taking the cross-correlation and dividing by the energy
    of the overlapping part, so a loud passage cannot win by sheer amplitude."""
    best_k, best_v = 0, 0.0
    span = min(max_offset, len(b) - len(a))
    mean_a = sum(a) / len(a) if a else 0.0
    da = [v - mean_a for v in a]
    energy_a = math.sqrt(sum(v * v for v in da)) or 1.0
    for k in range(max(0, span) + 1):
        seg = b[k:k + len(a)]
        mean_b = sum(seg) / len(seg)
        db = [v - mean_b for v in seg]
        energy_b = math.sqrt(sum(v * v for v in db))
        if energy_b == 0:
            continue
        v = sum(x * y for x, y in zip(da, db)) / (energy_a * energy_b)
        if v > best_v:
            best_k, best_v = k, v
    return best_k, best_v


def _read_gray_frames(mp4: Path, start: float, dur: float, fps: float) -> list[bytes]:
    """Decode downscaled greyscale frames of a window, one per 1/fps second."""
    proc = subprocess.run(
        [which_or_die("ffmpeg"), "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(mp4),
         "-vf", f"scale={ALIGN_FRAME_WIDTH}:{ALIGN_FRAME_HEIGHT},fps={fps}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True)
    size = ALIGN_FRAME_WIDTH * ALIGN_FRAME_HEIGHT
    data = proc.stdout
    return [data[i:i + size] for i in range(0, len(data) - size + 1, size)]


def frame_rmse(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two greyscale frames, in grey levels (0-255)."""
    n = min(len(a), len(b))
    if n == 0:
        return 255.0
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def expected_raw_time(t: float, ranges: list[tuple[float, float]]) -> float:
    """Where the timeline the cut ranges define says output time `t` came from."""
    cum = 0.0
    for s, e in ranges:
        if t < cum + (e - s):   # half-open: t exactly on a join belongs to the next
            return s + (t - cum)
        cum += e - s
    return ranges[-1][0] + (t - cum) if ranges else t


def measure_alignment(out_mp4: Path, raw_mp4: Path, ranges: list[tuple[float, float]],
                      samples: int, search: float, tmpdir: Path) -> list[dict]:
    """Measure, at `samples` points, how far the output's real content sits from the
    timeline its .srt was rebased onto.

    Audio is matched by cross-correlating the loudness envelope against raw.mp4, which
    pins the lag to the 10ms envelope block; video is matched by comparing a
    downscaled greyscale frame against raw frames on a 1/fps grid across ±`search`.
    The two are measured independently so a picture/sound divergence cannot hide.
    """
    duration = probe_duration(out_mp4)
    results: list[dict] = []
    usable = [t for t in (duration * (i + 0.5) / samples for i in range(samples))
              if t > ALIGN_AUDIO_WINDOW / 2 and t < duration - ALIGN_AUDIO_WINDOW / 2]
    wav = tmpdir / "align.wav"
    span = 2 * search + ALIGN_AUDIO_WINDOW
    for t in usable:
        exp = expected_raw_time(t, ranges)
        row = {"t": t, "expected_raw": exp, "audio": None, "video": None,
               "audio_peak": 0.0, "video_rmse": None}
        try:
            here = _envelope(_read_wav_mono(out_mp4, t - ALIGN_AUDIO_WINDOW / 2,
                                            ALIGN_AUDIO_WINDOW, wav))
            there = _envelope(_read_wav_mono(raw_mp4, exp - search - ALIGN_AUDIO_WINDOW / 2,
                                             span, wav))
            if here and there and len(there) > len(here):
                k, peak = normalized_correlation(here, there, len(there) - len(here))
                row["audio_peak"] = peak
                if peak >= ALIGN_AUDIO_MIN_PEAK:
                    # `here` starts at t - window/2, `there` at exp - search - window/2,
                    # so matching block k puts the content at exp - search + k*block.
                    row["audio"] = k * ALIGN_AUDIO_BLOCK - search
        except (OSError, subprocess.SubprocessError, wave.Error):
            pass
        try:
            # Compare a short burst, not a single frame: a screen recording is often
            # nearly identical from one second to the next, so one frame cannot say
            # where it came from but a few consecutive ones usually can.
            span_frames = max(1, int(round(ALIGN_BURST_SECONDS * ALIGN_FRAME_FPS)))
            target = _read_gray_frames(out_mp4, t, span_frames / ALIGN_FRAME_FPS,
                                       ALIGN_FRAME_FPS)
            candidates = _read_gray_frames(
                raw_mp4, exp - search,
                (2 * search) + span_frames / ALIGN_FRAME_FPS, ALIGN_FRAME_FPS)
            if len(target) >= span_frames and len(candidates) >= span_frames:
                burst = target[:span_frames]
                scores = [sum(frame_rmse(burst[j], candidates[i + j])
                              for j in range(span_frames)) / span_frames
                          for i in range(len(candidates) - span_frames + 1)]
                order = sorted(range(len(scores)), key=lambda i: scores[i])
                best = order[0]
                runner_up = scores[order[1]] if len(order) > 1 else 1e9
                row["video_rmse"] = scores[best]
                # Only trust the match when it clearly beats every other candidate:
                # on a static slide every offset scores the same, so the "best" one
                # would be arbitrary.
                clear = scores[best] * 1.5 < runner_up
                if scores[best] > ALIGN_VIDEO_MAX_RMSE:
                    row["video_note"] = "no match"
                elif clear:
                    row["video"] = (best / ALIGN_FRAME_FPS) - search
                else:
                    row["video_note"] = "ambiguous"   # e.g. a static slide
        except (OSError, subprocess.SubprocessError):
            pass
        results.append(row)
    return results


def shift_cues(cues: list[dict], samples: list[dict]) -> list[dict]:
    """Shift cues by the measured lag, interpolated between sample points.

    A cue is moved by -lag: if the output's content at time t comes from raw
    expected(t) + lag, then speech the cue describes at raw expected(t) is really
    heard at t - lag. Samples whose measurement was rejected are dropped, so a short
    output with only one good sample still gets a constant shift.
    """
    good = sorted((s for s in samples if s["audio"] is not None), key=lambda s: s["t"])
    if not good:
        return []
    times = [g["t"] for g in good]
    lags = [g["audio"] for g in good]

    def lag_at(t: float) -> float:
        if t <= times[0]:
            return lags[0]
        if t >= times[-1]:
            return lags[-1]
        for i in range(1, len(times)):
            if t <= times[i]:
                span = times[i] - times[i - 1]
                f = 0.0 if span <= 0 else (t - times[i - 1]) / span
                return lags[i - 1] + f * (lags[i] - lags[i - 1])
        return lags[-1]

    moved = []
    for cue in cues:
        start = max(0.0, cue["start"] - lag_at(cue["start"]))
        end = max(start, cue["end"] - lag_at(cue["end"]))
        moved.append({"start": start, "end": end, "text": cue["text"]})
    return moved


def align_output(out_mp4: Path, raw_mp4: Path, ranges: list[tuple[float, float]],
                 samples: int, search: float, repair: bool, tmpdir: Path,
                 report: list[str]) -> float:
    """Measure one output's alignment, optionally repair its .srt, and report.

    Returns the worst measured lag in seconds.
    """
    rows = measure_alignment(out_mp4, raw_mp4, ranges, samples, search, tmpdir)
    audio = [r["audio"] for r in rows if r["audio"] is not None]
    video = [r["video"] for r in rows if r["video"] is not None]
    worst = max((abs(v) for v in audio + video), default=0.0)
    report.append(f"{out_mp4.name}: {len(rows)} sample point(s), "
                  f"{len(audio)} audio / {len(video)} video measured, "
                  f"worst lag {worst * 1000:.0f}ms")
    for r in rows:
        a = f"{r['audio'] * 1000:+.0f}ms" if r["audio"] is not None else \
            f"  n/a (peak {r['audio_peak']:.2f})"
        v = f"{r['video'] * 1000:+.0f}ms" if r["video"] is not None else \
            f"  n/a ({r.get('video_note', 'unreadable')})"
        report.append(f"  t={r['t']:8.2f}s raw={r['expected_raw']:9.2f}s "
                      f"audio {a:>18} video {v:>18}")
    if not audio:
        log(f"edit: alignment {out_mp4.name}: no usable audio measurement, "
            f".srt left alone")
        return worst
    if repair:
        srt = out_mp4.with_suffix(".srt")
        if srt.exists():
            cues = parse_srt(srt)
            shifted = shift_cues(cues, rows)
            if shifted:
                write_srt(shifted, srt)
                report.append(f"  repaired {len(shifted)} cue(s) by the measured lag")
                log(f"edit: alignment {out_mp4.name}: shifted {len(shifted)} cue(s) "
                    f"by the measured lag (worst {worst * 1000:.0f}ms)")
    return worst


def write_chapters(data: dict, outdir: Path, day_title: str | None = None) -> None:
    """Human/ffmpeg-readable chapter list derived from the intermediate format.
    day_title (config mode) is the day this video belongs to, given by the config."""
    lines = []
    if day_title:
        lines.append(f"# {day_title}")
    for seg_idx, seg in enumerate(segments_of(data), 1):
        lines.append(f"segment {seg_idx} ({seg.get('kind')}): {seg.get('title','')} "
                     f"[{hms(seg['start'])}-{hms(seg['end'])}]")
        for sec in seg["sections"]:
            marker = " (REMOVED)" if sec["kind"] == "break" else ""
            lines.append(f"  - {sec.get('kind')}: {sec.get('title','')} "
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


def step_edit(workdir: Path, data: dict, subs: str, reencode: bool, combined: bool,
              snap_keyframes: bool = True, day_title: str | None = None, *,
              align_check: bool = True, align_samples: int = 8,
              align_search_seconds: float = 3.0, align_repair: bool = True) -> None:
    workdir = workdir.resolve()
    outdir = (workdir / "out").resolve()
    outdir.mkdir(exist_ok=True)
    raw_mp4 = workdir / "raw.mp4"
    kfs: list[list[float]] | None = None
    # Re-encoding seeks frame-accurately, so there is no reason to round cuts onto
    # keyframes; only stream copy needs them.
    if snap_keyframes and not reencode:
        kf_cache = workdir / "keyframes.json"
        kfs = load_keyframe_cache(kf_cache, raw_mp4)
        if kfs is not None:
            log(f"edit: loaded {len(kfs)} keyframes from {kf_cache.name}")
        else:
            log("edit: probing video keyframes of raw.mp4 (cached in keyframes.json)")
            kfs = probe_keyframes(raw_mp4)
            write_keyframe_cache(kf_cache, raw_mp4, kfs)
            log(f"edit: {len(kfs)} keyframes found")
    outputs = build_output_ranges(data, combined, keyframes=kfs)
    raw_srt = workdir / "raw.srt"
    raw_cues = parse_srt(raw_srt) if raw_srt.exists() else []
    if not raw_cues:
        log("edit: WARNING no cues in raw.srt -- outputs will have no subtitles")
    log(f"edit: cutting {len(outputs)} output(s) with "
        + ("a re-encode (frame-accurate cuts)" if reencode
           else "stream copy" + (", cuts snapped to keyframes)" if kfs else ")")))
    cut_ranges: dict[str, list[tuple[float, float]]] = {}
    for entry in outputs:
        name, ranges = entry["output"], entry["ranges"]
        if not ranges:
            log(f"edit: WARNING no non-break ranges for {name}, skipping")
            continue
        mp4 = outdir / name
        cut_ranges[name] = render_ranges(workdir, mp4, ranges, reencode)
        rebase_srt(raw_cues, cut_ranges[name], mp4.with_suffix(".srt"))
        log(f"edit: {name}: {len(ranges)} range(s), "
            f"{probe_duration(mp4):.1f}s, {len(parse_srt(mp4.with_suffix('.srt')))} cues")
    tmpdir = workdir / "edit-tmp"
    if tmpdir.exists() and not any(tmpdir.iterdir()):
        tmpdir.rmdir()
    write_chapters(data, outdir, day_title=day_title)

    # Sync report: verify each output's actual video timeline matches the timeline
    # its .srt was rebased onto.
    report = [f"# sync report {time.strftime('%Y-%m-%dT%H:%M:%S')}",
              "# drift = actual video range start - cumulative cut duration"]
    for entry in outputs:
        name = entry["output"]
        mp4 = outdir / name
        if name not in cut_ranges or not mp4.exists():
            log(f"edit: WARNING output missing: {mp4}")
            continue
        ranges = cut_ranges[name]
        cums: list[float] = []
        c = 0.0
        for s, e in ranges:
            cums.append(c)
            c += e - s
        actual = probe_video_range_starts(mp4, cums)
        shifts = [a - cum for a, cum in zip(actual, cums)]
        starts, _video_dur = probe_output_timeline(mp4)
        srt = mp4.with_suffix(".srt")
        first_cue = srt_first_cue(srt) if srt.exists() else float("nan")
        max_drift = max((abs(s) for s in shifts), default=0.0)
        report.append(
            f"{name}: video_start={starts.get('video', 0.0):.3f} "
            f"audio_start={starts.get('audio', 0.0):.3f} srt_first={first_cue:.3f} "
            f"ranges={len(ranges)} max_drift={max_drift * 1000:.0f}ms")
        for i, (act, cum) in enumerate(zip(actual, cums)):
            report.append(f"  range {i + 1}: nominal {cum:.3f} -> actual {act:.3f} "
                          f"(drift {shifts[i] * 1000:+.0f} ms)")
        if max_drift > 0.5:
            log(f"edit: WARNING sync drift {max_drift * 1000:.0f} ms in {name} "
                f"(see {outdir / 'sync-report.txt'})")
        else:
            log(f"edit: sync ok {name}: max drift {max_drift * 1000:.0f} ms "
                f"over {len(ranges)} range(s)")
    (outdir / "sync-report.txt").write_text("\n".join(report) + "\n")
    log(f"edit: sync report -> {outdir / 'sync-report.txt'}")

    if align_check:
        align_report = [f"# alignment report {time.strftime('%Y-%m-%dT%H:%M:%S')}",
                        "# lag = (where the output's content really comes from) minus",
                        "#       (where the cut ranges say it should come from); the .srt is",
                        "#       shifted by the measured lag so it sits on the speech"]
        worst_overall = 0.0
        for entry in outputs:
            name = entry["output"]
            mp4 = outdir / name
            if name not in cut_ranges or not mp4.exists():
                continue
            worst_overall = max(worst_overall,
                                align_output(mp4, raw_mp4, cut_ranges[name],
                                             align_samples, align_search_seconds,
                                             align_repair, workdir, align_report))
        (outdir / "align-report.txt").write_text("\n".join(align_report) + "\n")
        report.extend(["", "# measured alignment (see align-report.txt)"] + align_report)
        (outdir / "sync-report.txt").write_text("\n".join(report) + "\n")
        log(f"edit: align report -> {outdir / 'align-report.txt'} "
            f"(worst lag {worst_overall * 1000:.0f}ms)")
    else:
        log("edit: alignment pass disabled (--no-align-check)")

    # Mux/burn the sidecar .srt into each output.
    for entry in outputs:
        name = entry["output"]
        mp4 = outdir / name
        srt = mp4.with_suffix(".srt")
        if not mp4.exists():
            log(f"edit: WARNING output missing: {mp4}")
            continue
        if subs == "sidecar":
            continue
        final = outdir / ("subs-" + name) if subs == "embed" else outdir / name
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
            log(f"edit: WARNING no srt for {name}, copying as-is")

    # original mp4 with subtitles applied
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
# CLI / TOML config
# --------------------------------------------------------------------------- #

# Built-in defaults for every pipeline option. Resolution order:
# explicit CLI flag > TOML config value (global, then per-[[days]] override) > default.
# argparse therefore uses default=None for all of these and they are resolved later.
BUILTIN_DEFAULTS = {
    "workdir_root": "work",          # config mode: base dir for per-day workdirs
    "height": 1080,
    "language": None,
    "transcribe_provider": None,
    "segment_provider": None,
    "transcribe_model": None,
    "segment_model": None,
    "effort": None,
    "chunk_minutes": API_MAX_CHUNK_MINUTES,
    "subs": "embed",
    "reencode": False,
    "no_combined": False,
    "pi_bin": os.environ.get("PI_BIN", "pi"),
    "skip_download": False,
    "skip_transcribe": False,
    "skip_segment": False,
    "skip_edit": False,
    "force_edit": False,
    "no_snap_keyframes": False,
    "sandbox": "auto",
    "max_retries": 5,
    "retry_base_seconds": 2.0,
    "retry_max_seconds": 120.0,
    "silence_dbfs": DEFAULT_SILENCE_DBFS,
    "silence_min_seconds": DEFAULT_SILENCE_MIN_SECONDS,
    "chunk_silence_min_seconds": DEFAULT_CHUNK_SILENCE_MIN_SECONDS,
    "chunk_overlap_seconds": DEFAULT_CHUNK_OVERLAP_SECONDS,
    "transcript_coverage_check": True,
    "align_check": True,
    "align_samples": 8,
    "align_search_seconds": 3.0,
    "align_repair": True,
    "mock_transcript": None,
    "mock_segments": None,
}

PATH_KEYS = {"mock_transcript", "mock_segments"}
DAY_SOURCE_KEYS = {"url", "path", "title", "workdir"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", nargs="?", default=None,
                    help="YouTube URL or 11-char video ID (single-video mode; not used with --config)")
    ap.add_argument("--config", type=Path, default=None,
                    help="TOML config listing the day videos of an event ([[days]] entries with "
                         "url= or path=). Each video is treated as exactly one day -- the "
                         "segment stage does not guess days. See config.example.toml")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="working/output dir (single-video mode; in config mode: base dir for "
                         "the per-day workdirs, default: ./work)")
    ap.add_argument("--height", type=int, default=None,
                    help="max video height for download (0 = best available)")
    ap.add_argument("--language", default=None,
                    help="ISO 639-1 language code for transcription (default: auto-detect)")
    ap.add_argument("--transcribe-provider", choices=PROVIDER_DEFAULTS,
                    help="provider for audio transcription (default: mistral)")
    ap.add_argument("--segment-provider", choices=PROVIDER_DEFAULTS,
                    help="provider for pi segmentation (default: mistral)")
    ap.add_argument("--transcribe-model", default=None)
    ap.add_argument("--segment-model", default=None,
                    help="pi model for segmentation (Mistral Large line; literal Large 3 id: mistral/mistral-large-2512)")
    ap.add_argument("--effort", default=None,
                    help="pi thinking/effort level for segmentation")
    ap.add_argument("--chunk-minutes", type=int, default=None,
                    help="maximum audio seconds per transcription request, in minutes "
                         "(default: 55, OpenAI uploads must be under 25 MB); longer "
                         "speech regions are split into even overlapping pieces")
    ap.add_argument("--silence-dbfs", type=float, default=None,
                    help=f"level below which audio counts as silence (default: "
                         f"{DEFAULT_SILENCE_DBFS} dBFS)")
    ap.add_argument("--silence-min-seconds", type=float, default=None,
                    help=f"shortest silence the analysis reports (default: "
                         f"{DEFAULT_SILENCE_MIN_SECONDS}s)")
    ap.add_argument("--chunk-silence-min-seconds", type=float, default=None,
                    help=f"transcription chunks are split at silences at least this long, "
                         f"and the silence is never uploaded (default: "
                         f"{DEFAULT_CHUNK_SILENCE_MIN_SECONDS}s)")
    ap.add_argument("--chunk-overlap-seconds", type=float, default=None,
                    help=f"overlap between the even pieces of an over-long speech region "
                         f"(default: {DEFAULT_CHUNK_OVERLAP_SECONDS}s)")
    ap.add_argument("--no-transcript-coverage-check", dest="transcript_coverage_check",
                    action="store_false", default=None,
                    help="do not compare each chunk's transcript against its own audio "
                         "(the check that warns when a provider's timestamps drift)")
    ap.add_argument("--no-align-check", dest="align_check", action="store_false",
                    default=None,
                    help="skip the post-edit alignment pass (measured audio/video drift "
                         "per output and .srt repair)")
    ap.add_argument("--align-samples", type=int, default=None,
                    help="sample points per output for the alignment pass (default: 8)")
    ap.add_argument("--align-search-seconds", type=float, default=None,
                    help="how far either side the alignment pass searches for a match "
                         "(default: 3s)")
    ap.add_argument("--no-align-repair", dest="align_repair", action="store_false",
                    default=None,
                    help="measure alignment but leave the .srt untouched")
    ap.add_argument("--subs", choices=["embed", "burn", "sidecar"], default=None,
                    help="how to apply subtitles to outputs (default: embed as mov_text)")
    ap.add_argument("--reencode", action="store_true", default=None,
                    help="re-encode instead of stream-copying the cuts (frame-accurate cut "
                         "points and seamless joins, much slower)")
    ap.add_argument("--no-combined", action="store_true", default=None,
                    help="do not emit the combined full-nobreaks.mp4 output")
    ap.add_argument("--pi-bin", default=None)
    ap.add_argument("--skip-download", action="store_true", default=None)
    ap.add_argument("--skip-transcribe", action="store_true", default=None)
    ap.add_argument("--skip-segment", action="store_true", default=None)
    ap.add_argument("--skip-edit", action="store_true", default=None,
                    help="skip the edit stage even if not done yet")
    ap.add_argument("--force-edit", action="store_true", default=None,
                    help="redo the edit stage even if the .edit-done marker exists")
    ap.add_argument("--no-snap-keyframes", action="store_true", default=None,
                    help="do not snap cut points to video keyframes (stream-copied cuts then "
                         "start at the next keyframe and subtitles drift; the sync report in "
                         "out/sync-report.txt is still written)")
    ap.add_argument("--sandbox", choices=["auto", "bwrap", "none"], default=None,
                    help="sandbox the pi segment agent in a bubblewrap read-only mount: the "
                         "workdir is read-only except workdir/agent-scratch/ (default: auto = "
                         "bwrap when on PATH, else unsandboxed with a warning)")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="per-chunk transcription retry limit (5xx/429/network errors)")
    ap.add_argument("--retry-base-seconds", type=float, default=None,
                    help="exponential backoff base for transcription retries")
    ap.add_argument("--retry-max-seconds", type=float, default=None,
                    help="cap for the backoff delay between retries")
    ap.add_argument("--mock-transcript", type=Path, default=None,
                    help="testing: JSON file with {'segments': [...]} instead of API call")
    ap.add_argument("--mock-segments", type=Path, default=None,
                    help="testing: JSON file with the intermediate segments format")
    return ap.parse_args(argv)


def _norm_keys(table: dict, where: str) -> dict:
    out = {}
    for k, v in table.items():
        key = k.replace("-", "_")
        if key not in BUILTIN_DEFAULTS:
            sys.exit(f"ERROR: unknown option {k!r} in {where} "
                     f"(allowed: {', '.join(sorted(BUILTIN_DEFAULTS))})")
        if key in PATH_KEYS and v is not None:
            v = Path(v)
        if key == "subs" and v not in ("embed", "burn", "sidecar"):
            sys.exit(f"ERROR: subs must be embed|burn|sidecar, got {v!r} in {where}")
        if key == "sandbox" and v not in ("auto", "bwrap", "none"):
            sys.exit(f"ERROR: sandbox must be auto|bwrap|none, got {v!r} in {where}")
        if key in ("transcribe_provider", "segment_provider") and v not in tuple(PROVIDER_DEFAULTS):
            sys.exit(f"ERROR: {key} must be mistral|openai, got {v!r} in {where}")
        out[key] = v
    return out


def validate_provider_config(providers: dict, path: Path) -> None:
    if not isinstance(providers, dict):
        sys.exit(f"ERROR: providers must be a table in {path}")
    for provider, stages in providers.items():
        if provider not in PROVIDER_DEFAULTS or not isinstance(stages, dict):
            sys.exit(f"ERROR: invalid provider table providers.{provider} in {path}")
        for stage, settings in stages.items():
            defaults = PROVIDER_DEFAULTS[provider].get(stage)
            where = f"providers.{provider}.{stage} in {path}"
            if defaults is None or not isinstance(settings, dict):
                sys.exit(f"ERROR: invalid provider stage table {where}")
            for key, value in settings.items():
                if key not in defaults or not isinstance(value, str) or not value.strip():
                    sys.exit(f"ERROR: invalid setting {key}={value!r} in {where}")


def load_config(path: Path) -> tuple[dict, list[dict]]:
    """Parse the TOML config: top-level keys are global pipeline options, [[days]]
    entries are the day videos. Returns (global_options, day_entries)."""
    import tomllib
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"ERROR: invalid TOML in {path}: {e}")
    raw = dict(raw)
    days = raw.pop("days", None)
    if not days:
        sys.exit(f"ERROR: {path} has no [[days]] entries (each entry is one day video)")
    providers = raw.pop("providers", {})
    validate_provider_config(providers, path)
    globals_ = _norm_keys(raw, str(path))
    globals_["providers"] = providers
    entries: list[dict] = []
    for i, d in enumerate(days, 1):
        if not isinstance(d, dict):
            sys.exit(f"ERROR: [[days]] entry {i} in {path} is not a table")
        where = f"{path} [[days]] entry {i}"
        d = {k.replace("-", "_"): v for k, v in d.items()}
        url, p = d.get("url"), d.get("path")
        if bool(url) == bool(p):
            sys.exit(f"ERROR: {where} must have exactly one of 'url' (YouTube) or 'path' (local file)")
        if p and ("://" in p or re.fullmatch(r"[\w-]{11}", p)):
            sys.exit(f"ERROR: {where} 'path' looks like a YouTube URL/video ID ({p!r}) -- "
                     f"use 'url' for YouTube videos, 'path' only for local files")
        unknown = set(d) - DAY_SOURCE_KEYS - set(BUILTIN_DEFAULTS)
        if unknown:
            sys.exit(f"ERROR: unknown keys {sorted(unknown)} in {where}")
        overrides = _norm_keys({k: v for k, v in d.items() if k not in DAY_SOURCE_KEYS}, where)
        entries.append({
            "url": url,
            "path": Path(p) if p else None,
            "title": d.get("title"),
            "workdir": Path(d["workdir"]) if d.get("workdir") else None,
            "overrides": overrides,
        })
    return globals_, entries


def resolve_options(cli: argparse.Namespace, toml_globals: dict,
                    overrides: dict | None = None) -> dict:
    """CLI flag (if given) > per-day TOML override > global TOML value > built-in default."""
    opt = {}
    for key, default in BUILTIN_DEFAULTS.items():
        v = getattr(cli, key, None)
        if v is None and overrides is not None:
            v = overrides.get(key)
        if v is None:
            v = toml_globals.get(key)
        if v is None:
            v = default
        opt[key] = v

    chunk_minutes = opt["chunk_minutes"]
    if type(chunk_minutes) is not int or chunk_minutes <= 0:
        sys.exit("ERROR: chunk_minutes must be a positive integer")
    max_chunk_seconds = chunk_minutes * 60
    for key in ("silence_dbfs", "silence_min_seconds", "chunk_silence_min_seconds",
                "chunk_overlap_seconds", "align_search_seconds"):
        v = opt[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            sys.exit(f"ERROR: {key} must be a number")
        opt[key] = float(v)
    if opt["silence_min_seconds"] <= 0:
        sys.exit("ERROR: silence_min_seconds must be positive")
    if opt["silence_dbfs"] >= 0:
        sys.exit("ERROR: silence_dbfs must be negative (it is a level below full scale)")
    if opt["chunk_silence_min_seconds"] <= 0:
        sys.exit("ERROR: chunk_silence_min_seconds must be positive")
    if not 0 < opt["chunk_overlap_seconds"] < max_chunk_seconds:
        sys.exit(f"ERROR: chunk_overlap_seconds must be between 0 and the "
                 f"chunk-minutes limit ({max_chunk_seconds:.0f}s) exclusive")
    if type(opt["align_samples"]) is not int or opt["align_samples"] < 1:
        sys.exit("ERROR: align_samples must be a positive integer")
    if opt["align_search_seconds"] <= 0:
        sys.exit("ERROR: align_search_seconds must be positive")
    providers = toml_globals.get("providers", {})
    day_options = overrides or {}
    for stage in ("transcribe", "segment"):
        provider_key, model_key = f"{stage}_provider", f"{stage}_model"
        explicit_provider = opt[provider_key]
        model = opt[model_key]
        if explicit_provider:
            event_provider = toml_globals.get(provider_key)
            if not event_provider and stage == "segment":
                event_model = toml_globals.get(model_key)
                if isinstance(event_model, str) and "/" in event_model:
                    event_provider = event_model.split("/", 1)[0]
            event_provider = event_provider or "mistral"
            day_provider = day_options.get(provider_key, event_provider)
            # Provider switches carry forward only the matching TOML overrides.
            scoped = {}
            if event_provider == explicit_provider:
                scoped.update(toml_globals)
            if day_provider == explicit_provider:
                scoped.update(day_options)
            model = getattr(cli, model_key, None)
            if model is None:
                model = scoped.get(model_key)
            if stage == "segment" and getattr(cli, "effort", None) is None:
                opt["effort"] = scoped.get("effort")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            sys.exit(f"ERROR: {model_key} must be a nonempty string")
        # A qualified legacy pi model can still select its own provider.
        model_provider = model.split("/", 1)[0] if stage == "segment" and model and "/" in model else None
        provider = explicit_provider or model_provider or "mistral"
        defaults = PROVIDER_DEFAULTS.get(provider, {}).get(stage, {})
        settings = providers.get(provider, {}).get(stage, {})
        model = model or settings.get("model") or defaults.get("model")
        if stage == "segment":
            prefix, sep, _ = model.partition("/")
            if sep and prefix != provider:
                sys.exit(f"ERROR: segment_model {model!r} does not match segment_provider {provider!r}")
            if not sep and (explicit_provider or opt[model_key] is None):
                model = f"{provider}/{model}"
            opt["effort"] = opt["effort"] or settings.get("effort") or DEFAULT_EFFORT
            if not isinstance(opt["effort"], str):
                sys.exit("ERROR: effort must be a string")
        opt[provider_key] = provider
        opt[model_key] = model
    return opt


def run_pipeline(opt: dict, source: dict, day_title: str | None,
                 accts: list[Accounting] | None = None) -> None:
    """Run all stages for one input video. day_title is set in config mode, where the
    video is one pre-cut day and the segment stage must not guess days."""
    url, path = source.get("url"), source.get("path")
    if source.get("workdir"):
        workdir = Path(source["workdir"])
    elif day_title:
        workdir = Path(opt["workdir_root"]) / slugify(day_title)
    elif url:
        workdir = Path(opt["workdir_root"]) / video_id_from_url(url)
    else:
        workdir = Path(opt["workdir_root"]) / slugify(path.stem)
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"workdir: {workdir.resolve()}" + (f" (day: {day_title})" if day_title else ""))

    # API spend ledger: config mode -> one api-usage.csv at the workdir root (rows are
    # itemized per day via the day column); single-video mode -> the video's workdir
    if day_title:
        acct = Accounting(Path(opt["workdir_root"]) / USAGE_CSV_NAME, day=day_title)
    else:
        acct = Accounting(workdir / USAGE_CSV_NAME, day="-")
    if accts is not None:
        accts.append(acct)

    if not opt["skip_download"]:
        if url:
            step_download(url, workdir, opt["height"] or None)
        else:
            step_download_local(path, workdir, title=day_title)
    else:
        log("download: skipped")

    if not opt["no_snap_keyframes"]:
        # probe keyframes once, right after download, so a valid keyframes.json
        # exists before the segment agent runs (it only needs to read it)
        ensure_keyframe_cache(workdir)

    if not opt["skip_transcribe"]:
        step_transcribe(workdir, opt["transcribe_model"], opt["language"],
                        opt["chunk_minutes"], opt["mock_transcript"],
                        max_retries=opt["max_retries"], retry_base=opt["retry_base_seconds"],
                        retry_max=opt["retry_max_seconds"], acct=acct,
                        provider=opt["transcribe_provider"],
                        silence_dbfs=opt["silence_dbfs"],
                        silence_min_seconds=opt["silence_min_seconds"],
                        chunk_silence_min_seconds=opt["chunk_silence_min_seconds"],
                        chunk_overlap_seconds=opt["chunk_overlap_seconds"],
                        coverage_check=opt["transcript_coverage_check"])
    else:
        log("transcribe: skipped")

    if not opt["skip_segment"]:
        data = step_segment(workdir, opt["segment_model"], opt["effort"],
                            opt["pi_bin"], opt["mock_segments"], day_title=day_title,
                            sandbox=opt["sandbox"], acct=acct, provider=opt["segment_provider"])
    else:
        log("segment: skipped")
        data = None
        if not opt["skip_edit"]:
            seg_path = workdir / "segments.json"
            if not seg_path.exists():
                sys.exit(f"ERROR: {seg_path} is missing, but the edit stage needs it "
                         f"(run without --skip-segment, or add --skip-edit)")
            data = json.loads(seg_path.read_text())

    if opt["skip_edit"]:
        log("edit: skipped (--skip-edit)")
    elif (workdir / ".edit-done").exists() and not opt["force_edit"]:
        log("edit: .edit-done marker exists, skipping (use --force-edit to redo)")
    else:
        step_edit(workdir, data, opt["subs"], opt["reencode"], combined=not opt["no_combined"],
                  snap_keyframes=not opt["no_snap_keyframes"], day_title=day_title,
                  align_check=opt["align_check"], align_samples=opt["align_samples"],
                  align_search_seconds=opt["align_search_seconds"],
                  align_repair=opt["align_repair"])
    log(f"DONE. Outputs in {(workdir / 'out').resolve()}")


def main() -> None:
    args = parse_args()
    accts: list[Accounting] = []
    try:
        if args.config:
            if args.url:
                sys.exit("ERROR: pass either a YouTube URL or --config, not both")
            toml_globals, days = load_config(args.config)
            if args.workdir is not None:  # in config mode --workdir is the workdir root
                toml_globals = {**toml_globals, "workdir_root": str(args.workdir)}
            for i, day in enumerate(days, 1):
                opt = resolve_options(args, toml_globals, day["overrides"])
                title = day["title"] or f"Day {i}"
                log(f"=== day {i}/{len(days)}: {title} ({day['url'] or day['path']}) ===")
                run_pipeline(opt, day, day_title=title, accts=accts)
        else:
            if not args.url:
                sys.exit("ERROR: a YouTube URL/video ID is required without --config (see --help)")
            opt = resolve_options(args, {})
            run_pipeline(opt, {"url": args.url, "workdir": args.workdir}, day_title=None,
                         accts=accts)
    finally:
        # report the total API usage after any command, even a failed one
        report_api_usage(accts)


if __name__ == "__main__":
    main()
