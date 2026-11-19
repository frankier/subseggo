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
  3. segment     : `pi` oneshot (Mistral-Large-3, medium effort) over the transcript,
                   run in a bubblewrap read-only sandbox when available (--sandbox;
                   the workdir is read-only except agent-scratch/, so the agent cannot
                   clobber pipeline files). pi always runs with a BARE config:
                   PI_CODING_AGENT_DIR points at a fresh empty dir in agent-scratch,
                   so the user's ~/.pi/agent (settings, extensions, custom models,
                   auth) is never read -- same as inside the Docker image; the API
                   key comes from MISTRAL_API_KEY in the environment
                   -> segments.json (intermediate format: segments > sections; days are
                   never guessed -- with --config each video is one day given by the
                   config entry's title, single-video runs get no day level)
  4. edit        : segments.json -> editlist.yaml -> ffmpeg-editlist (--srt)
                   -> per-segment mp4 outputs with break sections removed,
                      plus the original mp4 with subtitles applied.
                   Cut points are snapped to video keyframes (raw.mp4 is probed once,
                   right after download, cached in workdir/keyframes.json; the cache
                   is fingerprint-bound to raw.mp4, validated on read and written
                   atomically, so a stale/corrupt cache is re-probed, not fatal)
                   so the stream-copied cuts start
                   exactly where the editlist says -- otherwise each range's video
                   begins at the next keyframe and the (nominal-time) subtitles drift
                   against it, accumulating a visible offset at every cut. A sync
                   report (out/sync-report.txt) verifies per output that the actual
                   video range starts match the nominal editlist times (--no-snap-keyframes
                   disables the snapping).

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
Requires env: MISTRAL_API_KEY (transcription API and, because the segment pi runs
with a bare config dir, pi's mistral auth too -- pi falls back to this env var,
exactly as in the Docker image).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

# List prices used to turn API usage into USD (the APIs return usage, not cost).
# pi reports its own catalog-derived cost in the session transcript; these are the
# fallback for chat tokens and the primary source for transcription (billed per audio
# hour). Override via env if Mistral changes pricing.
DEFAULT_TRANSCRIBE_USD_PER_AUDIO_HOUR = 0.24   # voxtral-small-transcribe-3
DEFAULT_CHAT_USD_PER_1M = {"input": 2.0, "output": 6.0, "cache_read": 0.2, "cache_write": 0.0}  # Mistral Large line

PI_SESSION_NAME = "pi-session.jsonl"      # pi's native session transcript (JSONL) in the workdir
PI_TRANSCRIPT_NAME = "pi-transcript.json"  # plain-JSON extraction of it (messages + usage totals)
PI_AGENT_DIR_NAME = "pi-agent"             # bare pi config dir inside agent-scratch (PI_CODING_AGENT_DIR)
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


def transcribe_usage_fields(usage: dict) -> dict:
    """Mistral /v1/audio/transcriptions usage -> accounting fields."""
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


def transcribe_cost_usd(usage: dict) -> float:
    """Cost of one transcription request. Mistral bills transcription per audio hour
    (usage.prompt_audio_seconds); fall back to token pricing if absent."""
    per_hour = float(os.environ.get("MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR",
                                    DEFAULT_TRANSCRIBE_USD_PER_AUDIO_HOUR))
    secs = usage.get("prompt_audio_seconds")
    if secs:
        return float(secs) / 3600.0 * per_hour
    return ((usage.get("prompt_tokens") or 0) * chat_price_per_1m("input")
            + (usage.get("completion_tokens") or 0) * chat_price_per_1m("output")) / 1e6


def estimate_chat_cost_usd(totals: dict) -> float:
    """Fallback USD cost of pi usage totals (pi usually reports cost itself in the
    session; this is for sessions without it)."""
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
    for acct in accts:
        rows = ledger_rows(acct.csv_path)
        if rows or acct.csv_path.exists():
            by_csv.setdefault(acct.csv_path, []).extend(rows)
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


def snap_to_keyframe(t: float, kfs: list[list[float]]) -> float:
    """Editlist start value that makes a stream-copy cut begin exactly at a keyframe.

    ffmpeg `-ss` (output option) with `-c copy` drops packets while dts < ss, so a cut
    at `t` actually starts at the first keyframe with dts >= t.  Returning that
    keyframe's dts as the new start keeps the keyframe (dts >= ss) and shrinks the
    leading hole to pts-dts (sub-frame); the sidecar .srt is rebased by the same
    value, so subtitles stay aligned with the video timeline."""
    for _pts, dts in kfs:
        if dts >= t - 1e-3:
            return max(0.0, dts - 0.001)
    return t


def hms_ms(t: float) -> str:
    """HH:MM:SS.mmm (ffmpeg-editlist's seconds() and ffmpeg -ss both accept this)."""
    t = max(0.0, float(t))
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def hms_to_sec(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


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
                     retry_max: float = 120.0, timeout: int = 3600) -> tuple[list[dict], dict]:
    """POST one audio chunk to /v1/audio/transcriptions, retrying 5xx/429/network
    errors with exponential backoff + jitter. 4xx (other than 429) is fatal.
    Returns (segments, usage) -- the API's usage dict (audio seconds + tokens)."""
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
                usage = body.get("usage") or {}
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


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, s in enumerate(segments, 1):
        text = re.sub(r"\s+", " ", s["text"]).strip()
        lines.append(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{text}\n")
    path.write_text("\n".join(lines))


def step_transcribe(workdir: Path, model: str, language: str | None,
                    chunk_minutes: int, mock_transcript: Path | None,
                    max_retries: int = 5, retry_base: float = 2.0,
                    retry_max: float = 120.0,
                    acct: Accounting | None = None) -> list[dict]:
    transcript_json = workdir / "transcript.json"
    if transcript_json.exists():
        log("transcribe: transcript.json exists, skipping")
        return json.loads(transcript_json.read_text())["segments"]
    total_usage: dict = {}
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
                payload = json.loads(cache_file.read_text())
                raw_segs = payload["segments"]
                # usage of a cached chunk was already recorded in the ledger when the
                # API call was made; only accumulate it into the transcript totals
                accumulate_usage(total_usage, payload.get("usage"))
            else:
                chunk = workdir / f"chunk-{idx:03d}.mp3"
                log(f"transcribe: chunk {idx} [{start:.0f}s +{dur:.0f}s] of {duration:.0f}s")
                extract_audio_chunk(raw_mp4, start, dur, chunk)
                raw_segs, usage = transcribe_chunk(chunk, model, language, api_key,
                                                   max_retries=max_retries, retry_base=retry_base,
                                                   retry_max=retry_max)
                chunk.unlink(missing_ok=True)
                if acct is not None:
                    acct.record(step="transcribe", model=model,
                                detail=f"chunk-{idx:03d} [{start:.0f}s +{dur:.0f}s]",
                                **transcribe_usage_fields(usage),
                                cost_usd=transcribe_cost_usd(usage))
                accumulate_usage(total_usage, usage)
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text(json.dumps({"model": model, "usage": usage, "segments": raw_segs}))
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
    doc = {"model": model, "segments": segments}
    if total_usage:
        doc["usage"] = total_usage
        doc["cost_usd"] = round(transcribe_cost_usd(total_usage), 6)
    (workdir / "transcript.json").write_text(json.dumps(doc, indent=2))
    write_srt(segments, workdir / "raw.srt")
    (workdir / "transcript.txt").write_text(
        "\n".join(f"[{srt_ts(s['start'])[:8]}] {s['text'].strip()}" for s in segments))
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
                       model_fallback: str) -> dict | None:
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
    cost = totals["cost"]["total"] or estimate_chat_cost_usd(totals)
    n_req = sum(1 for m in transcript["messages"]
                if m["role"] == "assistant" and m.get("usage"))
    if acct is not None:
        acct.record(step="segment", model=transcript.get("model") or model_fallback,
                    detail=f"pi oneshot (session {str(transcript.get('session_id'))[:8]})",
                    requests=max(n_req, 1), **pi_usage_fields(totals), cost_usd=cost)
    log(f"segment: pi transcript -> {session_out.name} + {PI_TRANSCRIPT_NAME} "
        f"({n_req} api response(s), {totals['totalTokens']} tokens, ${cost:.4f})")
    return transcript


def step_segment(workdir: Path, model: str, effort: str, pi_bin: str,
                 mock_segments: Path | None, day_title: str | None = None,
                 sandbox: str = "auto", acct: Accounting | None = None) -> dict:
    """Segment one video into segments > sections. day_title (config mode only) is
    the day this video belongs to, given by the caller -- passed to the prompt as
    context; the model never guesses days."""
    segments_json = workdir / "segments.json"
    if segments_json.exists():
        log("segment: segments.json exists, skipping")
        return json.loads(segments_json.read_text())
    duration = probe_duration(workdir / "raw.mp4")
    if mock_segments:
        log(f"segment: using mock segments {mock_segments}")
        data = json.loads(mock_segments.read_text())
    else:
        transcript = json.loads((workdir / "transcript.json").read_text())["segments"]
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
        scratch = (workdir / "agent-scratch").resolve()
        scratch.mkdir(exist_ok=True)
        session_scratch = scratch / PI_SESSION_NAME
        session_scratch.unlink(missing_ok=True)  # fresh session per attempt
        (workdir / PI_SESSION_NAME).unlink(missing_ok=True)
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
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, env=env)
        save_pi_transcript(workdir, session_scratch, acct, model)
        if proc.returncode != 0:
            sys.exit(f"ERROR: pi failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
        (workdir / "segment-raw-output.txt").write_text(proc.stdout)
        data = extract_json_block(proc.stdout)
    data = validate_segments(data, duration)
    segments_json.write_text(json.dumps(data, indent=2))
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
# Stage 4: intermediate format -> ffmpeg-editlist YAML -> apply
# --------------------------------------------------------------------------- #

def slugify(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\- ]+", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:maxlen].strip("-") or "segment"


def hms(t: float) -> str:
    return fmt_hms(float(t))


def build_editlist_yaml(data: dict, combined: bool, keyframes: list[list[float]] | None = None) -> list[dict]:
    """Convert segments.json (segments>sections) to an ffmpeg-editlist YAML list.

    One output per segment (icebreaker/episode), containing only non-break
    sections. Optionally one combined output with all breaks removed.

    If `keyframes` is given, every range start is snapped to the keyframe a
    stream-copied cut would actually start at (see snap_to_keyframe), so the actual
    video timeline matches the nominal timeline the sidecar .srt is rebased onto.
    Times are emitted with millisecond precision (hms_ms).
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
            s, e = float(sec["start"]), float(sec["end"])
            if keyframes:
                snapped = snap_to_keyframe(s, keyframes)
                if snapped < e:  # keep degenerate tiny sections unsnapped
                    s = snapped
            out.append((s, e, sec.get("title", "")))
        if not out:
            out = [(seg["start"], seg["end"], seg.get("title", ""))]
        return out

    for seg_idx, seg in enumerate(segments_of(data), 1):
        name = f"{seg_idx:02d}-{slugify(seg.get('title', seg['kind']))}.mp4"
        entry = {"output": name, "editlist": []}
        for (s, e, _title) in ranges_for(seg):
            entry["editlist"].append({"start": hms_ms(s)})
            entry["editlist"].append({"stop": hms_ms(e)})
        doc.append(entry)

    if combined:
        entry = {"output": "full-nobreaks.mp4", "editlist": []}
        all_ranges = []
        for seg in segments_of(data):
            all_ranges.extend(ranges_for(seg))
        all_ranges.sort()
        merged: list[list] = []
        for s, e, _title in all_ranges:
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            entry["editlist"].append({"start": hms_ms(s)})
            entry["editlist"].append({"stop": hms_ms(e)})
        doc.append(entry)
    return doc


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
              snap_keyframes: bool = True, day_title: str | None = None) -> None:
    workdir = workdir.resolve()
    outdir = (workdir / "out").resolve()
    outdir.mkdir(exist_ok=True)
    kfs: list[list[float]] | None = None
    if snap_keyframes:
        kf_cache = workdir / "keyframes.json"
        kfs = load_keyframe_cache(kf_cache, workdir / "raw.mp4")
        if kfs is not None:
            log(f"edit: loaded {len(kfs)} keyframes from {kf_cache.name}")
        else:
            log("edit: probing video keyframes of raw.mp4 (cached in keyframes.json)")
            kfs = probe_keyframes(workdir / "raw.mp4")
            write_keyframe_cache(kf_cache, workdir / "raw.mp4", kfs)
            log(f"edit: {len(kfs)} keyframes found")
    doc = build_editlist_yaml(data, combined=combined, keyframes=kfs)
    editlist_yaml = workdir / "editlist.yaml"
    editlist_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    log(f"edit: wrote {editlist_yaml} ({len(doc)-1} outputs"
        + (", cut points snapped to keyframes)" if kfs else ")"))

    fel = [sys.executable, "-m", "ffmpeg_editlist"]
    cmd = fel + [str(editlist_yaml), str(workdir), "-o", str(outdir),
                 "--srt", "--no-mkv-props", "-f"]
    if reencode:
        cmd.append("--reencode")
    run(cmd, cwd=workdir)
    write_chapters(data, outdir, day_title=day_title)

    # Sync report: verify each output's actual video timeline matches the nominal
    # editlist timeline that the sidecar .srt was rebased onto.
    report = [f"# sync report {time.strftime('%Y-%m-%dT%H:%M:%S')}",
              "# drift = actual video range start - nominal (cumulative) editlist start"]
    for entry in doc[1:]:
        mp4 = outdir / entry["output"]
        if not mp4.exists():
            log(f"edit: WARNING output missing: {mp4}")
            continue
        el = entry["editlist"]
        ranges = [(hms_to_sec(el[i]["start"]), hms_to_sec(el[i + 1]["stop"]))
                  for i in range(0, len(el) - 1, 2)]
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
            f"{entry['output']}: video_start={starts.get('video', 0.0):.3f} "
            f"audio_start={starts.get('audio', 0.0):.3f} srt_first={first_cue:.3f} "
            f"ranges={len(ranges)} max_drift={max_drift * 1000:.0f}ms")
        for i, (cum, act) in enumerate(zip(cums, actual)):
            report.append(f"  range {i + 1}: nominal {cum:.3f} -> actual {act:.3f} "
                          f"(drift {shifts[i] * 1000:+.0f} ms)")
        if max_drift > 0.5:
            log(f"edit: WARNING sync drift {max_drift * 1000:.0f} ms in {mp4.name} "
                f"(see {outdir / 'sync-report.txt'})")
        else:
            log(f"edit: sync ok {mp4.name}: max drift {max_drift * 1000:.0f} ms "
                f"over {len(ranges)} range(s)")
    (outdir / "sync-report.txt").write_text("\n".join(report) + "\n")
    log(f"edit: sync report -> {outdir / 'sync-report.txt'}")

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
# CLI / TOML config
# --------------------------------------------------------------------------- #

# Built-in defaults for every pipeline option. Resolution order:
# explicit CLI flag > TOML config value (global, then per-[[days]] override) > default.
# argparse therefore uses default=None for all of these and they are resolved later.
BUILTIN_DEFAULTS = {
    "workdir_root": "work",          # config mode: base dir for per-day workdirs
    "height": 1080,
    "language": None,
    "transcribe_model": DEFAULT_TRANSCRIBE_MODEL,
    "segment_model": DEFAULT_SEGMENT_MODEL,
    "effort": DEFAULT_EFFORT,
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
    ap.add_argument("--transcribe-model", default=None)
    ap.add_argument("--segment-model", default=None,
                    help="pi model for segmentation (Mistral Large line; literal Large 3 id: mistral/mistral-large-2512)")
    ap.add_argument("--effort", default=None,
                    help="pi thinking/effort level for segmentation")
    ap.add_argument("--chunk-minutes", type=int, default=None,
                    help="audio chunk size for transcription (API max is 60)")
    ap.add_argument("--subs", choices=["embed", "burn", "sidecar"], default=None,
                    help="how to apply subtitles to outputs (default: embed as mov_text)")
    ap.add_argument("--reencode", action="store_true", default=None,
                    help="re-encode segments in ffmpeg-editlist (seamless cuts, slower)")
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
                    help="do not snap editlist cut points to video keyframes (stream-copied cuts "
                         "then start at the next keyframe and subtitles drift; the sync report "
                         "in out/sync-report.txt is still written)")
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
        out[key] = v
    return out


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
    globals_ = _norm_keys(raw, str(path))
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
                        retry_max=opt["retry_max_seconds"], acct=acct)
    else:
        log("transcribe: skipped")

    if not opt["skip_segment"]:
        data = step_segment(workdir, opt["segment_model"], opt["effort"],
                            opt["pi_bin"], opt["mock_segments"], day_title=day_title,
                            sandbox=opt["sandbox"], acct=acct)
    else:
        log("segment: skipped")
        data = json.loads((workdir / "segments.json").read_text())

    if opt["skip_edit"]:
        log("edit: skipped (--skip-edit)")
    elif (workdir / ".edit-done").exists() and not opt["force_edit"]:
        log("edit: .edit-done marker exists, skipping (use --force-edit to redo)")
    else:
        step_edit(workdir, data, opt["subs"], opt["reencode"], combined=not opt["no_combined"],
                  snap_keyframes=not opt["no_snap_keyframes"], day_title=day_title)
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
