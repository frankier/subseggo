# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyyaml",
#   "requests",
# ]
# ///
"""Unit tests for pipeline.py helpers (keyframe cache, probe parsing, sandbox).

Run from the repo root:
    uv run --script tests/test_pipeline_units.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline  # noqa: E402


class KeyframeCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.mp4 = self.dir / "raw.mp4"
        self.mp4.write_bytes(b"\x00" * (2 << 20) + b"tail")
        self.cache = self.dir / "keyframes.json"

    def test_roundtrip_is_atomic(self):
        kfs = [[0.0, -0.03], [5.0, 5.0]]
        pipeline.write_keyframe_cache(self.cache, self.mp4, kfs)
        self.assertFalse((self.dir / "keyframes.json.tmp").exists())  # no tmp left
        payload = json.loads(self.cache.read_text())
        self.assertEqual(payload["version"], pipeline.KF_CACHE_VERSION)
        self.assertEqual(payload["fingerprint"], pipeline.file_fingerprint(self.mp4))
        self.assertEqual(pipeline.load_keyframe_cache(self.cache, self.mp4), kfs)

    def test_corrupt_csv_is_moved_aside_not_fatal(self):
        # the incident: raw ffprobe CSV redirected into keyframes.json
        self.cache.write_text("0.000000,-0.033333,K__\n0.166667,0.000000,___\n")
        self.assertIsNone(pipeline.load_keyframe_cache(self.cache, self.mp4))
        self.assertFalse(self.cache.exists())
        bad = list(self.dir.glob("keyframes.json.bad-*"))
        self.assertEqual(len(bad), 1)
        self.assertIn("K__", bad[0].read_text())

    def test_bad_schema_is_moved_aside(self):
        self.cache.write_text(json.dumps({"keyframes": "nope"}))
        self.assertIsNone(pipeline.load_keyframe_cache(self.cache, self.mp4))
        self.assertEqual(len(list(self.dir.glob("keyframes.json.bad-*"))), 1)

    def test_missing_file_returns_none(self):
        self.assertIsNone(pipeline.load_keyframe_cache(self.cache, self.mp4))

    def test_stale_fingerprint_is_ignored(self):
        pipeline.write_keyframe_cache(self.cache, self.mp4, [[1.0, 1.0]])
        with self.mp4.open("ab") as f:  # raw.mp4 changed -> cache is stale
            f.write(b"x")
        self.assertIsNone(pipeline.load_keyframe_cache(self.cache, self.mp4))
        # stale cache is ignored but not deleted (overwritten on re-probe)
        self.assertTrue(self.cache.exists())

    def test_legacy_bare_floats_normalized(self):
        payload = {"version": pipeline.KF_CACHE_VERSION, "source": "raw.mp4",
                   "fingerprint": pipeline.file_fingerprint(self.mp4),
                   "keyframes": [0.0, 5.5]}
        self.cache.write_text(json.dumps(payload))
        self.assertEqual(pipeline.load_keyframe_cache(self.cache, self.mp4),
                         [[0.0, 0.0], [5.5, 5.5]])

    def test_ensure_probes_once_then_uses_cache(self):
        with mock.patch.object(pipeline, "probe_keyframes",
                               return_value=[[1.0, 1.0]]) as pk:
            pipeline.ensure_keyframe_cache(self.dir)
            pipeline.ensure_keyframe_cache(self.dir)
            pk.assert_called_once_with(self.mp4)
        self.assertTrue(self.cache.exists())
        # a corrupt cache written between runs is recovered by re-probing
        self.cache.write_text("garbage,csv\n")
        with mock.patch.object(pipeline, "probe_keyframes",
                               return_value=[[2.0, 2.0]]) as pk:
            pipeline.ensure_keyframe_cache(self.dir)
            pk.assert_called_once()
        self.assertEqual(pipeline.load_keyframe_cache(self.cache, self.mp4), [[2.0, 2.0]])


class ProbeKeyframesTests(unittest.TestCase):
    def test_parses_keyframes_and_skips_bad_lines(self):
        csv_out = ("0.000000,-0.033333,K__\n"
                   "N/A,N/A,K__\n"          # unparseable timestamps: skipped
                   "0.166667,0.000000,___\n"  # not a keyframe: skipped
                   "5.000000,5.000000,K__\n")
        fake = subprocess.CompletedProcess([], 0, stdout=csv_out, stderr="")
        with mock.patch.object(pipeline.subprocess, "run", return_value=fake), \
                mock.patch.object(pipeline, "which_or_die", side_effect=lambda n: n):
            kfs = pipeline.probe_keyframes(Path("raw.mp4"))
        self.assertEqual(kfs, [[0.0, -0.033333], [5.0, 5.0]])


def make_session_jsonl() -> str:
    """A small synthetic pi session (JSONL) with a tool call and two assistant
    messages carrying usage+cost."""
    entries = [
        {"type": "session", "version": 3, "id": "abc123-def", "timestamp": "2026-01-01T10:00:00.000Z", "cwd": "/w"},
        {"type": "model_change", "id": "e1", "parentId": None, "timestamp": "2026-01-01T10:00:01.000Z",
         "provider": "mistral", "modelId": "mistral-large-2512"},
        {"type": "message", "id": "e2", "parentId": "e1", "timestamp": "2026-01-01T10:00:02.000Z",
         "message": {"role": "user", "content": "segment this", "timestamp": 1}},
        {"type": "message", "id": "e3", "parentId": "e2", "timestamp": "2026-01-01T10:00:10.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "hmm"},
             {"type": "text", "text": "let me look"},
             {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "ffprobe ..."}}],
             "api": "mistral-chat", "provider": "mistral", "model": "mistral-large-2512",
             "usage": {"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1, "totalTokens": 18,
                       "cost": {"input": 2e-5, "output": 3e-5, "cacheRead": 4e-6, "cacheWrite": 0.0, "total": 5.4e-5}},
             "stopReason": "toolUse", "timestamp": 2}},
        {"type": "message", "id": "e4", "parentId": "e3", "timestamp": "2026-01-01T10:00:11.000Z",
         "message": {"role": "toolResult", "toolCallId": "t1", "toolName": "bash",
                     "content": [{"type": "text", "text": "0.0,K__"}], "isError": False, "timestamp": 3}},
        {"type": "message", "id": "e5", "parentId": "e4", "timestamp": "2026-01-01T10:00:20.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "```json ... ```"}],
                     "api": "mistral-chat", "provider": "mistral", "model": "mistral-large-2512",
                     "usage": {"input": 20, "output": 7, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 27,
                               "cost": {"input": 4e-5, "output": 4.2e-5, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 8.2e-5}},
                     "stopReason": "stop", "timestamp": 4}},
        {"type": "usage", "id": "e6", "parentId": "e5", "timestamp": "2026-01-01T10:00:21.000Z",
         "kind": "cache_warm", "provider": "mistral", "model": "mistral-large-2512",
         "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}},
    ]
    return "\n".join(json.dumps(e) for e in entries) + "\n"


class ExtractPiTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.session = self.dir / "pi-session.jsonl"
        self.session.write_text(make_session_jsonl())

    def test_extracts_messages_model_and_usage_totals(self):
        t = pipeline.extract_pi_transcript(self.session)
        self.assertEqual(t["session_id"], "abc123-def")
        self.assertEqual(t["provider"], "mistral")
        self.assertEqual(t["model"], "mistral-large-2512")
        roles = [m["role"] for m in t["messages"]]
        self.assertEqual(roles, ["user", "assistant", "toolResult", "assistant"])
        self.assertEqual(t["messages"][0]["text"], "segment this")
        a1 = t["messages"][1]
        self.assertEqual(a1["text"], "let me look")
        self.assertEqual(a1["thinking"], "hmm")
        self.assertEqual(a1["tool_calls"], [{"id": "t1", "name": "bash",
                                             "arguments": {"command": "ffprobe ..."}}])
        tr = t["messages"][2]
        self.assertEqual(tr["tool_call_id"], "t1")
        self.assertEqual(tr["tool_name"], "bash")
        self.assertFalse(tr["is_error"])
        self.assertEqual(tr["text"], "0.0,K__")
        # usage totals sum the assistant messages only
        self.assertEqual(t["usage"]["input"], 30)
        self.assertEqual(t["usage"]["output"], 12)
        self.assertEqual(t["usage"]["cacheRead"], 2)
        self.assertEqual(t["usage"]["cacheWrite"], 1)
        self.assertEqual(t["usage"]["totalTokens"], 45)
        self.assertAlmostEqual(t["usage"]["cost"]["total"], 5.4e-5 + 8.2e-5)
        # standalone usage entries are kept separately
        self.assertEqual(len(t["other_usage"]), 1)
        self.assertEqual(t["other_usage"][0]["kind"], "cache_warm")

    def test_tolerates_blank_and_corrupt_lines(self):
        self.session.write_text("\nnot json\n" + make_session_jsonl())
        t = pipeline.extract_pi_transcript(self.session)
        self.assertEqual(t["session_id"], "abc123-def")
        self.assertEqual(len(t["messages"]), 4)


class CostTests(unittest.TestCase):
    def test_transcribe_cost_from_audio_seconds(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR", None)
            self.assertAlmostEqual(
                pipeline.transcribe_cost_usd({"prompt_audio_seconds": 3600}), 0.24)
            self.assertAlmostEqual(
                pipeline.transcribe_cost_usd({"prompt_audio_seconds": 1800}), 0.12)

    def test_transcribe_cost_env_override(self):
        with mock.patch.dict("os.environ", {"MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR": "1.20"}):
            self.assertAlmostEqual(
                pipeline.transcribe_cost_usd({"prompt_audio_seconds": 3600}), 1.20)

    def test_transcribe_cost_token_fallback(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MISTRAL_TRANSCRIBE_USD_PER_AUDIO_HOUR", None)
            os.environ.pop("MISTRAL_USD_PER_1M_INPUT", None)
            os.environ.pop("MISTRAL_USD_PER_1M_OUTPUT", None)
            cost = pipeline.transcribe_cost_usd({"prompt_tokens": 1_000_000,
                                                  "completion_tokens": 1_000_000})
            self.assertAlmostEqual(cost, 2.0 + 6.0)  # input + output per-1M defaults

    def test_estimate_chat_cost(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            for k in ("MISTRAL_USD_PER_1M_INPUT", "MISTRAL_USD_PER_1M_OUTPUT",
                      "MISTRAL_USD_PER_1M_CACHE_READ", "MISTRAL_USD_PER_1M_CACHE_WRITE"):
                os.environ.pop(k, None)
            cost = pipeline.estimate_chat_cost_usd(
                {"input": 1_000_000, "output": 500_000, "cacheRead": 1_000_000, "cacheWrite": 0})
            self.assertAlmostEqual(cost, 2.0 + 3.0 + 0.2)

    def test_usage_field_mappings(self):
        self.assertEqual(
            pipeline.transcribe_usage_fields({"prompt_tokens": 4, "completion_tokens": 6,
                                              "total_tokens": 10, "prompt_audio_seconds": 20}),
            {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10, "audio_seconds": 20.0})
        self.assertEqual(
            pipeline.pi_usage_fields({"input": 1, "output": 2, "cacheRead": 3,
                                      "cacheWrite": 4, "totalTokens": 10}),
            {"input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 3,
             "cache_write_tokens": 4, "total_tokens": 10})


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.csv = self.dir / "api-usage.csv"
        self.acct = pipeline.Accounting(self.csv, day="Day 1")

    def test_record_appends_rows_with_header_once(self):
        self.acct.record(step="transcribe", detail="chunk-000 [0s +3300s]",
                         model="voxtral-small-transcribe-3",
                         **pipeline.transcribe_usage_fields(
                             {"prompt_tokens": 4, "completion_tokens": 6,
                              "total_tokens": 10, "prompt_audio_seconds": 3300}),
                         cost_usd=pipeline.transcribe_cost_usd({"prompt_audio_seconds": 3300}))
        self.acct.record(step="segment", detail="pi oneshot (session abc123)",
                         model="mistral-large-2512", requests=2,
                         **pipeline.pi_usage_fields({"input": 30, "output": 12,
                                                     "cacheRead": 2, "cacheWrite": 1,
                                                     "totalTokens": 45}),
                         cost_usd=0.000136)
        text = self.csv.read_text()
        self.assertEqual(text.count("timestamp,day,step"), 1)  # header written once
        rows = pipeline.ledger_rows(self.csv)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["day"], "Day 1")
        self.assertEqual(rows[0]["step"], "transcribe")
        self.assertEqual(rows[0]["audio_seconds"], "3300.0")
        self.assertEqual(rows[1]["step"], "segment")
        self.assertEqual(rows[1]["requests"], "2")
        self.assertEqual(rows[1]["cache_read_tokens"], "2")
        t = self.acct.run_totals()
        self.assertEqual(t["requests"], 3)
        self.assertEqual(t["total_tokens"], 55)
        self.assertAlmostEqual(t["audio_seconds"], 3300.0)
        self.assertAlmostEqual(t["cost_usd"], round(3300 / 3600 * 0.24, 6) + 0.000136, places=6)

    def test_ledger_of_missing_csv_is_empty(self):
        self.assertEqual(pipeline.ledger_rows(self.dir / "nope.csv"), [])

    def test_report_api_usage_prints_run_and_ledger_totals(self):
        import contextlib
        import io
        self.acct.record(step="segment", detail="pi oneshot", model="m",
                         total_tokens=45, cost_usd=0.000136)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline.report_api_usage([self.acct])
            pipeline.report_api_usage([])  # no accts: no-op
        out = buf.getvalue()
        self.assertIn("API usage this run", out)
        self.assertIn("[Day 1] segment", out)
        self.assertIn("total API usage (ledger)", out)
        self.assertIn("TOTAL", out)
        self.assertIn(str(self.csv), out)


class PiOneshotCmdTests(unittest.TestCase):
    def test_session_flag_replaces_no_session(self):
        with mock.patch.object(pipeline, "pi_supports_effort", return_value=True), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=True):
            cmd = pipeline.build_pi_oneshot_cmd("pi", "mistral/mistral-large-latest", "medium",
                                                Path("/w/segment-prompt.md"),
                                                Path("/w/agent-scratch/pi-session.jsonl"))
        self.assertIn("--print", cmd)
        self.assertNotIn("--no-session", cmd)
        i = cmd.index("--session")
        self.assertEqual(cmd[i + 1], "/w/agent-scratch/pi-session.jsonl")
        self.assertIn("--model", cmd)
        self.assertIn("--effort", cmd)
        self.assertIn("--tools", cmd)
        self.assertTrue(any(a.startswith("@") and a.endswith("segment-prompt.md") for a in cmd))

    def test_optional_flags_dropped_when_unsupported(self):
        with mock.patch.object(pipeline, "pi_supports_effort", return_value=False), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=False):
            cmd = pipeline.build_pi_oneshot_cmd("pi", "m", "medium", Path("/w/p.md"), Path("/w/s.jsonl"))
        self.assertNotIn("--effort", cmd)
        self.assertNotIn("--tools", cmd)


class StepSegmentTranscriptTests(unittest.TestCase):
    """step_segment end-to-end with a fake pi: the session file must land in the
    workdir (pi-session.jsonl + pi-transcript.json) and the usage CSV row recorded."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.workdir = self.dir / "work"
        self.workdir.mkdir()
        (self.workdir / "raw.info.json").write_text(json.dumps(
            {"id": "x", "title": "t", "uploader": "u", "description": "", "duration": 100}))
        (self.workdir / "transcript.json").write_text(json.dumps(
            {"model": "m", "segments": [{"start": 0.0, "end": 10.0, "text": "hi"}]}))
        self.acct = pipeline.Accounting(self.dir / "api-usage.csv", day="Day 1")

    def _fake_pi_run(self, cmd, **kw):
        i = cmd.index("--session")
        Path(cmd[i + 1]).write_text(make_session_jsonl())
        out = '```json\n{"segments": [{"kind": "episode", "title": "ep", "start": 0, "end": 100,'
        out += ' "sections": [{"kind": "content", "title": "c", "start": 0, "end": 100}]}]}\n```\n'
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    def test_segment_saves_transcript_and_accounts(self):
        with mock.patch.object(pipeline, "probe_duration", return_value=100.0), \
                mock.patch.object(pipeline, "pi_supports_effort", return_value=True), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=True), \
                mock.patch.object(pipeline.subprocess, "run", side_effect=self._fake_pi_run):
            data = pipeline.step_segment(self.workdir, "mistral/mistral-large-latest", "medium",
                                         "pi", None, day_title="Day 1", sandbox="none",
                                         acct=self.acct)
        self.assertEqual(len(data["segments"]), 1)
        self.assertTrue((self.workdir / "pi-session.jsonl").exists())
        tr = json.loads((self.workdir / "pi-transcript.json").read_text())
        self.assertEqual(tr["model"], "mistral-large-2512")
        self.assertEqual(tr["usage"]["totalTokens"], 45)
        self.assertAlmostEqual(tr["usage"]["cost"]["total"], 5.4e-5 + 8.2e-5)
        self.assertTrue((self.workdir / "segments.json").exists())
        # one ledger row for the pi run, with pi's own cost
        rows = pipeline.ledger_rows(self.dir / "api-usage.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], "segment")
        self.assertEqual(rows[0]["day"], "Day 1")
        self.assertEqual(rows[0]["model"], "mistral-large-2512")
        self.assertEqual(rows[0]["requests"], "2")
        self.assertAlmostEqual(float(rows[0]["cost_usd"]), 5.4e-5 + 8.2e-5, places=9)

    def test_save_pi_transcript_without_session_file(self):
        self.assertIsNone(pipeline.save_pi_transcript(self.workdir, self.dir / "nope.jsonl",
                                                      self.acct, "m"))
        self.assertEqual(self.acct.rows, [])


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_resolve_none(self):
        self.assertFalse(pipeline.resolve_sandbox("none"))

    def test_resolve_auto_uses_bwrap_when_present(self):
        with mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/bwrap"):
            self.assertTrue(pipeline.resolve_sandbox("auto"))

    def test_resolve_bwrap_missing_is_fatal(self):
        with mock.patch.object(pipeline.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit):
                pipeline.resolve_sandbox("bwrap")

    def test_resolve_auto_missing_falls_back_with_warning(self):
        with mock.patch.object(pipeline.shutil, "which", return_value=None):
            self.assertFalse(pipeline.resolve_sandbox("auto"))

    def test_bwrap_command_makes_only_scratch_writable(self):
        with mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/bwrap"):
            cmd = pipeline.bwrap_command(["pi", "-p", "hi"], self.dir)
        self.assertEqual(cmd[0], "/usr/bin/bwrap")
        self.assertEqual(cmd[cmd.index("--ro-bind") + 1:cmd.index("--ro-bind") + 3],
                         ["/", "/"])
        i = cmd.index("--bind")
        scratch = str((self.dir / "agent-scratch").resolve())
        self.assertEqual(cmd[i + 1:i + 3], [scratch, scratch])
        self.assertIn("--tmpfs", cmd)
        self.assertIn("/tmp/pi-home", cmd)  # HOME redirected into the tmpfs
        self.assertEqual(cmd[cmd.index("--") + 1:], ["pi", "-p", "hi"])  # cmd intact
        self.assertTrue((self.dir / "agent-scratch").is_dir())


if __name__ == "__main__":
    unittest.main()
