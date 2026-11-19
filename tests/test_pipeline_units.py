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
