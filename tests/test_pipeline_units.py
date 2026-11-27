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


class ProbeAudioSilencesTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.mp4 = self.dir / "raw.mp4"
        self.mp4.write_bytes(b"\x00" * 4096)

    def _probe(self, stderr: str, returncode: int = 0):
        fake = subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)
        with mock.patch.object(pipeline.subprocess, "run", return_value=fake), \
                mock.patch.object(pipeline, "which_or_die", side_effect=lambda n: n), \
                mock.patch.object(pipeline, "probe_duration", return_value=100.0):
            return pipeline.probe_audio_silences(self.mp4)

    def test_parses_silencedetect_output(self):
        stderr = ("[Parsed_silencedetect_0 @ 0x1] silence_start: 0\n"
                  "[Parsed_silencedetect_0 @ 0x1] silence_end: 428.714853 | "
                  "silence_duration: 428.714853\n"
                  "[Parsed_silencedetect_0 @ 0x1] silence_start: 430.056417\n"
                  "[Parsed_silencedetect_0 @ 0x1] silence_end: 431.044059 | "
                  "silence_duration: 0.987642\n")
        self.assertEqual(self._probe(stderr), [[0.0, 428.714853], [430.056417, 431.044059]])

    def test_trailing_silence_is_clamped_to_duration(self):
        # ffmpeg can end the file without emitting silence_end
        stderr = "[Parsed_silencedetect_0 @ 0x1] silence_start: 90.5\n"
        self.assertEqual(self._probe(stderr), [[90.5, 100.0]])

    def test_failure_means_whole_file_is_speech(self):
        self.assertEqual(self._probe("no audio stream", returncode=1), [])


class SilenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.mp4 = self.dir / "raw.mp4"
        self.mp4.write_bytes(b"\x00" * (1 << 20))
        self.cache = self.dir / "silences.json"
        self.params = pipeline.silence_params(-60.0, 0.5)

    def test_roundtrip_is_atomic(self):
        sil = [[0.0, 428.7], [12309.7, 12657.4]]
        pipeline.write_silence_cache(self.cache, self.mp4, sil, self.params)
        self.assertFalse((self.dir / "silences.json.tmp").exists())
        payload = json.loads(self.cache.read_text())
        self.assertEqual(payload["version"], pipeline.SILENCE_CACHE_VERSION)
        self.assertEqual(payload["params"], self.params)
        self.assertEqual(pipeline.load_silence_cache(self.cache, self.mp4, self.params), sil)

    def test_changed_analysis_params_invalidate(self):
        pipeline.write_silence_cache(self.cache, self.mp4, [[0.0, 1.0]], self.params)
        other = pipeline.silence_params(-45.0, 0.5)
        self.assertIsNone(pipeline.load_silence_cache(self.cache, self.mp4, other))

    def test_stale_fingerprint_is_ignored(self):
        pipeline.write_silence_cache(self.cache, self.mp4, [[0.0, 1.0]], self.params)
        with self.mp4.open("ab") as f:
            f.write(b"x")
        self.assertIsNone(pipeline.load_silence_cache(self.cache, self.mp4, self.params))

    def test_corrupt_cache_is_moved_aside_not_fatal(self):
        self.cache.write_text("silence_start: 0\n")
        self.assertIsNone(pipeline.load_silence_cache(self.cache, self.mp4, self.params))
        self.assertEqual(len(list(self.dir.glob("silences.json.bad-*"))), 1)

    def test_ensure_probes_once_then_uses_cache(self):
        with mock.patch.object(pipeline, "probe_audio_silences",
                               return_value=[[0.0, 10.0]]) as pa:
            pipeline.ensure_silence_cache(self.dir, -60.0, 0.5)
            pipeline.ensure_silence_cache(self.dir, -60.0, 0.5)
            pa.assert_called_once()
        self.assertTrue(self.cache.exists())


class BuildAudioChunksTests(unittest.TestCase):
    MAX = 3300.0

    def chunks(self, duration, silences, **kw):
        kw.setdefault("long_silence_seconds", 30.0)
        kw.setdefault("overlap_seconds", 60.0)
        return pipeline.build_audio_chunks(duration, silences, self.MAX, **kw)

    def test_long_silence_splits_and_is_never_uploaded(self):
        chunks = self.chunks(1000.0, [[400.0, 500.0]])
        self.assertEqual([(c.start, c.end) for c in chunks], [(0.0, 400.0), (500.0, 1000.0)])
        # no chunk straddles the silence, and none of it is inside a chunk
        for c in chunks:
            self.assertFalse(c.start < 450.0 < c.end)

    def test_short_pause_stays_inside_its_chunk(self):
        chunks = self.chunks(1000.0, [[400.0, 420.0]])   # 20s < 30s
        self.assertEqual([(c.start, c.end) for c in chunks], [(0.0, 1000.0)])

    def test_leading_and_trailing_silence_are_trimmed(self):
        chunks = self.chunks(1000.0, [[0.0, 100.0], [900.0, 1000.0]])
        self.assertEqual([(c.start, c.end) for c in chunks], [(100.0, 900.0)])

    def test_oversize_region_splits_into_even_pieces_under_the_limit(self):
        chunks = self.chunks(7000.0, [])
        self.assertEqual(len(chunks), 3)                  # ceil((7000-60)/(3300-60))
        for c in chunks:
            self.assertLessEqual(c.duration, self.MAX + 1e-9)
            self.assertAlmostEqual(c.duration, chunks[0].duration, places=6)
        self.assertAlmostEqual(chunks[0].start, 0.0, places=6)
        self.assertAlmostEqual(chunks[-1].end, 7000.0, places=6)
        # neighbouring pieces overlap by exactly 60s and cover the region without gaps
        for prev, nxt in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(nxt.start, prev.end - 60.0, places=6)

    def test_region_just_over_the_limit_still_fits(self):
        chunks = self.chunks(self.MAX + 1.0, [])
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.duration, self.MAX + 1e-9)
        self.assertAlmostEqual(chunks[-1].end, self.MAX + 1.0, places=6)

    def test_overlap_tail_is_recorded_only_between_split_pieces(self):
        chunks = self.chunks(7000.0, [])
        self.assertEqual(chunks[0].drop_before, chunks[0].start)
        for c in chunks[1:]:
            self.assertAlmostEqual(c.drop_before, c.start + 60.0, places=6)
        # a chunk that follows a silence has no overlap to drop
        after_silence = self.chunks(1000.0, [[400.0, 500.0]])[1]
        self.assertEqual(after_silence.drop_before, after_silence.start)

    def test_all_silence_still_yields_one_chunk(self):
        chunks = self.chunks(500.0, [[0.0, 500.0]])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].duration, 500.0)


class ChunkCoverageTests(unittest.TestCase):
    def report(self, segs, silences, duration=600.0):
        part = pipeline.AudioChunk(0.0, duration, 0.0)
        return pipeline.chunk_coverage_report(0, part, segs, silences)

    def test_well_aligned_chunk_passes(self):
        # cue from 0-100, silence 100-300, cue 300-400
        segs = [{"start": 0.0, "end": 100.0}, {"start": 300.0, "end": 400.0}]
        r = self.report(segs, [[100.0, 300.0]])
        self.assertEqual(r["spurious"], 0.0)
        self.assertTrue(r["ok"])

    def test_cues_sitting_on_silence_are_flagged(self):
        # the day-2 failure mode: text over audio that is measurably empty
        segs = [{"start": 0.0, "end": 100.0}, {"start": 100.0, "end": 300.0}]
        r = self.report(segs, [[100.0, 300.0]])
        self.assertAlmostEqual(r["spurious"], 200.0, places=6)
        self.assertFalse(r["ok"])

    def test_short_silences_are_ignored(self):
        # a cue running over a 1s inter-word pause is normal, not an error
        segs = [{"start": 0.0, "end": 100.0}]
        r = self.report(segs, [[40.0, 41.0]], duration=100.0)
        self.assertEqual(r["spurious"], 0.0)
        self.assertTrue(r["ok"])

    def test_empty_transcript_is_flagged(self):
        r = self.report([], [])
        self.assertAlmostEqual(r["missed_fraction"], 1.0, places=6)
        self.assertFalse(r["ok"])

    def test_short_chunk_with_little_speech_is_not_flagged(self):
        # a brief interjection in a short chunk: low cue time must not look like drift
        segs = [{"start": 0.0, "end": 5.0}]
        r = self.report(segs, [[30.0, 32.0]], duration=70.0)
        self.assertTrue(r["ok"])
        self.assertGreater(r["missed_fraction"], 0.5)  # informational, not gated


class IntervalMathTests(unittest.TestCase):
    def test_union_merges_touching_intervals(self):
        self.assertEqual(pipeline._intervals_union([[0, 5], [5, 9], [20, 21]]),
                         [[0, 9], [20, 21]])

    def test_minus_and_intersect(self):
        self.assertEqual(pipeline._intervals_minus([[0, 100]], [[10, 20], [50, 60]]),
                         [[0, 10], [20, 50], [60, 100]])
        self.assertEqual(pipeline._intervals_intersect([[0, 50]], [[25, 100]]), [[25, 50]])

    def test_empty_intervals_are_dropped(self):
        self.assertEqual(pipeline._intervals_union([[5, 5], [7, 3]]), [])


class SrtTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_compose_parse_roundtrip(self):
        path = self.dir / "a.srt"
        cues = [{"start": 0.0, "end": 4.4, "text": "Hello, hello."},
                {"start": 5.64, "end": 8.54, "text": "So you've been  active."}]
        pipeline.write_srt(cues, path)
        self.assertEqual(pipeline.parse_srt(path),
                         [{"start": 0.0, "end": 4.4, "text": "Hello, hello."},
                          {"start": 5.64, "end": 8.54, "text": "So you've been active."}])

    def test_parse_tolerates_bom_crlf_and_junk(self):
        path = self.dir / "b.srt"
        path.write_bytes("\ufeff1\r\n00:00:01,000 --> 00:00:02,500\r\nhi\r\n\r\n"
                         "not a cue\r\n\r\n"
                         "2\r\n00:00:03.000 --> 00:00:04.000\r\nthere\r\n".encode("utf-8"))
        self.assertEqual([c["text"] for c in pipeline.parse_srt(path)], ["hi", "there"])

    def test_srt_seconds_handles_missing_short_ms(self):
        self.assertAlmostEqual(pipeline.srt_seconds("00:00:01,5"), 1.5)
        self.assertAlmostEqual(pipeline.srt_seconds("01:02:03,250"), 3723.25)


class RebaseSrtTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def rebase(self, cues, ranges):
        out = self.dir / "o.srt"
        pipeline.rebase_srt(cues, ranges, out)
        return pipeline.parse_srt(out)

    def test_offsets_follow_the_range_starts(self):
        cues = [{"start": 100.0, "end": 104.0, "text": "a"},
                {"start": 210.0, "end": 214.0, "text": "b"}]
        out = self.rebase(cues, [(100.0, 200.0), (210.0, 300.0)])
        self.assertAlmostEqual(out[0]["start"], 0.0, places=6)
        # second range starts at output 100s and is 100s long
        self.assertAlmostEqual(out[1]["start"], 100.0, places=6)

    def test_content_in_a_removed_break_disappears(self):
        cues = [{"start": 100.0, "end": 104.0, "text": "a"},
                {"start": 250.0, "end": 254.0, "text": "break talk"},
                {"start": 310.0, "end": 314.0, "text": "b"}]
        out = self.rebase(cues, [(100.0, 200.0), (300.0, 400.0)])
        self.assertEqual([c["text"] for c in out], ["a", "b"])

    def test_cue_crossing_a_cut_is_split_then_merged_not_duplicated(self):
        cues = [{"start": 198.0, "end": 202.0, "text": "spanning"}]
        out = self.rebase(cues, [(100.0, 200.0), (200.0, 300.0)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "spanning")
        self.assertAlmostEqual(out[0]["start"], 98.0, places=6)
        self.assertAlmostEqual(out[0]["end"], 102.0, places=6)


class SnapRangeTests(unittest.TestCase):
    KFS = [[0.0, -0.033], [10.1, 10.0], [20.2, 20.1]]

    def test_nearest_keyframe_and_keyframe_list_not_required(self):
        self.assertEqual(pipeline.snap_range_start(9.0, 30.0, self.KFS)[0], 10.1)
        self.assertEqual(pipeline.snap_range_start(11.0, 30.0, self.KFS)[0], 10.1)
        self.assertEqual(pipeline.snap_range_start(9.0, 30.0, None), (9.0, 30.0))

    def test_section_inside_one_gop_snaps_to_that_gop(self):
        # A stream-copied cut at 19.0 lands on the 10.1 keyframe anyway, so report
        # 10.1: pretending the cut starts at 19.0 would misplace every subtitle after
        # it. The resulting output is longer than the section, which stream copy
        # cannot avoid.
        self.assertEqual(pipeline.snap_range_start(19.0, 19.5, self.KFS), (10.1, 19.5))


class ExpectedRawTimeTests(unittest.TestCase):
    def test_maps_across_ranges(self):
        ranges = [(100.0, 200.0), (300.0, 400.0)]
        self.assertAlmostEqual(pipeline.expected_raw_time(0.0, ranges), 100.0)
        self.assertAlmostEqual(pipeline.expected_raw_time(50.0, ranges), 150.0)
        self.assertAlmostEqual(pipeline.expected_raw_time(100.0, ranges), 300.0)
        self.assertAlmostEqual(pipeline.expected_raw_time(150.0, ranges), 350.0)


class ShiftCuesTests(unittest.TestCase):
    def test_constant_lag_shifts_every_cue(self):
        samples = [{"t": 0.0, "audio": 0.5}, {"t": 100.0, "audio": 0.5}]
        out = pipeline.shift_cues([{"start": 10.0, "end": 20.0, "text": "x"}], samples)
        self.assertAlmostEqual(out[0]["start"], 9.5, places=6)
        self.assertAlmostEqual(out[0]["end"], 19.5, places=6)

    def test_lag_is_interpolated_between_samples(self):
        samples = [{"t": 0.0, "audio": 0.0}, {"t": 100.0, "audio": 2.0}]
        out = pipeline.shift_cues([{"start": 50.0, "end": 50.0, "text": "x"}], samples)
        self.assertAlmostEqual(out[0]["start"], 49.0, places=6)

    def test_rejected_samples_are_ignored(self):
        samples = [{"t": 0.0, "audio": None}, {"t": 100.0, "audio": 1.0}]
        out = pipeline.shift_cues([{"start": 10.0, "end": 20.0, "text": "x"}], samples)
        self.assertAlmostEqual(out[0]["start"], 9.0, places=6)

    def test_no_usable_samples_shifts_nothing(self):
        self.assertEqual(pipeline.shift_cues([{"start": 1.0, "end": 2.0, "text": "x"}],
                                             [{"t": 0.0, "audio": None}]), [])

    def test_shift_never_produces_a_negative_or_inverted_cue(self):
        samples = [{"t": 0.0, "audio": 5.0}, {"t": 100.0, "audio": 5.0}]
        out = pipeline.shift_cues([{"start": 1.0, "end": 1.5, "text": "x"}], samples)
        self.assertEqual(out[0]["start"], 0.0)
        self.assertGreaterEqual(out[0]["end"], out[0]["start"])


class CorrelationTests(unittest.TestCase):
    def test_finds_a_known_offset(self):
        signal = [0.0, 0.0, 1.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        window = [1.0, 5.0, 2.0]
        b = signal[:2] + window + signal[5:]
        k, peak = pipeline.normalized_correlation(window, b, len(b) - len(window))
        self.assertEqual(k, 2)
        self.assertGreater(peak, 0.99)

    def test_flat_signal_has_no_usable_peak(self):
        k, peak = pipeline.normalized_correlation([1.0] * 5, [1.0] * 10, 5)
        self.assertLess(peak, 0.5)

    def test_frame_rmse(self):
        self.assertAlmostEqual(pipeline.frame_rmse(b"\x00\x10", b"\x00\x10"), 0.0)
        self.assertAlmostEqual(pipeline.frame_rmse(b"\x00\x00", b"\x0a\x14"), 15.0)
        self.assertEqual(pipeline.frame_rmse(b"", b""), 255.0)


class BuildOutputRangesTests(unittest.TestCase):
    DATA = {"segments": [
        {"kind": "episode", "title": "One", "start": 0.0, "end": 100.0,
         "sections": [{"kind": "content", "title": "c", "start": 0.0, "end": 60.0},
                      {"kind": "break", "title": "b", "start": 60.0, "end": 80.0},
                      {"kind": "content", "title": "c2", "start": 80.0, "end": 100.0}]},
        {"kind": "episode", "title": "Two", "start": 100.0, "end": 200.0,
         "sections": [{"kind": "content", "title": "c", "start": 100.0, "end": 200.0}]},
    ]}

    def test_one_output_per_segment_breaks_removed(self):
        out = pipeline.build_output_ranges(self.DATA, combined=False)
        self.assertEqual([o["output"] for o in out], ["01-one.mp4", "02-two.mp4"])
        self.assertEqual(out[0]["ranges"], [(0.0, 60.0), (80.0, 100.0)])
        self.assertEqual(out[1]["ranges"], [(100.0, 200.0)])

    def test_combined_output_merges_adjacent_ranges(self):
        out = pipeline.build_output_ranges(self.DATA, combined=True)
        self.assertEqual(out[-1]["output"], "full-nobreaks.mp4")
        self.assertEqual(out[-1]["ranges"], [(0.0, 60.0), (80.0, 200.0)])

    def test_keyframes_move_starts(self):
        kfs = [[0.0, -0.03], [58.0, 57.9], [79.0, 78.9], [100.0, 99.9]]
        out = pipeline.build_output_ranges(self.DATA, combined=False, keyframes=kfs)
        self.assertEqual(out[0]["ranges"], [(0.0, 60.0), (79.0, 100.0)])
        self.assertEqual(out[1]["ranges"], [(100.0, 200.0)])


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


def make_error_session_jsonl(err: str) -> str:
    """A synthetic pi session whose last assistant message failed with `err`
    (stopReason error + errorMessage, as pi records provider failures)."""
    entries = [
        {"type": "session", "version": 3, "id": "err-1", "timestamp": "2026-01-01T10:00:00.000Z", "cwd": "/w"},
        {"type": "model_change", "id": "e1", "parentId": None, "timestamp": "2026-01-01T10:00:01.000Z",
         "provider": "mistral", "modelId": "mistral-large-2512"},
        {"type": "message", "id": "e2", "parentId": "e1", "timestamp": "2026-01-01T10:00:02.000Z",
         "message": {"role": "user", "content": "segment this", "timestamp": 1}},
        {"type": "message", "id": "e3", "parentId": "e2", "timestamp": "2026-01-01T10:00:10.000Z",
         "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}],
                     "api": "mistral-conversations", "provider": "mistral", "model": "mistral-large-2512",
                     "usage": {"input": 100, "output": 5, "cacheRead": 0, "cacheWrite": 0,
                               "totalTokens": 105,
                               "cost": {"input": 2e-4, "output": 3e-5, "cacheRead": 0.0,
                                        "cacheWrite": 0.0, "total": 2.3e-4}},
                     "stopReason": "error", "errorMessage": err, "timestamp": 2}},
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

    def test_accumulate_usage_sums_scalars_and_skips_nested_objects(self):
        total = {}
        pipeline.accumulate_usage(total, {"prompt_tokens": 4, "completion_tokens": 6,
                                          "total_tokens": 10, "prompt_audio_seconds": 20,
                                          "prompt_tokens_details": {"cached_tokens": 0}})
        pipeline.accumulate_usage(total, {"prompt_tokens": 1, "completion_tokens": 2,
                                          "total_tokens": 3, "prompt_audio_seconds": 5,
                                          "prompt_tokens_details": {"cached_tokens": 0}})
        self.assertEqual(total, {"prompt_tokens": 5, "completion_tokens": 8,
                                 "total_tokens": 13, "prompt_audio_seconds": 25})
        # missing/None usage is a no-op (old cache files have no usage at all)
        pipeline.accumulate_usage(total, None)
        self.assertEqual(total["total_tokens"], 13)


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
        second = pipeline.Accounting(self.csv, "Day 2")
        second.record(step="transcribe", detail="chunk-000 provider=openai", model="whisper-1",
                      audio_seconds=10, cost_usd=0.001)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline.report_api_usage([self.acct, second])
            pipeline.report_api_usage([])  # no accts: no-op
        out = buf.getvalue()
        self.assertIn("API usage this run", out)
        self.assertIn("[Day 1] segment", out)
        self.assertIn("total API usage (ledger)", out)
        self.assertIn("TOTAL", out)
        self.assertIn(f"TOTAL {self.csv}: 2 req", out)


class ProviderConfigTests(unittest.TestCase):
    def test_defaults_and_provider_overrides(self):
        defaults = pipeline.resolve_options(pipeline.parse_args([]), {})
        self.assertEqual(defaults["transcribe_provider"], "mistral")
        self.assertEqual(defaults["segment_provider"], "mistral")
        self.assertEqual(defaults["transcribe_model"], "voxtral-small-transcribe-3")
        self.assertEqual(defaults["segment_model"], "mistral/mistral-large-latest")
        self.assertEqual(defaults["effort"], "medium")

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "event.toml"
            config.write_text("""
transcribe_provider = "mistral"
transcribe_model = "voxtral-custom"
segment_provider = "openai"

[providers.openai.segment]
model = "gpt-6.1-terra"
effort = "high"

[[days]]
path = "day-1.mp4"
transcribe_provider = "openai"

[[days]]
path = "day-2.mp4"
segment_provider = "mistral"
""")
            event, days = pipeline.load_config(config)

        cli = pipeline.parse_args([])
        first = pipeline.resolve_options(cli, event, days[0]["overrides"])
        self.assertEqual(first["transcribe_model"], "whisper-1")
        self.assertEqual(first["segment_model"], "openai/gpt-6.1-terra")
        self.assertEqual(first["effort"], "high")
        second = pipeline.resolve_options(cli, event, days[1]["overrides"])
        self.assertEqual(second["transcribe_model"], "voxtral-custom")
        self.assertEqual(second["segment_model"], "mistral/mistral-large-latest")
        self.assertEqual(second["effort"], "medium")

        cli = pipeline.parse_args(["--segment-provider", "openai", "--effort", "low"])
        second = pipeline.resolve_options(cli, event, days[1]["overrides"])
        self.assertEqual(second["segment_model"], "openai/gpt-6.1-terra")
        self.assertEqual(second["effort"], "low")

    def test_legacy_qualified_pi_model_still_selects_provider(self):
        cli = pipeline.parse_args(["--segment-model", "openai/gpt-6.1-terra"])
        options = pipeline.resolve_options(cli, {})
        self.assertEqual(options["segment_provider"], "openai")
        self.assertEqual(options["effort"], "medium")


class TranscriptionProviderTests(unittest.TestCase):
    def test_switch_to_openai_replaces_mistral_cache_and_resumes_without_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "raw.mp4").write_bytes(b"video")
            (work / "transcript.json").write_text(json.dumps(
                {"model": "voxtral-small-transcribe-3", "segments": []}))
            (work / "chunks").mkdir()
            (work / "chunks/chunk-000.json").write_text(json.dumps(
                {"model": "voxtral-small-transcribe-3", "segments": []}))
            (work / ".edit-done").touch()
            acct = pipeline.Accounting(work / "api-usage.csv", "Day 1")

            response = mock.Mock(status_code=200)
            response.json.return_value = {
                "duration": 10, "text": "Hello",
                "segments": [{"start": 1, "end": 3, "text": "Hello"}],
            }
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                    mock.patch.object(pipeline, "probe_duration", return_value=10), \
                    mock.patch.object(pipeline, "extract_audio_chunk",
                                      side_effect=lambda _raw, _start, _dur, out: out.touch()), \
                    mock.patch.object(pipeline.requests, "post", return_value=response) as post:
                segments = pipeline.step_transcribe(work, "whisper-1", "en", 55, None,
                                                     acct=acct, provider="openai")

            post.assert_called_once()
            self.assertEqual(post.call_args.args[0], f"{pipeline.OPENAI_API_BASE}/audio/transcriptions")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
            self.assertEqual(post.call_args.kwargs["data"], {
                "model": "whisper-1", "response_format": "verbose_json",
                "timestamp_granularities[]": "segment", "language": "en",
            })
            self.assertEqual(segments, [{"start": 1.0, "end": 3.0, "text": "Hello", "speaker_id": None}])
            self.assertIn("00:00:01,000 --> 00:00:03,000", (work / "raw.srt").read_text())
            self.assertFalse((work / ".edit-done").exists())
            self.assertEqual(acct.rows[0]["audio_seconds"], 10)
            self.assertAlmostEqual(acct.rows[0]["cost_usd"], 0.001)
            self.assertIn("provider=openai", acct.rows[0]["detail"])

            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(pipeline.step_transcribe(work, "whisper-1", "en", 55, None,
                                                          provider="openai"), segments)
            (work / "transcript.json").unlink()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                    mock.patch.object(pipeline, "probe_duration", return_value=10), \
                    mock.patch.object(pipeline.requests, "post") as post:
                self.assertEqual(pipeline.step_transcribe(work, "whisper-1", "en", 55, None,
                                                          provider="openai"), segments)
            post.assert_not_called()


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

    def test_probes_run_with_the_same_env_as_the_oneshot(self):
        env = {"PI_CODING_AGENT_DIR": "/x/pi-agent"}
        with mock.patch.object(pipeline, "pi_supports_effort", return_value=False) as m_eff, \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=True) as m_tools:
            pipeline.build_pi_oneshot_cmd("pi", "m", "medium", Path("/w/p.md"),
                                          Path("/w/s.jsonl"), env=env)
        m_eff.assert_called_once_with("pi", env)
        m_tools.assert_called_once_with("pi", env)


class PiBareEnvTests(unittest.TestCase):
    """pi must run with a bare config (PI_CODING_AGENT_DIR -> empty dir in
    agent-scratch), never the user's ~/.pi/agent -- also outside the sandbox."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.scratch = self.dir / "agent-scratch"
        self.scratch.mkdir()

    def test_bare_env_points_at_fresh_empty_dir_in_scratch(self):
        stale = self.scratch / pipeline.PI_AGENT_DIR_NAME
        stale.mkdir()
        (stale / "settings.json").write_text('{"packages": ["npm:evil"]}')  # must be wiped
        env = pipeline.pi_bare_env(self.scratch)
        agent_dir = Path(env["PI_CODING_AGENT_DIR"])
        self.assertEqual(agent_dir, (self.scratch / pipeline.PI_AGENT_DIR_NAME).resolve())
        self.assertTrue(agent_dir.is_dir())
        self.assertEqual(list(agent_dir.iterdir()), [])  # bare: no settings/extensions left
        # the pipeline's own process env is not mutated, the rest passes through
        self.assertNotIn("PI_CODING_AGENT_DIR", os.environ)
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_bare_env_dir_is_inside_the_bwrap_writable_bind(self):
        # bwrap_command makes only agent-scratch writable; the bare pi dir must
        # live there so pi can write its cache/auth inside the sandbox too
        which = lambda name: {"bwrap": "/usr/bin/bwrap", "pi": "/x/bin/pi"}.get(name)
        with mock.patch.object(pipeline.shutil, "which", side_effect=which):
            cmd = pipeline.bwrap_command(["pi", "-p", "hi"], self.dir)
        i = cmd.index("--bind")
        writable = Path(cmd[i + 1])
        env = pipeline.pi_bare_env(self.dir / "agent-scratch")
        self.assertEqual(Path(env["PI_CODING_AGENT_DIR"]).parent, writable)


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

    def test_segment_runs_pi_with_bare_agent_dir(self):
        captured = {}

        def fake_run(cmd, **kw):
            captured.update(kw)
            return self._fake_pi_run(cmd, **kw)

        with mock.patch.object(pipeline, "probe_duration", return_value=100.0), \
                mock.patch.object(pipeline, "pi_supports_effort", return_value=False), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=True), \
                mock.patch.object(pipeline.subprocess, "run", side_effect=fake_run):
            pipeline.step_segment(self.workdir, "m", "medium", "pi", None,
                                  day_title="Day 1", sandbox="none", acct=self.acct)
        env = captured.get("env") or {}
        agent_dir = Path(env["PI_CODING_AGENT_DIR"])
        self.assertEqual(agent_dir.parent, (self.workdir / "agent-scratch").resolve())
        self.assertEqual(agent_dir.name, pipeline.PI_AGENT_DIR_NAME)
        self.assertTrue(agent_dir.is_dir())
        self.assertEqual(list(agent_dir.iterdir()), [])

    def test_save_pi_transcript_without_session_file(self):
        self.assertIsNone(pipeline.save_pi_transcript(self.workdir, self.dir / "nope.jsonl",
                                                      self.acct, "m"))
        self.assertEqual(self.acct.rows, [])

    def test_terra_cache_tracks_effort_and_transcript(self):
        (self.workdir / "segments.json").write_text('{"segments": []}')
        (self.workdir / ".edit-done").touch()

        def run_segment(effort):
            return pipeline.step_segment(self.workdir, "openai/gpt-6.1-terra", effort,
                                         "pi", None, sandbox="none", provider="openai")

        with mock.patch.object(pipeline, "probe_duration", return_value=100.0), \
                mock.patch.object(pipeline, "pi_supports_effort", return_value=False), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=True), \
                mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/pi"), \
                mock.patch.object(pipeline.subprocess, "run", side_effect=self._fake_pi_run) as pi:
            first = run_segment("medium")
            self.assertEqual(run_segment("medium"), first)
            self.assertEqual(pi.call_count, 1)
            cmd = pi.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--model") + 1], "openai/gpt-6.1-terra")
            self.assertEqual(cmd[cmd.index("--thinking") + 1], "medium")
            self.assertFalse((self.workdir / ".edit-done").exists())

            run_segment("high")
            self.assertEqual(pi.call_count, 2)
            (self.workdir / "transcript.json").write_text(json.dumps(
                {"segments": [{"start": 0.0, "end": 10.0, "text": "different transcript"}]}))
            run_segment("high")
            self.assertEqual(pi.call_count, 3)


class ClassifyPiFailureTests(unittest.TestCase):
    """classify_pi_failure: money/auth/bad-request are fatal, transient and random
    failures (incl. the 'terminated' of the incident) are retryable."""

    def test_out_of_money_is_fatal(self):
        for txt in ("HTTP 402: payment required",
                    "insufficient credits for this workspace",
                    "your credit balance is too low, top up your account"):
            kind, cause = pipeline.classify_pi_failure(1, f"stderr tail: {txt}")
            self.assertEqual(kind, "fatal", txt)
            self.assertIn("money", cause)

    def test_auth_is_fatal(self):
        kind, cause = pipeline.classify_pi_failure(1, "401 Unauthorized: invalid api key")
        self.assertEqual(kind, "fatal")
        self.assertIn("auth", cause)

    def test_prompt_too_large_is_fatal(self):
        kind, _ = pipeline.classify_pi_failure(1, "400 bad request: context length exceeded")
        self.assertEqual(kind, "fatal")

    def test_server_error_is_retryable(self):
        kind, cause = pipeline.classify_pi_failure(
            1, "pi session error(s): Provider stopped with: error (server error)")
        self.assertEqual(kind, "retryable")
        self.assertIn("5xx", cause)

    def test_server_error_names_the_status_code_and_message(self):
        # pi records the provider error verbatim: the cause carries the real code
        # and message, not just "5xx"
        kind, cause = pipeline.classify_pi_failure(
            1, 'pi session error(s): Mistral API error (503): {"object":"error",'
               '"message":"Service unavailable","type":"server_error"}')
        self.assertEqual(kind, "retryable")
        self.assertEqual(cause, "API server error (5xx): 503 Service unavailable")

    def test_server_error_falls_back_to_the_reason_phrase(self):
        kind, cause = pipeline.classify_pi_failure(
            1, "stderr tail: Error: HTTP 502 Bad Gateway")
        self.assertEqual(kind, "retryable")
        self.assertEqual(cause, "API server error (5xx): 502 Bad Gateway")

    def test_terminated_is_retryable(self):
        # the incident: pi exited rc=1 with only "terminated" on stderr
        kind, cause = pipeline.classify_pi_failure(1, "stderr tail: terminated")
        self.assertEqual(kind, "retryable")
        self.assertIn("terminated", cause)

    def test_429_is_retryable(self):
        kind, _ = pipeline.classify_pi_failure(1, "HTTP 429: rate limit exceeded")
        self.assertEqual(kind, "retryable")

    def test_network_error_is_retryable(self):
        kind, cause = pipeline.classify_pi_failure(1, "fetch failed: ECONNRESET")
        self.assertEqual(kind, "retryable")
        self.assertIn("network", cause)

    def test_killed_by_signal_is_retryable(self):
        kind, cause = pipeline.classify_pi_failure(-15, "")
        self.assertEqual(kind, "retryable")
        self.assertIn("signal 15", cause)

    def test_unknown_is_retryable_random_failure(self):
        self.assertEqual(pipeline.classify_pi_failure(1, "weird stuff"),
                         ("retryable", "unknown/random failure"))


class StepSegmentRetryTests(unittest.TestCase):
    """step_segment retries a failing pi oneshot up to PI_MAX_ATTEMPTS total attempts,
    records the spend of every attempt, and dies immediately (no retry) on fatal
    causes like an out-of-money API."""

    GOOD_OUT = ('```json\n{"segments": [{"kind": "episode", "title": "ep", "start": 0, '
                '"end": 100, "sections": [{"kind": "content", "title": "c", '
                '"start": 0, "end": 100}]}]}\n```\n')

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.workdir = self.dir / "work"
        self.workdir.mkdir()
        (self.workdir / "raw.info.json").write_text(json.dumps(
            {"id": "x", "title": "t", "uploader": "u", "description": "", "duration": 100}))
        (self.workdir / "transcript.json").write_text(json.dumps(
            {"model": "m", "segments": [{"start": 0.0, "end": 10.0, "text": "hi"}]}))
        self.acct = pipeline.Accounting(self.dir / "api-usage.csv", day="Day 1")

    def _fake_pi(self, failures: int, err: str, rc: int = 1):
        """A fake subprocess.run: the first `failures` calls fail (rc, stderr=err,
        session with errorMessage=err), later calls succeed like the other tests."""
        state = {"n": 0}

        def fake_run(cmd, **kw):
            state["n"] += 1
            i = cmd.index("--session")
            if state["n"] <= failures:
                Path(cmd[i + 1]).write_text(make_error_session_jsonl(err))
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)
            Path(cmd[i + 1]).write_text(make_session_jsonl())
            return subprocess.CompletedProcess(cmd, 0, stdout=self.GOOD_OUT, stderr="")
        return fake_run, state

    def _run_segment(self, fake_run):
        with mock.patch.object(pipeline, "probe_duration", return_value=100.0), \
                mock.patch.object(pipeline, "pi_supports_effort", return_value=False), \
                mock.patch.object(pipeline, "pi_supports_tools", return_value=False), \
                mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/pi"), \
                mock.patch.object(pipeline.time, "sleep") as slp, \
                mock.patch.object(pipeline.subprocess, "run", side_effect=fake_run):
            data = pipeline.step_segment(self.workdir, "m", "medium", "pi", None,
                                         day_title="Day 1", sandbox="none", acct=self.acct)
        return data, slp

    def test_transient_failures_retried_then_succeed(self):
        fake, state = self._fake_pi(2, "Provider stopped with: error (server error)")
        data, slp = self._run_segment(fake)
        self.assertEqual(state["n"], 3)          # 2 failures + 1 success = 3 attempts
        self.assertEqual(slp.call_count, 2)      # backoff slept between attempts
        self.assertEqual(len(data["segments"]), 1)
        self.assertTrue((self.workdir / "segments.json").exists())
        rows = pipeline.ledger_rows(self.dir / "api-usage.csv")
        self.assertEqual(len(rows), 3)           # spend of every attempt recorded
        self.assertTrue(all(r["step"] == "segment" for r in rows))

    def test_fatal_out_of_money_dies_immediately_without_retry(self):
        fake, state = self._fake_pi(99, "HTTP 402: payment required -- insufficient credits")
        with self.assertRaises(SystemExit) as cm:
            self._run_segment(fake)
        self.assertEqual(state["n"], 1)          # fatal: no retry
        msg = str(cm.exception)
        self.assertIn("out of money", msg)
        self.assertIn("NOT retried", msg)

    def test_gives_up_after_max_attempts_with_informative_error(self):
        fake, state = self._fake_pi(99, "terminated")
        with self.assertRaises(SystemExit) as cm:
            self._run_segment(fake)
        self.assertEqual(state["n"], pipeline.PI_MAX_ATTEMPTS)
        msg = str(cm.exception)
        self.assertIn(f"after {pipeline.PI_MAX_ATTEMPTS} attempts", msg)
        self.assertIn("terminated", msg)         # diagnostics from session + stderr
        self.assertIn("pi session:", msg)
        # spend of the failed attempts was still recorded
        self.assertEqual(len(pipeline.ledger_rows(self.dir / "api-usage.csv")),
                         pipeline.PI_MAX_ATTEMPTS)


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

    def _which(self, name):
        return {"bwrap": "/usr/bin/bwrap", "pi": str(self.dir / "bin" / "pi")}.get(name)

    def test_bwrap_command_mounts_only_system_dirs_pi_and_workdir_inputs(self):
        # realistic workdir: the input files + a previous run's outputs; plus a
        # sibling day workdir and the workdir-root ledger of a --config event
        work = self.dir / "event" / "day1"
        work.mkdir(parents=True)
        for name in pipeline.SEGMENT_SANDBOX_INPUTS:
            (work / name).write_text("input")
        for name in ("segments.json", "pi-session.jsonl", "pi-transcript.json",
                     "segment-raw-output.txt", "api-usage.csv"):
            (work / name).write_text("SECRET")
        (self.dir / "event" / "day2").mkdir()
        (self.dir / "event" / "day2" / "transcript.json").write_text("DAY2")
        (self.dir / "event" / "api-usage.csv").write_text("LEDGER")
        (self.dir / "bin").mkdir()
        (self.dir / "bin" / "pi").write_text("#!/bin/sh\n")
        with mock.patch.object(pipeline.shutil, "which", side_effect=self._which):
            cmd = pipeline.bwrap_command(["pi", "-p", "hi"], work)
        self.assertEqual(cmd[0], "/usr/bin/bwrap")
        ro = [(cmd[i + 1], cmd[i + 2]) for i, a in enumerate(cmd) if a == "--ro-bind"]
        ro_srcs = [s for s, _ in ro]
        # whitelist, not a whole-filesystem bind
        self.assertNotIn("/", ro_srcs)
        for d in ("/usr", "/etc", "/run"):
            self.assertIn(d, ro_srcs)
        # pi itself is bound (installed outside the system dirs here)
        self.assertIn(str(self.dir / "bin" / "pi"), ro_srcs)
        # every existing input file is bound read-only at its ORIGINAL path
        for name in pipeline.SEGMENT_SANDBOX_INPUTS:
            self.assertIn((str(work / name), str(work / name)), ro)
        # previous-run outputs, the sibling day and the root ledger are NOT mounted
        # (keyframes.json/silences.json ARE mounted -- they are in
        # SEGMENT_SANDBOX_INPUTS, asserted by the loop above)
        for secret in ("segments.json", "pi-session.jsonl", "pi-transcript.json",
                       "segment-raw-output.txt", "api-usage.csv"):
            self.assertNotIn(str(work / secret), ro_srcs)
        self.assertNotIn(str(self.dir / "event" / "day2"), ro_srcs)
        self.assertNotIn(str(self.dir / "event" / "api-usage.csv"), ro_srcs)
        self.assertNotIn(str(self.dir / "event"), ro_srcs)
        # the workdir itself is only an empty --dir chain; the payload's cwd is it
        dirs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--dir"]
        self.assertIn(str(work.resolve()), dirs)
        self.assertEqual(cmd[cmd.index("--chdir") + 1], str(work.resolve()))
        # agent-scratch is the only writable bind
        binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
        self.assertEqual(binds, [str((work / "agent-scratch").resolve())])
        self.assertTrue((work / "agent-scratch").is_dir())
        # ephemeral /tmp + HOME redirect; the payload command is intact
        self.assertIn("--tmpfs", cmd)
        self.assertIn("/tmp/pi-home", cmd)
        self.assertEqual(cmd[cmd.index("--") + 1:], ["pi", "-p", "hi"])

    def test_bwrap_command_skips_input_files_that_do_not_exist(self):
        work = self.dir / "w"
        work.mkdir()
        (work / "raw.mp4").write_text("x")
        with mock.patch.object(pipeline.shutil, "which", side_effect=self._which):
            cmd = pipeline.bwrap_command(["pi"], work)
        ro_srcs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--ro-bind"]
        self.assertIn(str(work / "raw.mp4"), ro_srcs)
        self.assertNotIn(str(work / "keyframes.json"), ro_srcs)
        self.assertNotIn(str(work / "silences.json"), ro_srcs)
        self.assertNotIn(str(work / "transcript.json"), ro_srcs)
        self.assertNotIn(str(work / "segment-prompt.md"), ro_srcs)

    def test_sandbox_pi_binds_cover_npm_global_install(self):
        # ~/bin/pi -> ~/lib/node_modules/@scope/pi-pkg/bin/cli.js: the entry must be
        # recreated as the SAME symlink (wrappers locate their install via
        # `readlink -f $0`), the real script and the node_modules tree bound ro
        home = self.dir / "home" / "u"
        cli = home / "lib" / "node_modules" / "@scope" / "pi-pkg" / "bin" / "cli.js"
        cli.parent.mkdir(parents=True)
        cli.write_text("// pi")
        (home / "bin").mkdir()
        (home / "bin" / "pi").symlink_to(cli)
        with mock.patch.object(pipeline.shutil, "which",
                               return_value=str(home / "bin" / "pi")):
            binds = {(op, s): d for op, s, d in pipeline._sandbox_pi_binds("pi")}
        entry = str(home / "bin" / "pi")
        self.assertEqual(binds[("symlink", str(cli))], entry)
        self.assertEqual(binds[("ro-bind", str(cli))], str(cli))
        self.assertEqual(binds[("ro-bind", str(home / "lib" / "node_modules"))],
                         str(home / "lib" / "node_modules"))

    def test_sandbox_pi_binds_managed_install_tree(self):
        # ~/.pi/agent/bin/pi with the versioned install at ~/.pi/agent/install/...:
        # the install marker dir must be bound, but NOT the whole ~/.pi
        home = self.dir / "home" / "u"
        wrapper = home / ".pi" / "agent" / "bin" / "pi"
        (home / ".pi" / "agent" / "install" / "releases" / "1.0.0").mkdir(parents=True)
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("#!/bin/sh\n")
        with mock.patch.object(pipeline.shutil, "which", return_value=str(wrapper)):
            binds = {(op, s): d for op, s, d in pipeline._sandbox_pi_binds("pi")}
        self.assertEqual(binds[("ro-bind", str(wrapper))], str(wrapper))
        self.assertEqual(binds[("ro-bind", str(home / ".pi" / "agent" / "bin"))],
                         str(home / ".pi" / "agent" / "bin"))
        self.assertEqual(binds[("ro-bind", str(home / ".pi" / "agent" / "install"))],
                         str(home / ".pi" / "agent" / "install"))
        self.assertNotIn(("ro-bind", str(home / ".pi")), binds)

    def test_sandbox_pi_binds_under_system_dirs_are_dropped(self):
        # a pi installed in /usr needs no extra mounts (and bwrap cannot mount over
        # the /usr/bin/sh-style symlinks there anyway)
        with mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/pi"):
            self.assertEqual(pipeline._sandbox_pi_binds("pi"), [])

class SkipStageTests(unittest.TestCase):
    """--skip-* must not demand the artefacts of the stages they skip."""

    def _run(self, work: Path, *flags):
        opt = pipeline.resolve_options(pipeline.parse_args(list(flags)), {})
        opt["workdir_root"] = str(work)
        with mock.patch.object(pipeline, "ensure_keyframe_cache"), \
                mock.patch.object(pipeline, "step_edit") as edit:
            pipeline.run_pipeline(opt, {"url": "https://youtu.be/aaaaaaaaaaa",
                                        "workdir": str(work)}, day_title=None)
        return edit

    def test_skip_edit_does_not_require_segments_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            edit = self._run(work, "--skip-download", "--skip-transcribe",
                             "--skip-segment", "--skip-edit")
            edit.assert_not_called()

    def test_missing_segments_json_is_a_clear_error_when_editing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with self.assertRaises(SystemExit) as ctx:
                self._run(work, "--skip-download", "--skip-transcribe", "--skip-segment")
            self.assertIn("segments.json", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
