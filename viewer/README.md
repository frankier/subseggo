# yt-segment viewer

Client-side-only viewer for the **unsegmented** YouTube video: the original video
plays in a control-less YouTube embed while `transcript.json` drives a synced
subtitle panel and `segments.json` drives a foldable day > segment > section
tree plus a colored seek bar. No backend; the built `dist/` is a plain static site.

## Usage

```sh
npm install
npm run watch    # rebuild on change + serve dist/ at http://localhost:8080/
npm run build    # one dev build into dist/
npm run bundle   # production bundle into dist/ (minified, hashed assets)
npm run serve    # serve an existing dist/ (PORT env, default 8080)
```

Deploy: upload `dist/` to any static host. Serve over http (not `file://`) — the
page fetches its JSON at runtime.

## Config

`config.json` (path overridable with `?config=`) comes in two shapes:

```json
{
  "days": [
    {"title": "Day 1", "videoId": "VIDEO_ID", "dir": "days/day-1"},
    {"title": "Day 2", "videoId": "VIDEO_ID_2", "dir": "days/day-2"}
  ]
}
```

or the single-video shortcut (one implicit day rooted at the site root):

```json
{"videoId": "VIDEO_ID", "title": "optional day label"}
```

Each day's `dir` (relative to the site root, default `.`) must contain that day's
`transcript.json` (`{"segments": [{start, end, text, speaker_id}]}`) and
`segments.json` (`{"segments": [{kind, title, start, end, sections: [...]}]}`).
Per-day `transcriptUrl`/`segmentsUrl` can override the file locations.

The days become the top level of the segments hierarchy: clicking a day loads its
video and data; clicking a segment/section seeks (switching day first if needed).

URL query overrides: `?day=<index|title>` (initial day), `?v=VIDEO_ID`,
`?t=SECONDS` (initial seek), `?transcript=URL`, `?segments=URL`, `?config=URL`.

`npm run bundle` copies root-level `transcript.json` / `segments.json` /
`config.json` from `$DATA_DIR` over the samples when set (for the single-video
config shape), e.g. `DATA_DIR=../work/my-video npm run bundle`.

## UI behavior

- The embed has no YouTube controls; transport is the viewer's own play/pause
  button, a custom seek bar, and a clock showing time relative to the start of
  the current segment ("episode").
- Seek bar: one colored block per section, colors shared with the segment
  overview badges, hover tooltip with `kind: title`, click to seek.
- Transcript: prev/next cue buttons, cue timestamps relative to the episode start
  (only visible while hovering/focusing the list), and a Follow toggle (default
  on) that keeps the active cue scrolled into view.
- Segments tree: day > segment > section, durations in minutes (`<1 min` under a
  minute), foldable; refolds as playback moves so only the path to the current
  segment is expanded — that path cannot be folded.
- Theme: follows `prefers-color-scheme`; the subtle header toggle overrides it
  and persists in `localStorage`.

## Layout

```
src/main.ts       boot: theme, config + per-day data loading, player wiring, sync loop
src/youtube.ts    YouTube IFrame API loader + control-less player wrapper
src/seekbar.ts    custom seek bar (section blocks, tooltips, playhead)
src/subtitles.ts  cue list, current cue, follow mode, prev/next cue jumping
src/segments.ts   day > segment > section tree with folding + jumping
src/util.ts       time/duration formatting, url joining, stepping helpers
scripts/build.mjs dev build / watch+serve / production bundle into dist/
scripts/serve.mjs static server for dist/
public/           index.html, styles.css, config.json, sample day under days/day-1/
```
