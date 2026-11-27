import '../public/styles.css';
import { SegmentsView, type DayData } from './segments';
import { SeekBar } from './seekbar';
import { SubtitleView } from './subtitles';
import { configDays, type Cue, type Segment, type SegmentsDoc, type TranscriptDoc, type ViewerConfig } from './types';
import { activeIndex, fetchJson, fmtTime, joinUrl } from './util';
import { createPlayer, PLAYER_STATE, type PlayerHandle } from './youtube';

function showError(message: string): void {
  const box = document.getElementById('error')!;
  box.hidden = false;
  box.textContent = message;
}

// --- theme: follow prefers-color-scheme, with a subtle manual override persisted in localStorage
function initTheme(): void {
  const toggle = document.getElementById('theme-toggle') as HTMLButtonElement;
  const mq = window.matchMedia('(prefers-color-scheme: light)');
  const apply = (): void => {
    const stored = localStorage.getItem('theme');
    const theme = stored ?? (mq.matches ? 'light' : 'dark');
    document.documentElement.dataset.theme = theme;
    toggle.textContent = theme === 'light' ? 'Light' : 'Dark';
  };
  toggle.addEventListener('click', () => {
    const current = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
    localStorage.setItem('theme', current === 'light' ? 'dark' : 'light');
    apply();
  });
  mq.addEventListener?.('change', apply);
  apply();
}

const isNum = (v: unknown): v is number => typeof v === 'number';

function normalizeCues(raw: unknown): Cue[] {
  if (!Array.isArray(raw)) throw new Error('transcript.json: "segments" is not an array');
  return (raw as Record<string, unknown>[])
    .filter((s): s is Record<string, unknown> & { start: number; end: number } =>
      !!s && isNum(s.start) && isNum(s.end) && s.end > s.start && typeof s.text === 'string')
    .map((s) => ({ start: s.start, end: s.end, text: s.text as string, speaker_id: (s.speaker_id as string | null) ?? null }))
    .sort((a, b) => a.start - b.start);
}

function normalizeSegments(raw: unknown): Segment[] {
  if (!Array.isArray(raw)) throw new Error('segments.json: "segments" is not an array');
  return (raw as Record<string, unknown>[])
    .filter((s): s is Record<string, unknown> & { start: number; end: number } => !!s && isNum(s.start) && isNum(s.end))
    .map((s) => ({
      kind: String(s.kind ?? 'content'),
      title: String(s.title ?? ''),
      start: s.start,
      end: s.end,
      sections: (Array.isArray(s.sections) ? s.sections : [])
        .filter((x): x is Record<string, unknown> & { start: number; end: number } =>
          !!x && isNum((x as Record<string, unknown>).start) && isNum((x as Record<string, unknown>).end))
        .map((x) => ({
          kind: String(x.kind ?? 'content'),
          title: String(x.title ?? ''),
          start: x.start,
          end: x.end,
        }))
        .sort((a, b) => a.start - b.start),
    }))
    .sort((a, b) => a.start - b.start);
}

async function main(): Promise<void> {
  initTheme();
  const qs = new URLSearchParams(location.search);

  let config: ViewerConfig = {};
  try {
    config = (await fetchJson(qs.get('config') ?? 'config.json')) as ViewerConfig;
  } catch {
    // config.json is optional; ?day=/?v= alone won't work without it, but the error below says so
  }
  const days = configDays(config);
  if (days.length === 0) {
    showError('config.json has neither a "days" list nor a "videoId"');
    return;
  }

  // initial day: ?day=<index|title>, default the first
  let dayIdx = 0;
  const dq = qs.get('day');
  if (dq !== null) {
    const n = Number(dq);
    if (Number.isInteger(n) && days[n]) dayIdx = n;
    else {
      const byTitle = days.findIndex((d) => d.title === dq);
      if (byTitle >= 0) dayIdx = byTitle;
    }
  }

  // load every day's data up front so the hierarchy can render all days
  const dayData: (DayData & { cues: Cue[] | null })[] = await Promise.all(days.map(async (day) => {
    const base = { day };
    try {
      const transcriptUrl = qs.get('transcript') ?? day.transcriptUrl ?? joinUrl(day.dir, 'transcript.json');
      const segmentsUrl = qs.get('segments') ?? day.segmentsUrl ?? joinUrl(day.dir, 'segments.json');
      const [tr, sg] = await Promise.all([
        fetchJson(transcriptUrl) as Promise<TranscriptDoc>,
        fetchJson(segmentsUrl) as Promise<SegmentsDoc>,
      ]);
      return { ...base, cues: normalizeCues(tr?.segments), segments: normalizeSegments(sg?.segments) };
    } catch (err) {
      return { ...base, cues: null, segments: null, error: err instanceof Error ? err.message : String(err) };
    }
  }));

  const clockEl = document.getElementById('clock')!;
  const playBtn = document.getElementById('play-pause') as HTMLButtonElement;

  let player: PlayerHandle;
  let activeDay = dayIdx;
  let activeCues: Cue[] = [];
  let activeSegments: Segment[] = [];
  let subtitleView: SubtitleView;
  let segmentsView: SegmentsView;
  let seekBar: SeekBar;

  const seek = (t: number): void => player.seekTo(Math.max(0, t));

  const renderDay = (i: number): void => {
    activeDay = i;
    activeCues = dayData[i].cues ?? [];
    activeSegments = dayData[i].segments ?? [];
    subtitleView = new SubtitleView(activeCues, seek);
    segmentsView = new SegmentsView(dayData, {
      onSelectDay: (idx, time) => selectDay(idx, time),
      onSeek: seek,
    });
    segmentsView.setActiveDay(i);
    const sections = activeSegments.flatMap((s) => s.sections);
    const end = activeSegments.length > 0 ? Math.max(...activeSegments.map((s) => s.end)) : 0;
    seekBar.render(sections, end);
    if (dayData[i].error) showError(`day ${i + 1} (${dayData[i].day.title ?? ''}): ${dayData[i].error}`);
  };

  const selectDay = (i: number, time?: number): void => {
    if (i === activeDay && time === undefined) return;
    const videoId = i === dayIdx && qs.get('v') ? qs.get('v')! : days[i].videoId;
    renderDay(i);
    player.loadVideo(videoId, time ?? 0);
  };

  const initialVideoId = qs.get('v') ?? days[dayIdx].videoId;
  const startAt = Number(qs.get('t')) || 0;

  // function declaration: hoisted, safe to reference from the player callbacks below
  function updatePlayButton(state: number): void {
    playBtn.textContent = state === PLAYER_STATE.PLAYING ? 'Pause' : 'Play';
  }

  player = await createPlayer(initialVideoId, document.getElementById('player')!, {
    onReady: (p) => {
      if (startAt > 0) p.seekTo(startAt);
      updatePlayButton(p.getState());
    },
    onStateChange: (state) => updatePlayButton(state),
  });
  playBtn.addEventListener('click', () => {
    if (player.getState() === PLAYER_STATE.PLAYING) player.pause();
    else player.play();
  });

  seekBar = new SeekBar(seek);
  renderDay(dayIdx);

  const tick = (): void => {
    const t = player.getCurrentTime();
    const dur = player.getDuration();
    if (dur > 0) seekBar.setDuration(dur);
    const segIdx = activeIndex(activeSegments, t);
    const episodeStart = segIdx >= 0 ? activeSegments[segIdx].start : 0;
    clockEl.textContent = fmtTime(Math.max(0, t - episodeStart));
    seekBar.update(t);
    subtitleView.update(t, episodeStart);
    segmentsView.update(t);
  };
  tick();
  setInterval(tick, 200);
}

main().catch((err) => showError(String(err)));
