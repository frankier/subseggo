import type { Cue } from './types';
import { activeIndex, el, fmtTime, stepTime } from './util';

// Subtitle panel: scrollable cue list synced to playback, a prominent current
// cue, prev/next buttons that jump the video between cues, and a follow toggle
// that keeps the active cue scrolled into view. Cue timestamps are relative to
// the current episode start and only shown while the user interacts with the
// list (see the #cue-list:hover rules in styles.css).
export class SubtitleView {
  private cues: Cue[];
  private rows: { li: HTMLLIElement; time: HTMLSpanElement }[] = [];
  private activeIdx = -1;
  private follow = true;
  private lastTime = 0;
  private lastEpisodeStart = Number.NaN;

  private listEl: HTMLElement;
  private currentEl: HTMLElement;
  private posEl: HTMLElement;
  private prevBtn: HTMLButtonElement;
  private nextBtn: HTMLButtonElement;
  private followBtn: HTMLButtonElement;
  private onSeek: (t: number) => void;

  constructor(cues: Cue[], onSeek: (t: number) => void) {
    this.cues = [...cues].sort((a, b) => a.start - b.start);
    this.onSeek = onSeek;

    this.listEl = document.getElementById('cue-list')!;
    this.listEl.replaceChildren();
    this.currentEl = document.getElementById('current-cue')!;
    this.posEl = document.getElementById('cue-pos')!;
    this.prevBtn = document.getElementById('prev-cue') as HTMLButtonElement;
    this.nextBtn = document.getElementById('next-cue') as HTMLButtonElement;
    this.followBtn = document.getElementById('follow-toggle') as HTMLButtonElement;

    for (const cue of this.cues) {
      const li = el('li');
      li.tabIndex = 0;
      const time = el('span', 'cue-time', '');
      li.append(time);
      if (cue.speaker_id) li.append(el('span', 'speaker', cue.speaker_id));
      li.append(el('span', 'cue-text', cue.text.trim()));
      li.addEventListener('click', () => this.onSeek(cue.start));
      this.listEl.append(li);
      this.rows.push({ li, time });
    }

    this.prevBtn.addEventListener('click', () => {
      const t = stepTime(this.cues, this.lastTime, -1);
      if (t !== null) this.onSeek(t);
    });
    this.nextBtn.addEventListener('click', () => {
      const t = stepTime(this.cues, this.lastTime, 1);
      if (t !== null) this.onSeek(t);
    });
    this.followBtn.addEventListener('click', () => this.setFollow(!this.follow));
    this.setFollow(true);
  }

  setFollow(on: boolean): void {
    this.follow = on;
    this.followBtn.setAttribute('aria-pressed', String(on));
    this.followBtn.classList.toggle('on', on);
  }

  private relabel(episodeStart: number): void {
    for (let i = 0; i < this.rows.length; i++) {
      this.rows[i].time.textContent = fmtTime(Math.max(0, this.cues[i].start - episodeStart));
    }
  }

  update(t: number, episodeStart: number): void {
    this.lastTime = t;
    if (episodeStart !== this.lastEpisodeStart) {
      this.lastEpisodeStart = episodeStart;
      this.relabel(episodeStart);
    }
    // the last cue starting at or before t stays "current" even in a gap after its end
    const current = activeIndex(this.cues, t);
    if (current !== this.activeIdx) {
      if (this.activeIdx >= 0) this.rows[this.activeIdx].li.classList.remove('active');
      this.activeIdx = current;
      if (this.activeIdx >= 0) {
        this.rows[this.activeIdx].li.classList.add('active');
        if (this.follow) this.rows[this.activeIdx].li.scrollIntoView({ block: 'nearest' });
      }
    }
    const cue = this.activeIdx >= 0 ? this.cues[this.activeIdx] : null;
    this.currentEl.replaceChildren();
    if (cue) {
      this.currentEl.append(el('span', 'cue-time', fmtTime(Math.max(0, cue.start - episodeStart))));
      this.currentEl.append(document.createTextNode(cue.text.trim()));
    }
    this.posEl.textContent = cue ? `cue ${this.activeIdx + 1}/${this.cues.length}` : 'no cue';
    this.prevBtn.disabled = stepTime(this.cues, t, -1) === null;
    this.nextBtn.disabled = stepTime(this.cues, t, 1) === null;
  }
}
