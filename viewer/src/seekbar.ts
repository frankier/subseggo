import type { Section } from './types';
import { el } from './util';

// Custom seek bar for the control-less embed: one colored block per section
// (colors shared with the segment-overview badges via the .k-* classes),
// a hover tooltip with the section's kind and title, click-to-seek, and a
// playhead.
export class SeekBar {
  private track: HTMLElement;
  private blocksEl: HTMLElement;
  private playhead: HTMLElement;
  private tooltip: HTMLElement;
  private sections: Section[] = [];
  private duration = 0;
  private onSeek: (t: number) => void;

  constructor(onSeek: (t: number) => void) {
    this.onSeek = onSeek;
    this.track = document.getElementById('seekbar')!;
    this.tooltip = document.getElementById('seek-tooltip')!;
    this.blocksEl = el('div', 'seek-blocks');
    this.playhead = el('div', 'seek-playhead');
    this.track.append(this.blocksEl, this.playhead);
    this.track.addEventListener('click', (ev) => {
      if (this.duration <= 0) return;
      const rect = this.track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
      this.onSeek(frac * this.duration);
    });
  }

  render(sections: Section[], duration: number): void {
    this.sections = [...sections].sort((a, b) => a.start - b.start);
    this.duration = duration;
    this.blocksEl.replaceChildren();
    for (const sec of this.sections) {
      const block = el('div', `seek-block k-${sec.kind}`);
      const left = this.duration > 0 ? (sec.start / this.duration) * 100 : 0;
      const width = this.duration > 0 ? Math.max(0.25, ((sec.end - sec.start) / this.duration) * 100) : 0;
      block.style.left = `${left}%`;
      block.style.width = `${width}%`;
      block.addEventListener('mouseenter', () => {
        this.tooltip.textContent = `${sec.kind}: ${sec.title}`;
        this.tooltip.hidden = false;
        const br = block.getBoundingClientRect();
        const tr = this.track.getBoundingClientRect();
        const center = br.left - tr.left + br.width / 2;
        this.tooltip.style.left = `${Math.min(Math.max(center, 60), tr.width - 60)}px`;
      });
      block.addEventListener('mouseleave', () => {
        this.tooltip.hidden = true;
      });
      this.blocksEl.append(block);
    }
    this.update(0);
  }

  setDuration(duration: number): void {
    if (duration > 0 && duration !== this.duration) this.render(this.sections, duration);
  }

  update(t: number): void {
    const frac = this.duration > 0 ? Math.min(1, Math.max(0, t / this.duration)) : 0;
    this.playhead.style.left = `${frac * 100}%`;
  }
}
