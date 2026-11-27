import type { DayConfig, Segment } from './types';
import { activeIndex, el, fmtDurationMin, stepTime } from './util';

export interface DayData {
  day: DayConfig;
  segments: Segment[] | null; // null when the day's segments.json failed to load
  error?: string;
}

interface FoldNode {
  root: HTMLDivElement;
  children: HTMLDivElement;
  chevron: HTMLButtonElement;
  expanded: boolean;
  onPath: boolean;
}

// Segments panel: the config-driven hierarchy day > segment > section.
// Day and segment nodes are foldable; whenever the playback position moves to
// a different segment everything is refolded so only the path to the current
// segment (its day + the segment itself) stays expanded, and the fold buttons
// on that path are disabled so it cannot be folded. Clicking a day loads that
// day's video; clicking a segment/section seeks (switching day first when the
// row belongs to another day). Rows show durations in minutes, not times.
export class SegmentsView {
  private treeEl: HTMLElement;
  private prevBtn: HTMLButtonElement;
  private nextBtn: HTMLButtonElement;
  private days: DayData[];
  private dayNodes: FoldNode[] = [];
  private segNodes: { node: FoldNode; dayIdx: number; segment: Segment }[] = [];
  private activeDay = 0;
  private activeSeg = -1;
  private lastTime = 0;
  private onSelectDay: (dayIdx: number, time?: number) => void;
  private onSeek: (t: number) => void;

  constructor(days: DayData[], handlers: { onSelectDay: (dayIdx: number, time?: number) => void; onSeek: (t: number) => void }) {
    this.days = days;
    this.onSelectDay = handlers.onSelectDay;
    this.onSeek = handlers.onSeek;
    this.treeEl = document.getElementById('segment-tree')!;
    this.treeEl.replaceChildren();
    this.prevBtn = document.getElementById('prev-segment') as HTMLButtonElement;
    this.nextBtn = document.getElementById('next-segment') as HTMLButtonElement;

    days.forEach((entry, dayIdx) => {
      const node = this.makeFoldNode('day');
      const head = el('div', 'node-head day-head');
      head.append(node.chevron);
      head.append(el('span', 'seg-title', entry.day.title ?? `Day ${dayIdx + 1}`));
      const segs = entry.segments ?? [];
      if (segs.length > 0) {
        const end = Math.max(...segs.map((s) => s.end));
        head.append(el('span', 'dur', fmtDurationMin(end - (segs[0]?.start ?? 0))));
      } else if (entry.error) {
        head.append(el('span', 'dur', 'load failed'));
      }
      head.addEventListener('click', (ev) => {
        if (ev.target === node.chevron) return;
        this.onSelectDay(dayIdx);
      });
      node.root.append(head, node.children);
      this.treeEl.append(node.root);
      this.dayNodes.push(node);

      for (const segment of segs) {
        const segNode = this.makeFoldNode('segment');
        const segHead = el('div', 'node-head segment-head');
        segHead.append(segNode.chevron);
        segHead.append(el('span', `badge k-${segment.kind}`, segment.kind));
        segHead.append(el('span', 'seg-title', segment.title));
        segHead.append(el('span', 'dur', fmtDurationMin(segment.end - segment.start)));
        segHead.addEventListener('click', (ev) => {
          if (ev.target === segNode.chevron) return;
          this.activate(dayIdx, segment.start);
        });
        segNode.root.append(segHead, segNode.children);
        node.children.append(segNode.root);
        this.segNodes.push({ node: segNode, dayIdx, segment });

        for (const section of segment.sections ?? []) {
          const row = el('div', 'node-head section');
          row.append(el('span', `badge k-${section.kind}`, section.kind));
          row.append(el('span', 'seg-title', section.title));
          row.append(el('span', 'dur', fmtDurationMin(section.end - section.start)));
          row.addEventListener('click', () => this.activate(dayIdx, section.start));
          segNode.children.append(row);
        }
      }
    });

    this.prevBtn.addEventListener('click', () => {
      const segs = this.segmentsOfActiveDay();
      const t = stepTime(segs, this.lastTime, -1);
      if (t !== null) this.onSeek(t);
    });
    this.nextBtn.addEventListener('click', () => {
      const segs = this.segmentsOfActiveDay();
      const t = stepTime(segs, this.lastTime, 1);
      if (t !== null) this.onSeek(t);
    });
  }

  private makeFoldNode(kind: string): FoldNode {
    const root = el('div', `fold ${kind}`);
    const children = el('div', 'children');
    const chevron = el('button', 'chevron');
    chevron.type = 'button';
    chevron.setAttribute('aria-label', 'fold/expand');
    const node: FoldNode = { root, children, chevron, expanded: true, onPath: false };
    chevron.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (node.onPath) return; // the path to the current segment cannot be folded
      this.setExpanded(node, !node.expanded);
    });
    return node;
  }

  private setExpanded(node: FoldNode, expanded: boolean): void {
    node.expanded = expanded;
    node.root.classList.toggle('folded', !expanded);
    node.chevron.classList.toggle('open', expanded);
    node.chevron.disabled = node.onPath;
  }

  private activate(dayIdx: number, time: number): void {
    if (dayIdx === this.activeDay) this.onSeek(time);
    else this.onSelectDay(dayIdx, time);
  }

  private segmentsOfActiveDay(): Segment[] {
    return [...(this.days[this.activeDay]?.segments ?? [])].sort((a, b) => a.start - b.start);
  }

  setActiveDay(dayIdx: number): void {
    this.activeDay = dayIdx;
    this.activeSeg = -1;
    this.refold();
  }

  // Refold so that only the path to the current segment is expanded: the active
  // day and the segment containing the playback position stay open, everything
  // else folds, and the fold buttons on the path are disabled.
  private refold(): void {
    const segs = this.segmentsOfActiveDay();
    this.activeSeg = activeIndex(segs, this.lastTime);
    this.dayNodes.forEach((node, i) => {
      node.onPath = i === this.activeDay;
      this.setExpanded(node, node.onPath);
      node.root.classList.toggle('active', node.onPath);
    });
    for (const { node, dayIdx, segment } of this.segNodes) {
      const onPath = dayIdx === this.activeDay && segs[this.activeSeg] === segment;
      node.onPath = onPath;
      this.setExpanded(node, onPath);
      node.root.classList.toggle('active', onPath);
    }
  }

  update(t: number): void {
    this.lastTime = t;
    const segs = this.segmentsOfActiveDay();
    const segIdx = activeIndex(segs, t);
    if (segIdx !== this.activeSeg) {
      this.refold();
      if (this.activeSeg >= 0) {
        const found = this.segNodes.find((s) => s.dayIdx === this.activeDay && s.segment === segs[this.activeSeg]);
        found?.node.root.scrollIntoView({ block: 'nearest' });
      }
    }
    this.prevBtn.disabled = stepTime(segs, t, -1) === null;
    this.nextBtn.disabled = stepTime(segs, t, 1) === null;
  }
}
