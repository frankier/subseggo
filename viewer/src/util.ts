export async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status} ${res.statusText}`);
  return res.json();
}

export function fmtTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h > 0 ? `${h}:` : ''}${h > 0 ? String(m).padStart(2, '0') : String(m)}:${String(s).padStart(2, '0')}`;
}

// Duration in whole minutes; anything under a minute is "<1 min".
export function fmtDurationMin(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 60) return '<1 min';
  return `${Math.round(seconds / 60)} min`;
}

// Joins a day directory with a file name; "." / "" means the site root.
export function joinUrl(dir: string, file: string): string {
  const d = dir.replace(/\/+$/, '');
  return d === '' || d === '.' ? file : `${d}/${file}`;
}

// Index of the item active at time t: the last item whose start <= t (-1 if none).
export function activeIndex(items: { start: number }[], t: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].start <= t) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return idx;
}

// Where a prev/next jump should land. Next: first item starting after t.
// Prev: restart the current item if we are more than 1.5 s into it, otherwise
// the item before it (0 when already at the first). Returns null when there is
// no next item.
export function stepTime(items: { start: number }[], t: number, dir: 1 | -1): number | null {
  const i = activeIndex(items, t);
  if (dir === 1) return i + 1 < items.length ? items[i + 1].start : null;
  if (i >= 0 && t - items[i].start > 1.5) return items[i].start;
  if (i - 1 >= 0) return items[i - 1].start;
  return i >= 0 ? items[i].start : 0;
}

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
