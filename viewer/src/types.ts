// Shapes of the pipeline's intermediate files plus the viewer's own config.

export interface Cue {
  start: number;
  end: number;
  text: string;
  speaker_id: string | null;
}

export interface TranscriptDoc {
  model?: string;
  provider?: string;
  segments: Cue[];
  usage?: unknown;
  cost_usd?: number;
  _cache?: unknown;
}

export interface Section {
  kind: string; // content | exercise | break
  title: string;
  start: number;
  end: number;
}

export interface Segment {
  kind: string; // icebreaker | episode
  title: string;
  start: number;
  end: number;
  sections: Section[];
}

// segments.json is {"segments": [...]}; the day level comes from config.json.
export interface SegmentsDoc {
  segments: Segment[];
  _cache?: unknown;
}

export interface DayConfig {
  title: string | null;
  videoId: string;
  dir: string; // directory (relative to the site root) holding transcript.json / segments.json
  transcriptUrl?: string;
  segmentsUrl?: string;
}

export interface ViewerConfig {
  // single-video shortcut: one implicit day rooted at the site root
  videoId?: string;
  title?: string;
  transcriptUrl?: string;
  segmentsUrl?: string;
  // multi-day event: one entry per day video
  days?: {
    title?: string;
    videoId: string;
    dir?: string;
    transcriptUrl?: string;
    segmentsUrl?: string;
  }[];
}

export function configDays(config: ViewerConfig): DayConfig[] {
  if (Array.isArray(config.days) && config.days.length > 0) {
    return config.days.map((d, i) => ({
      title: d.title ?? `Day ${i + 1}`,
      videoId: d.videoId,
      dir: d.dir ?? '.',
      transcriptUrl: d.transcriptUrl,
      segmentsUrl: d.segmentsUrl,
    }));
  }
  if (config.videoId) {
    return [{
      title: config.title ?? null,
      videoId: config.videoId,
      dir: '.',
      transcriptUrl: config.transcriptUrl,
      segmentsUrl: config.segmentsUrl,
    }];
  }
  return [];
}
