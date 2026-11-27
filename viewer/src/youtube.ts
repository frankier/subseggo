// Thin wrapper around the YouTube IFrame Player API. The API's script only
// exposes a global YT object, so declare just the surface this viewer uses.
// The embed is created with controls=0; transport (play/pause, seek bar) is
// the viewer's own UI.

declare global {
  namespace YT {
    interface PlayerEvent {
      target: Player;
    }
    interface PlayerOptions {
      videoId?: string;
      playerVars?: Record<string, unknown>;
      events?: {
        onReady?: (event: PlayerEvent) => void;
        onStateChange?: (event: PlayerEvent) => void;
      };
    }
    class Player {
      constructor(element: HTMLElement | string, options: PlayerOptions);
      seekTo(seconds: number, allowSeekAhead: boolean): void;
      getCurrentTime(): number;
      getDuration(): number;
      playVideo(): void;
      pauseVideo(): void;
      getPlayerState(): number;
      loadVideoById(options: { videoId: string; startSeconds?: number }): void;
      destroy(): void;
    }
  }
  interface Window {
    YT?: typeof YT;
    onYouTubeIframeAPIReady?: () => void;
  }
}

export const PLAYER_STATE = { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 } as const;

export interface PlayerHandle {
  seekTo(seconds: number): void;
  getCurrentTime(): number;
  getDuration(): number;
  play(): void;
  pause(): void;
  getState(): number;
  loadVideo(videoId: string, startSeconds?: number): void;
}

let apiPromise: Promise<void> | null = null;

export function loadYouTubeApi(): Promise<void> {
  if (window.YT?.Player) return Promise.resolve();
  if (!apiPromise) {
    apiPromise = new Promise((resolve) => {
      const previous = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => {
        previous?.();
        resolve();
      };
      const tag = document.createElement('script');
      tag.src = 'https://www.youtube.com/iframe_api';
      tag.async = true;
      document.head.appendChild(tag);
    });
  }
  return apiPromise;
}

export interface PlayerHandlers {
  onReady?: (player: PlayerHandle) => void;
  onStateChange?: (state: number) => void;
}

export async function createPlayer(
  videoId: string,
  element: HTMLElement,
  handlers: PlayerHandlers,
): Promise<PlayerHandle> {
  await loadYouTubeApi();
  return new Promise((resolve) => {
    let handle: PlayerHandle;
    const player = new YT.Player(element, {
      videoId,
      playerVars: { controls: 0, rel: 0, modestbranding: 1, disablekb: 1 },
      events: {
        onReady: () => {
          handlers.onReady?.(handle);
          resolve(handle);
        },
        onStateChange: (event) => handlers.onStateChange?.(event.target.getPlayerState()),
      },
    });
    handle = {
      seekTo: (s) => player.seekTo(s, true),
      getCurrentTime: () => player.getCurrentTime(),
      getDuration: () => player.getDuration(),
      play: () => player.playVideo(),
      pause: () => player.pauseVideo(),
      getState: () => player.getPlayerState(),
      loadVideo: (v, t) => player.loadVideoById({ videoId: v, startSeconds: t }),
    };
  });
}
