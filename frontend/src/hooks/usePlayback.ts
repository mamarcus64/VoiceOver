import { useState, useRef, useCallback, useEffect } from "react";

export interface PlaybackState {
  currentTime: number;
  duration: number;
  playing: boolean;
  speed: number;
}

export function usePlayback() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<PlaybackState>({
    currentTime: 0,
    duration: 0,
    playing: false,
    speed: 1,
  });

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const onTimeUpdate = () =>
      setState((s) => ({ ...s, currentTime: v.currentTime }));
    const onDurationChange = () =>
      setState((s) => ({ ...s, duration: v.duration || 0 }));
    const onPlay = () => setState((s) => ({ ...s, playing: true }));
    const onPause = () => setState((s) => ({ ...s, playing: false }));
    const onRateChange = () =>
      setState((s) => ({ ...s, speed: v.playbackRate }));

    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("durationchange", onDurationChange);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ratechange", onRateChange);

    return () => {
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("durationchange", onDurationChange);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ratechange", onRateChange);
    };
  }, []);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  }, []);

  const seek = useCallback((time: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = time;
  }, []);

  const setSpeed = useCallback((rate: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
  }, []);

  return { videoRef, state, togglePlay, seek, setSpeed };
}
