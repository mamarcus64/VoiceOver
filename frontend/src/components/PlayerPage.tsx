import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { usePlayback } from "../hooks/usePlayback";
import TranscriptTrack from "./TranscriptTrack";
import EmotionTrack from "./EmotionTrack";
import AnnotationOverlay from "./AnnotationOverlay";
import type { Utterance, AudioVADData, EyegazeVADData } from "../types";

const API_BASE = "/api/videos";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "24px",
    maxWidth: "1200px",
    margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  header: { display: "flex", alignItems: "center", gap: "16px", marginBottom: "20px" },
  backBtn: {
    padding: "8px 14px", backgroundColor: "#334155", color: "#e2e8f0",
    border: "1px solid #475569", borderRadius: "6px", cursor: "pointer", fontSize: "0.9rem",
  },
  title: { fontSize: "1.5rem", fontWeight: 600, color: "#f8fafc", margin: 0 },
  videoContainer: { maxWidth: "960px", margin: "0 auto 16px" },
  video: { width: "100%", borderRadius: "8px", backgroundColor: "#000" },
  controls: {
    display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" as const,
    padding: "12px 16px", backgroundColor: "#1e293b", borderRadius: "8px", marginBottom: "20px",
  },
  playPause: {
    padding: "8px 16px", backgroundColor: "#3b82f6", color: "#fff",
    border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: 600, fontSize: "0.9rem",
  },
  speedGroup: { display: "flex", gap: "4px" },
  speedBtn: {
    padding: "6px 10px", border: "1px solid #475569", borderRadius: "4px",
    cursor: "pointer", fontSize: "0.8rem", fontWeight: 500, backgroundColor: "#334155", color: "#e2e8f0",
  },
  speedActive: { backgroundColor: "#3b82f6", borderColor: "#3b82f6", color: "#fff" },
  timeDisplay: { color: "#94a3b8", fontSize: "0.9rem", fontVariantNumeric: "tabular-nums", minWidth: "100px" },
  seekBar: { flex: 1, minWidth: "200px", accentColor: "#3b82f6" },
  grid: { display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "16px", marginBottom: "20px" },
  colTitle: { fontSize: "1rem", fontWeight: 600, color: "#f8fafc", marginBottom: "8px" },
  emotionStack: { display: "flex", flexDirection: "column" as const, gap: "12px" },
  msg: { textAlign: "center" as const, padding: "32px", color: "#64748b" },
};

const SPEEDS = [0.5, 1, 1.5, 2];

export default function PlayerPage() {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();
  const { videoRef, state, togglePlay, seek, setSpeed } = usePlayback();

  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [audioEmotion, setAudioEmotion] = useState<AudioVADData | null>(null);
  const [eyegazeEmotion, setEyegazeEmotion] = useState<EyegazeVADData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!videoId) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [tRes, aRes, eRes] = await Promise.all([
          fetch(`${API_BASE}/${videoId}/transcript`),
          fetch(`${API_BASE}/${videoId}/audio-emotion`),
          fetch(`${API_BASE}/${videoId}/eyegaze-emotion`).catch(() => null),
        ]);
        if (cancelled) return;
        if (!tRes.ok) { setError(`Transcript: ${tRes.status}`); return; }
        setUtterances(await tRes.json());
        if (aRes.ok) setAudioEmotion(await aRes.json());
        if (eRes?.ok) setEyegazeEmotion(await eRes.json());
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Load failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [videoId]);

  if (!videoId) return <div style={st.page}><p>No video ID</p></div>;

  return (
    <div style={st.page}>
      <header style={st.header}>
        <button style={st.backBtn} onClick={() => navigate("/")}>&#8592; Back</button>
        <h1 style={st.title}>{videoId}</h1>
      </header>

      <div style={st.videoContainer}>
        <video ref={videoRef} src={`${API_BASE}/${videoId}/stream`} preload="auto" playsInline style={st.video} />
      </div>

      <div style={st.controls}>
        <button style={st.playPause} onClick={togglePlay}>
          {state.playing ? "\u23F8 Pause" : "\u25B6 Play"}
        </button>
        <div style={st.speedGroup}>
          {SPEEDS.map((r) => (
            <button key={r} style={{ ...st.speedBtn, ...(state.speed === r ? st.speedActive : {}) }} onClick={() => setSpeed(r)}>
              {r}x
            </button>
          ))}
        </div>
        <span style={st.timeDisplay}>{formatTime(state.currentTime)} / {formatTime(state.duration)}</span>
        <input type="range" min={0} max={state.duration || 1} step={0.1} value={state.currentTime}
          onChange={(e) => seek(Number(e.target.value))} style={st.seekBar} />
      </div>

      {loading && <p style={st.msg}>Loading transcript &amp; emotions&hellip;</p>}
      {error && <p style={{ ...st.msg, color: "#f87171" }}>{error}</p>}

      <div style={st.grid}>
        <div>
          <h3 style={st.colTitle}>Transcript</h3>
          <TranscriptTrack utterances={utterances} currentTimeMs={state.currentTime * 1000} onSeek={(ms) => seek(ms / 1000)} />
        </div>
        <div style={st.emotionStack}>
          <EmotionTrack title="Audio Emotion" segments={audioEmotion?.segments ?? []} currentTime={state.currentTime}
            duration={state.duration} type="audio" onSeek={seek} />
          <EmotionTrack title="Eyegaze Emotion" segments={eyegazeEmotion?.segments ?? []} currentTime={state.currentTime}
            duration={state.duration} type="eyegaze" onSeek={seek} />
        </div>
      </div>

      <AnnotationOverlay currentTime={state.currentTime} duration={state.duration} videoId={videoId} />
    </div>
  );
}
