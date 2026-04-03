import { useState, useEffect, useRef, useCallback } from "react";

const API = "/api";
const CTX_BEFORE = 3;
const CTX_AFTER = 20;

interface SmileInfo {
  video_id: string;
  start_ts: number;
  end_ts: number;
  score: number;
  frac_looking: number;
  deviation_euc: number;
  deviation_x: number;
  deviation_y: number;
}

interface Pair {
  subject_id: string;
  interviewer_angle_x: number;
  interviewer_angle_y: number;
  looking_at: SmileInfo;
  looking_away: SmileInfo;
}

function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00.0";
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
}

function radToDeg(r: number): string {
  return (r * (180 / Math.PI)).toFixed(1);
}

function SmilePlayer({
  smile,
  label,
  accentColor,
  syncRef,
}: {
  smile: SmileInfo;
  label: string;
  accentColor: string;
  syncRef?: React.MutableRefObject<HTMLVideoElement | null>;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const playStart = Math.max(0, smile.start_ts - CTX_BEFORE);
  const playEnd = smile.end_ts + CTX_AFTER;
  const isInSmile = currentTime >= smile.start_ts && currentTime <= smile.end_ts;

  useEffect(() => {
    if (syncRef) syncRef.current = videoRef.current;
  });

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const onTimeUpdate = () => {
      setCurrentTime(v.currentTime);
      if (v.currentTime >= playEnd) {
        v.currentTime = playStart;
        v.pause();
      }
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onLoadedData = () => {
      v.currentTime = playStart;
    };

    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("loadeddata", onLoadedData);
    return () => {
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("loadeddata", onLoadedData);
    };
  }, [smile.video_id, playStart, playEnd]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = playStart;
      v.load();
    }
  }, [smile.video_id, smile.start_ts]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  }, []);

  const restart = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = playStart;
    v.play();
  }, [playStart]);

  const seekBarFrac =
    playEnd > playStart
      ? ((currentTime - playStart) / (playEnd - playStart)) * 100
      : 0;

  const smileLeft =
    playEnd > playStart
      ? ((smile.start_ts - playStart) / (playEnd - playStart)) * 100
      : 0;
  const smileWidth =
    playEnd > playStart
      ? ((smile.end_ts - smile.start_ts) / (playEnd - playStart)) * 100
      : 0;

  return (
    <div style={{ flex: "1 1 0", minWidth: 0 }}>
      <div
        style={{
          textAlign: "center",
          padding: "6px 10px",
          backgroundColor: accentColor + "22",
          borderRadius: "8px 8px 0 0",
          borderBottom: `2px solid ${accentColor}`,
        }}
      >
        <span style={{ fontWeight: 700, color: accentColor, fontSize: "0.95rem" }}>
          {label}
        </span>
        <span style={{ fontSize: "0.75rem", color: "#94a3b8", marginLeft: "12px" }}>
          {smile.video_id} @ {fmtTime(smile.start_ts)}
        </span>
        <span style={{ fontSize: "0.75rem", color: "#64748b", marginLeft: "8px" }}>
          frac={smile.frac_looking.toFixed(2)} dev={radToDeg(smile.deviation_euc)}&deg;
        </span>
      </div>

      <div
        style={{
          borderRadius: "0 0 8px 8px",
          padding: "3px",
          backgroundColor: isInSmile ? "#f59e0b" : "transparent",
          transition: "background-color 0.2s",
        }}
      >
        <video
          ref={videoRef}
          src={`${API}/videos/${smile.video_id}/stream`}
          preload="auto"
          playsInline
          style={{
            width: "100%",
            borderRadius: "6px",
            backgroundColor: "#000",
            display: "block",
          }}
        />
      </div>

      {/* Seek bar */}
      <div
        style={{
          position: "relative",
          height: "12px",
          marginTop: "3px",
          backgroundColor: "#1e293b",
          borderRadius: "3px",
          cursor: "pointer",
          overflow: "hidden",
        }}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const pct = (e.clientX - rect.left) / rect.width;
          const v = videoRef.current;
          if (v) v.currentTime = playStart + pct * (playEnd - playStart);
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${smileLeft}%`,
            width: `${smileWidth}%`,
            top: 0,
            bottom: 0,
            backgroundColor: "#f59e0b55",
            borderLeft: "2px solid #f59e0b",
            borderRight: "2px solid #f59e0b",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${seekBarFrac}%`,
            top: 0,
            bottom: 0,
            width: "3px",
            backgroundColor: accentColor,
            borderRadius: "2px",
            transition: "left 0.05s linear",
          }}
        />
      </div>

      {/* Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "4px 6px",
          marginTop: "3px",
        }}
      >
        <button onClick={togglePlay} style={btnStyle}>
          {playing ? "\u23F8" : "\u25B6"}
        </button>
        <button onClick={restart} style={btnStyle}>
          \u21A9
        </button>
        {[0.5, 1, 1.5].map((r) => (
          <button
            key={r}
            style={{
              ...spdStyle,
              ...(speed === r ? { backgroundColor: accentColor, borderColor: accentColor, color: "#fff" } : {}),
            }}
            onClick={() => {
              setSpeed(r);
              const v = videoRef.current;
              if (v) v.playbackRate = r;
            }}
          >
            {r}x
          </button>
        ))}
        <span style={{ fontSize: "0.75rem", color: "#94a3b8", fontVariantNumeric: "tabular-nums" }}>
          {fmtTime(currentTime)}
        </span>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "4px 10px",
  backgroundColor: "#334155",
  color: "#e2e8f0",
  border: "1px solid #475569",
  borderRadius: "5px",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: "0.8rem",
};

const spdStyle: React.CSSProperties = {
  padding: "2px 6px",
  fontSize: "0.7rem",
  fontWeight: 500,
  border: "1px solid #475569",
  borderRadius: "3px",
  cursor: "pointer",
  backgroundColor: "#334155",
  color: "#e2e8f0",
};

export default function GazeVerify() {
  const [pairs, setPairs] = useState<Pair[]>([]);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/gaze-verify-pairs?n=15&seed=42`);
        const data = await res.json();
        if (data.error) {
          setError(data.error);
        } else {
          setPairs(data.pairs);
        }
      } catch (e: any) {
        setError(e.message);
      }
      setLoading(false);
    })();
  }, []);

  const pair = pairs[idx];

  if (loading) {
    return (
      <div style={pageStyle}>
        <div style={{ textAlign: "center", padding: "60px", color: "#64748b" }}>
          Loading gaze verification pairs...
        </div>
      </div>
    );
  }

  if (error || pairs.length === 0) {
    return (
      <div style={pageStyle}>
        <div style={{ textAlign: "center", padding: "60px", color: "#ef4444" }}>
          {error || "No eligible pairs found. Run the gaze pipeline first."}
        </div>
      </div>
    );
  }

  return (
    <div style={pageStyle}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          padding: "8px 14px",
          backgroundColor: "#1e293b",
          borderRadius: "8px",
          marginBottom: "10px",
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontWeight: 700, fontSize: "1.05rem", color: "#f8fafc" }}>
          Gaze Direction Verification
        </span>
        <span style={{ color: "#64748b", fontSize: "0.8rem" }}>
          Subject {pair.subject_id} &middot; Pair {idx + 1}/{pairs.length}
        </span>
        <span style={{ color: "#475569", fontSize: "0.75rem" }}>
          interviewer at ({radToDeg(pair.interviewer_angle_x)}&deg;,{" "}
          {radToDeg(pair.interviewer_angle_y)}&deg;)
        </span>

        <div style={{ marginLeft: "auto", display: "flex", gap: "6px" }}>
          <button
            style={btnStyle}
            onClick={() => setIdx((i) => Math.max(0, i - 1))}
            disabled={idx === 0}
          >
            &larr; Prev
          </button>
          <button
            style={btnStyle}
            onClick={() => setIdx((i) => Math.min(pairs.length - 1, i + 1))}
            disabled={idx === pairs.length - 1}
          >
            Next &rarr;
          </button>
        </div>
      </div>

      {/* Instruction */}
      <div
        style={{
          padding: "6px 14px",
          backgroundColor: "#1c1917",
          border: "1px solid #78350f",
          borderRadius: "6px",
          marginBottom: "10px",
          fontSize: "0.82rem",
          color: "#fbbf24",
          lineHeight: 1.5,
        }}
      >
        <strong>Goal:</strong> Verify the interviewer position estimate. The left
        clip is a smile where we think the subject is <strong>looking at the
        interviewer</strong>; the right is a smile where they are
        <strong> looking away</strong>. Play both (especially after the smile)
        and see if the gaze directions make sense — can you tell where the
        interviewer is?&nbsp;
        <span style={{ color: "#94a3b8" }}>
          Orange border = smile window. 20 s of context plays after the smile.
        </span>
      </div>

      {/* Side by side players */}
      <div style={{ display: "flex", gap: "16px", alignItems: "flex-start" }}>
        <SmilePlayer
          key={`${pair.subject_id}-at-${pair.looking_at.start_ts}`}
          smile={pair.looking_at}
          label="Looking AT interviewer"
          accentColor="#22c55e"
        />
        <SmilePlayer
          key={`${pair.subject_id}-away-${pair.looking_away.start_ts}`}
          smile={pair.looking_away}
          label="Looking AWAY from interviewer"
          accentColor="#ef4444"
        />
      </div>

      {/* Keyboard hint */}
      <div
        style={{
          marginTop: "12px",
          fontSize: "0.72rem",
          color: "#475569",
          textAlign: "center",
        }}
      >
        Click each video to load, then play. Use Prev/Next to navigate subjects.
      </div>
    </div>
  );
}

const pageStyle: React.CSSProperties = {
  padding: "10px 20px",
  maxWidth: "1600px",
  margin: "0 auto",
  fontFamily:
    'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  color: "#e2e8f0",
  backgroundColor: "#0f172a",
  minHeight: "100vh",
};
