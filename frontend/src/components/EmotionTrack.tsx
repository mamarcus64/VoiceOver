import { useEffect, useRef, useCallback, useState } from "react";

const DARK_BG = "#1a1a2e";
const VALENCE_COLOR = "#22c55e";
const AROUSAL_COLOR = "#ef4444";
const DOMINANCE_COLOR = "#3b82f6";
const PADDING_LEFT = 40;
const PADDING_RIGHT = 8;

// Half-width of the zoom window in seconds
const ZOOM_HALF = 5;

interface AudioSegment {
  start: number;
  end: number;
  valence: number;
  arousal: number;
  dominance: number;
}

interface EyegazeSegment {
  timestamp: number;
  valence: number;
  arousal: number;
  dominance: number;
}

interface EmotionTrackProps {
  title: string;
  segments: AudioSegment[] | EyegazeSegment[];
  currentTime: number;
  duration: number;
  type: "audio" | "eyegaze";
  onSeek?: (time: number) => void;
}

export default function EmotionTrack({
  title,
  segments,
  currentTime,
  duration,
  type,
  onSeek,
}: EmotionTrackProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [showValence, setShowValence] = useState(true);
  const [showArousal, setShowArousal] = useState(true);
  const [showDominance, setShowDominance] = useState(false);

  // The visible time window centered on currentTime
  const winStart = Math.max(0, currentTime - ZOOM_HALF);
  const winEnd = Math.min(Math.max(duration, 0.001), currentTime + ZOOM_HALF);
  const winDuration = winEnd - winStart;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = DARK_BG;
    ctx.fillRect(0, 0, w, h);

    const padding = { top: 8, right: PADDING_RIGHT, bottom: 20, left: PADDING_LEFT };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Map absolute time → canvas x using the zoom window
    const x = (t: number) =>
      padding.left + ((t - winStart) / Math.max(winDuration, 0.001)) * chartW;
    const y = (v: number) => padding.top + chartH * (1 - v);

    // Horizontal grid lines at y=0.0, 0.5, 1.0
    const gridColor = "rgba(100,116,139,0.3)";
    for (const vVal of [0, 0.5, 1]) {
      ctx.beginPath();
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.moveTo(padding.left, y(vVal));
      ctx.lineTo(padding.left + chartW, y(vVal));
      ctx.stroke();
    }

    // Y-axis labels
    const labelColor = "#64748b";
    ctx.fillStyle = labelColor;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const labelX = padding.left - 6;
    ctx.fillText("1.0", labelX, y(1));
    ctx.fillText("0.5", labelX, y(0.5));
    ctx.fillText("0.0", labelX, y(0));

    const drawLine = (
      points: { t: number; v: number }[],
      color: string,
      lineWidth: number = 2
    ) => {
      // Only keep points within the visible window (with a small buffer)
      const visible = points.filter(
        (p) => p.t >= winStart - 0.5 && p.t <= winEnd + 0.5
      );
      if (visible.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.moveTo(x(visible[0].t), y(visible[0].v));
      for (let i = 1; i < visible.length; i++) {
        ctx.lineTo(x(visible[i].t), y(visible[i].v));
      }
      ctx.stroke();
    };

    if (type === "audio") {
      const segs = segments as AudioSegment[];
      const vPoints: { t: number; v: number }[] = [];
      const aPoints: { t: number; v: number }[] = [];
      const dPoints: { t: number; v: number }[] = [];

      for (const s of segs) {
        vPoints.push({ t: s.start, v: s.valence });
        vPoints.push({ t: s.end, v: s.valence });
        aPoints.push({ t: s.start, v: s.arousal });
        aPoints.push({ t: s.end, v: s.arousal });
        dPoints.push({ t: s.start, v: s.dominance });
        dPoints.push({ t: s.end, v: s.dominance });
      }

      const sortByT = (a: { t: number }, b: { t: number }) => a.t - b.t;
      vPoints.sort(sortByT);
      aPoints.sort(sortByT);
      dPoints.sort(sortByT);

      if (showValence) drawLine(vPoints, VALENCE_COLOR);
      if (showArousal) drawLine(aPoints, AROUSAL_COLOR);
      if (showDominance) drawLine(dPoints, DOMINANCE_COLOR);
    } else {
      const segs = segments as EyegazeSegment[];
      const vPoints = segs.map((s) => ({ t: s.timestamp, v: s.valence }));
      const aPoints = segs.map((s) => ({ t: s.timestamp, v: s.arousal }));
      const dPoints = segs.map((s) => ({ t: s.timestamp, v: s.dominance }));

      vPoints.sort((a, b) => a.t - b.t);
      aPoints.sort((a, b) => a.t - b.t);
      dPoints.sort((a, b) => a.t - b.t);

      if (showValence) drawLine(vPoints, VALENCE_COLOR);
      if (showArousal) drawLine(aPoints, AROUSAL_COLOR);
      if (showDominance) drawLine(dPoints, DOMINANCE_COLOR);
    }

    // Playhead is always centered in the zoom window
    const playheadX = x(currentTime);
    ctx.beginPath();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.moveTo(playheadX, padding.top);
    ctx.lineTo(playheadX, h - padding.bottom);
    ctx.stroke();

    // Time labels: left edge, center, right edge of window
    ctx.fillStyle = labelColor;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const timeY = h - padding.bottom + 3;
    const fmt = (t: number) => {
      const m = Math.floor(t / 60);
      const s = Math.floor(t % 60);
      return `${m}:${s.toString().padStart(2, "0")}`;
    };
    ctx.fillText(fmt(winStart), padding.left, timeY);
    ctx.fillText(fmt(currentTime), padding.left + chartW / 2, timeY);
    ctx.fillText(fmt(winEnd), padding.left + chartW, timeY);
  }, [
    segments,
    currentTime,
    duration,
    type,
    showValence,
    showArousal,
    showDominance,
    winStart,
    winEnd,
    winDuration,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const chartW = rect.width - PADDING_LEFT - PADDING_RIGHT;
    const xPos = e.clientX - rect.left - PADDING_LEFT;
    const frac = Math.max(0, Math.min(xPos, chartW)) / Math.max(chartW, 1);
    const t = winStart + frac * winDuration;
    onSeek(Math.max(0, Math.min(t, duration)));
  };

  const toggleBtn = (
    label: string,
    color: string,
    active: boolean,
    toggle: () => void
  ) => (
    <button
      onClick={toggle}
      style={{
        padding: "2px 10px",
        borderRadius: "12px",
        border: `1.5px solid ${color}`,
        backgroundColor: active ? color : "transparent",
        color: active ? "#1a1a2e" : color,
        fontSize: "11px",
        fontWeight: 600,
        cursor: "pointer",
        transition: "background 0.15s, color 0.15s",
      }}
    >
      {label}
    </button>
  );

  return (
    <div
      style={{
        backgroundColor: DARK_BG,
        borderRadius: "8px",
        padding: "8px 10px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "8px",
        }}
      >
        <span style={{ color: "#e2e8f0", fontWeight: 600, fontSize: "14px" }}>
          {title}
        </span>
        <div style={{ display: "flex", gap: "6px" }}>
          {toggleBtn("Valence", VALENCE_COLOR, showValence, () =>
            setShowValence((v) => !v)
          )}
          {toggleBtn("Arousal", AROUSAL_COLOR, showArousal, () =>
            setShowArousal((v) => !v)
          )}
          {toggleBtn("Dominance", DOMINANCE_COLOR, showDominance, () =>
            setShowDominance((v) => !v)
          )}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          width: "100%",
          height: "90px",
          display: "block",
          borderRadius: "4px",
          cursor: onSeek ? "pointer" : "default",
        }}
      />
    </div>
  );
}
