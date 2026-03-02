import { useEffect, useRef, useCallback } from "react";

const DARK_BG = "#1a1a2e";
const VALENCE_COLOR = "#22c55e";
const AROUSAL_COLOR = "#ef4444";
const DOMINANCE_COLOR = "#3b82f6";
const PADDING_LEFT = 40;
const PADDING_RIGHT = 8;

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

    const x = (t: number) =>
      padding.left + (t / Math.max(duration, 0.001)) * chartW;
    const y = (v: number) =>
      padding.top + chartH - (v * chartH) / Math.max(chartH, 0.001);

    const drawLine = (
      points: { t: number; v: number }[],
      color: string,
      lineWidth: number = 2
    ) => {
      if (points.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.moveTo(x(points[0].t), y(points[0].v));
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(x(points[i].t), y(points[i].v));
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

      drawLine(vPoints, VALENCE_COLOR);
      drawLine(aPoints, AROUSAL_COLOR);
      drawLine(dPoints, DOMINANCE_COLOR);
    } else {
      const segs = segments as EyegazeSegment[];
      const vPoints = segs.map((s) => ({ t: s.timestamp, v: s.valence }));
      const aPoints = segs.map((s) => ({ t: s.timestamp, v: s.arousal }));
      const dPoints = segs.map((s) => ({ t: s.timestamp, v: s.dominance }));

      vPoints.sort((a, b) => a.t - b.t);
      aPoints.sort((a, b) => a.t - b.t);
      dPoints.sort((a, b) => a.t - b.t);

      drawLine(vPoints, VALENCE_COLOR);
      drawLine(aPoints, AROUSAL_COLOR);
      drawLine(dPoints, DOMINANCE_COLOR);
    }

    const playheadX = x(currentTime);
    ctx.beginPath();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.moveTo(playheadX, padding.top);
    ctx.lineTo(playheadX, h - padding.bottom);
    ctx.stroke();
  }, [segments, currentTime, duration, type]);

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
    const x = e.clientX - rect.left - PADDING_LEFT;
    const t = (Math.max(0, Math.min(x, chartW)) / Math.max(chartW, 1)) * duration;
    onSeek(t);
  };

  return (
    <div
      style={{
        backgroundColor: DARK_BG,
        borderRadius: "8px",
        padding: "12px",
      }}
    >
      <div
        style={{
          marginBottom: "8px",
          color: "#e2e8f0",
          fontWeight: 600,
          fontSize: "14px",
        }}
      >
        {title}
      </div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          width: "100%",
          height: "120px",
          display: "block",
          borderRadius: "4px",
          cursor: onSeek ? "pointer" : "default",
        }}
      />
    </div>
  );
}
