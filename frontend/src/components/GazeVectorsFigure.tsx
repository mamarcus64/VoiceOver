import { useEffect, useRef, useCallback, type RefObject } from "react";
import type { EyegazeVectorSample } from "../types";

const G0_COLOR = "#22d3ee";
const G1_COLOR = "#fbbf24";
const AXIS_ALPHA = 0.35;

function vecLen3(v: readonly [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize3(v: readonly [number, number, number]): [number, number, number] | null {
  const L = vecLen3(v);
  if (L < 1e-8) return null;
  return [v[0] / L, v[1] / L, v[2] / L];
}

function lerp3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  u: number
): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * u,
    a[1] + (b[1] - a[1]) * u,
    a[2] + (b[2] - a[2]) * u,
  ];
}

function interpAt(
  samples: EyegazeVectorSample[],
  t: number
): { g0: [number, number, number]; g1: [number, number, number] } | null {
  if (samples.length === 0) return null;
  if (t <= samples[0].t) {
    return { g0: [...samples[0].g0], g1: [...samples[0].g1] };
  }
  const last = samples[samples.length - 1];
  if (t >= last.t) {
    return { g0: [...last.g0], g1: [...last.g1] };
  }
  let lo = 0;
  let hi = samples.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = samples[lo];
  const b = samples[hi];
  const span = b.t - a.t;
  const u = span > 1e-12 ? (t - a.t) / span : 0;
  return { g0: lerp3(a.g0, b.g0, u), g1: lerp3(a.g1, b.g1, u) };
}

/** Weak perspective: head-space (x right, y down, z forward) → canvas. */
function project(
  x: number,
  y: number,
  z: number,
  cx: number,
  cy: number,
  scale: number
): [number, number] {
  const yaw = 0.55;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const xr = x * cos + z * sin;
  const zr = -x * sin + z * cos;
  const pitch = 0.32;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const yr = y * cp - zr * sp;
  return [cx + xr * scale, cy + yr * scale];
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  head = 8
) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const L = Math.hypot(dx, dy);
  if (L < 1e-6) return;
  const ux = dx / L;
  const uy = dy / L;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const bx = x1 - ux * head;
  const by = y1 - uy * head;
  const px = -uy;
  const py = ux;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(bx + px * (head * 0.45), by + py * (head * 0.45));
  ctx.lineTo(bx - px * (head * 0.45), by - py * (head * 0.45));
  ctx.closePath();
  ctx.fill();
}

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  samples: EyegazeVectorSample[] | null;
}

export default function GazeVectorsFigure({ videoRef, samples }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const scale = Math.min(w, h) * 0.38;

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = `rgba(148, 163, 184, ${AXIS_ALPHA})`;
    ctx.lineWidth = 1;
    const axisLen = 1.15;
    const axes: [number, number, number][] = [
      [axisLen, 0, 0],
      [-axisLen, 0, 0],
      [0, axisLen, 0],
      [0, -axisLen, 0],
      [0, 0, axisLen],
      [0, 0, -axisLen],
    ];
    for (const [ax, ay, az] of axes) {
      const [px, py] = project(ax, ay, az, cx, cy, scale);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(px, py);
      ctx.stroke();
    }

    ctx.fillStyle = "#64748b";
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();

    const t = v.currentTime;
    if (!samples?.length) return;
    const g = interpAt(samples, t);
    if (!g) return;

    const arrowScale = scale * 0.92;
    const n0 = normalize3(g.g0);
    const n1 = normalize3(g.g1);

    if (n0) {
      const [ex, ey] = project(n0[0], n0[1], n0[2], cx, cy, arrowScale);
      drawArrow(ctx, cx, cy, ex, ey, G0_COLOR);
    }
    if (n1) {
      const [ex, ey] = project(n1[0], n1[1], n1[2], cx, cy, arrowScale);
      drawArrow(ctx, cx, cy, ex, ey, G1_COLOR);
    }
  }, [videoRef, samples]);

  useEffect(() => {
    let id = 0;
    const tick = () => {
      drawFrame();
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [drawFrame]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const nw = Math.max(1, Math.round(rect.width * dpr));
      const nh = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== nw || canvas.height !== nh) {
        canvas.width = nw;
        canvas.height = nh;
      }
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  if (!samples?.length) return null;

  return (
    <div
      style={{
        marginTop: "10px",
        padding: "10px 12px",
        backgroundColor: "#1e293b",
        borderRadius: "8px",
        border: "1px solid #334155",
      }}
    >
      <div
        style={{
          fontSize: "0.85rem",
          fontWeight: 600,
          color: "#f8fafc",
          marginBottom: "8px",
        }}
      >
        Binocular gaze (head space)
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "200px", display: "block", borderRadius: "6px" }}
      />
      <div
        style={{
          display: "flex",
          gap: "16px",
          marginTop: "8px",
          fontSize: "0.75rem",
          color: "#94a3b8",
          flexWrap: "wrap",
        }}
      >
        <span>
          <span style={{ color: G0_COLOR, fontWeight: 600 }}>■</span> gaze_0
        </span>
        <span>
          <span style={{ color: G1_COLOR, fontWeight: 600 }}>■</span> gaze_1
        </span>
        <span style={{ marginLeft: "auto", opacity: 0.85 }}>
          Arrows: unit direction; faint lines: ±X, ±Y, ±Z in camera view
        </span>
      </div>
    </div>
  );
}
