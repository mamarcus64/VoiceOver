import { useState, useEffect, useMemo, useCallback } from "react";
import type { SmilingSegmentsData, SmileSegment, SmileParams } from "../types";
import { DEFAULT_SMILE_PARAMS } from "../types";

const STORAGE_KEY = "voiceover_smile_params";

interface FilteredMoment {
  start_ts: number;
  end_ts: number;
  peak_r: number;
  mean_r: number;
  play_start: number;
  play_end: number;
}

function loadParams(): SmileParams {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_SMILE_PARAMS, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  return { ...DEFAULT_SMILE_PARAMS };
}

function filterAndMerge(segments: SmileSegment[], params: SmileParams): FilteredMoment[] {
  let filtered = segments.filter((s) => s.mean_r >= params.intensityThreshold);

  filtered.sort((a, b) => a.start_ts - b.start_ts);

  const merged: { start_ts: number; end_ts: number; peak_r: number; mean_r: number }[] = [];
  for (const seg of filtered) {
    const last = merged[merged.length - 1];
    if (last && seg.start_ts - last.end_ts <= params.mergeDistance) {
      last.end_ts = Math.max(last.end_ts, seg.end_ts);
      last.peak_r = Math.max(last.peak_r, seg.peak_r);
      last.mean_r = (last.mean_r + seg.mean_r) / 2;
    } else {
      merged.push({ ...seg });
    }
  }

  const sustained = merged.filter((s) => s.end_ts - s.start_ts >= params.minDuration);

  return sustained.map((s) => ({
    ...s,
    play_start: Math.max(0, s.start_ts - params.contextBefore),
    play_end: s.end_ts + params.contextAfter,
  }));
}

interface Props {
  videoId: string;
  currentTime: number;
  onSeek: (time: number) => void;
}

export default function SmilingMoments({ videoId, currentTime, onSeek }: Props) {
  const [data, setData] = useState<SmilingSegmentsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<SmileParams>(loadParams);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);

  useEffect(() => {
    fetch(`/api/videos/${videoId}/smiling-segments`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setData)
      .catch((e) => setError(e.message));
  }, [videoId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  }, [params]);

  const moments = useMemo(
    () => (data ? filterAndMerge(data.segments, params) : []),
    [data, params]
  );

  const stats = useMemo(() => {
    if (!moments.length) return null;
    const totalDur = moments.reduce((s, m) => s + (m.end_ts - m.start_ts), 0);
    const meanDur = totalDur / moments.length;
    const meanPeak = moments.reduce((s, m) => s + m.peak_r, 0) / moments.length;
    return { count: moments.length, totalDur, meanDur, meanPeak };
  }, [moments]);

  const updateParam = useCallback(<K extends keyof SmileParams>(key: K, val: SmileParams[K]) => {
    setParams((p) => ({ ...p, [key]: val }));
  }, []);

  const handleMomentClick = useCallback((idx: number) => {
    setActiveIdx(idx);
    onSeek(moments[idx].play_start);
  }, [moments, onSeek]);

  const fmt = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (error) return <div style={{ color: "#f87171", padding: "12px" }}>No smiling data: {error}</div>;
  if (!data) return <div style={{ color: "#64748b", padding: "12px" }}>Loading smiling segments…</div>;

  return (
    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
      {/* Left: Parameters */}
      <div style={{
        flex: "0 0 280px", backgroundColor: "#1a1a2e", borderRadius: "8px", padding: "16px",
      }}>
        <h3 style={{ color: "#f8fafc", fontSize: "1rem", marginBottom: "16px" }}>Smile Parameters</h3>
        {([
          ["intensityThreshold", "Intensity threshold", 0.5, 4.0, 0.1],
          ["mergeDistance", "Merge distance (s)", 0, 5, 0.1],
          ["minDuration", "Min duration (s)", 0, 5, 0.1],
          ["contextBefore", "Context before (s)", 0, 10, 0.5],
          ["contextAfter", "Context after (s)", 0, 10, 0.5],
        ] as [keyof SmileParams, string, number, number, number][]).map(([key, label, min, max, step]) => (
          <div key={key} style={{ marginBottom: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", color: "#94a3b8", fontSize: "12px", marginBottom: "4px" }}>
              <span>{label}</span>
              <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{params[key]}</span>
            </div>
            <input
              type="range" min={min} max={max} step={step} value={params[key]}
              onChange={(e) => updateParam(key, parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "#f59e0b" }}
            />
          </div>
        ))}

        {/* Stats */}
        {stats && (
          <div style={{ marginTop: "16px", padding: "12px", backgroundColor: "#16213e", borderRadius: "6px" }}>
            <div style={{ color: "#f8fafc", fontSize: "13px", fontWeight: 600, marginBottom: "8px" }}>Stats</div>
            <div style={{ color: "#94a3b8", fontSize: "12px", lineHeight: 1.8 }}>
              Moments: <b style={{ color: "#f59e0b" }}>{stats.count}</b><br />
              Total smile: <b style={{ color: "#f59e0b" }}>{stats.totalDur.toFixed(1)}s</b><br />
              Mean duration: <b style={{ color: "#f59e0b" }}>{stats.meanDur.toFixed(2)}s</b><br />
              Mean peak: <b style={{ color: "#f59e0b" }}>{stats.meanPeak.toFixed(2)}</b><br />
              Raw segments: <b style={{ color: "#64748b" }}>{data.num_segments}</b>
            </div>
          </div>
        )}
      </div>

      {/* Right: Moments list */}
      <div style={{ flex: 1, minWidth: "300px", maxHeight: "500px", overflowY: "auto" }}>
        {moments.length === 0 ? (
          <div style={{ color: "#64748b", padding: "24px", textAlign: "center" }}>
            No smiling moments match current parameters.
          </div>
        ) : (
          moments.map((m, i) => {
            const isActive = activeIdx === i ||
              (currentTime >= m.play_start && currentTime <= m.play_end);
            return (
              <div
                key={`${m.start_ts}-${i}`}
                onClick={() => handleMomentClick(i)}
                style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "10px 14px", marginBottom: "4px", borderRadius: "6px", cursor: "pointer",
                  backgroundColor: isActive ? "#422006" : "#16213e",
                  border: isActive ? "1px solid #f59e0b" : "1px solid transparent",
                }}
              >
                <div>
                  <span style={{ color: "#f59e0b", fontWeight: 600, fontSize: "13px" }}>#{i + 1}</span>
                  <span style={{ color: "#e2e8f0", fontSize: "13px", marginLeft: "10px" }}>
                    {fmt(m.start_ts)} – {fmt(m.end_ts)}
                  </span>
                  <span style={{ color: "#64748b", fontSize: "12px", marginLeft: "8px" }}>
                    ({(m.end_ts - m.start_ts).toFixed(1)}s)
                  </span>
                </div>
                <div style={{ textAlign: "right" }}>
                  <span style={{ color: "#94a3b8", fontSize: "12px" }}>
                    peak {m.peak_r.toFixed(2)} · avg {m.mean_r.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
