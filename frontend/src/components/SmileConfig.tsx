import { useState, useEffect, useCallback } from "react";
import type { SmileConfigData, SmilePreviewStats } from "../types";

const DEFAULTS: SmileConfigData = {
  intensityThreshold: 1.5,
  mergeDistance: 1.0,
  minDuration: 0.5,
  maxPerVideo: 10,
  contextBefore: 10.0,
  contextAfter: 5.0,
};

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "24px 32px",
    maxWidth: "900px",
    margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  title: { fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", marginBottom: "24px" },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "16px",
    marginBottom: "24px",
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: "10px",
    padding: "16px 20px",
  },
  sectionTitle: {
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "#94a3b8",
    marginBottom: "12px",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
  },
  field: { marginBottom: "12px" },
  label: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "0.85rem",
    color: "#cbd5e1",
    marginBottom: "4px",
  },
  value: { fontWeight: 700, color: "#f8fafc" },
  slider: { width: "100%", accentColor: "#3b82f6" },
  btnRow: { display: "flex", gap: "10px", marginBottom: "24px" },
  btn: {
    padding: "10px 20px",
    fontSize: "0.9rem",
    fontWeight: 600,
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    color: "#fff",
  },
  statsCard: {
    backgroundColor: "#1e293b",
    borderRadius: "10px",
    padding: "20px 24px",
  },
  statRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    borderBottom: "1px solid #334155",
    fontSize: "0.9rem",
  },
  statLabel: { color: "#94a3b8" },
  statValue: { color: "#f8fafc", fontWeight: 600 },
  msg: { fontSize: "0.85rem", marginTop: "8px" },
};

type ParamField = {
  key: keyof SmileConfigData;
  label: string;
  min: number;
  max: number;
  step: number;
};

const TASK_PARAMS: ParamField[] = [
  { key: "intensityThreshold", label: "Intensity Threshold", min: 0.5, max: 4.0, step: 0.1 },
  { key: "mergeDistance", label: "Merge Distance (s)", min: 0, max: 5, step: 0.1 },
  { key: "minDuration", label: "Min Duration (s)", min: 0, max: 5, step: 0.1 },
  { key: "maxPerVideo", label: "Max Smiles Per Video", min: 1, max: 50, step: 1 },
];

const UI_PARAMS: ParamField[] = [
  { key: "contextBefore", label: "Context Before (s)", min: 0, max: 30, step: 0.5 },
  { key: "contextAfter", label: "Context After (s)", min: 0, max: 30, step: 0.5 },
];

export default function SmileConfig() {
  const [config, setConfig] = useState<SmileConfigData>({ ...DEFAULTS });
  const [preview, setPreview] = useState<SmilePreviewStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  useEffect(() => {
    fetch("/api/smile-config")
      .then((r) => r.json())
      .then((d) => setConfig({ ...DEFAULTS, ...d }))
      .catch(() => {});
  }, []);

  const update = useCallback(<K extends keyof SmileConfigData>(key: K, val: SmileConfigData[K]) => {
    setConfig((c) => ({ ...c, [key]: val }));
  }, []);

  const handlePreview = async () => {
    setLoading(true);
    setMsg(null);
    try {
      const res = await fetch("/api/smile-config/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      setPreview(await res.json());
    } catch {
      setMsg({ text: "Preview failed", ok: false });
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!confirm("Generate task manifest? This will overwrite any existing manifest.")) return;
    setLoading(true);
    setMsg(null);
    try {
      const res = await fetch("/api/smile-config/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      setMsg({ text: `Manifest generated: ${data.total_tasks} tasks across ${data.videos_with_tasks} videos`, ok: true });
    } catch {
      setMsg({ text: "Generation failed", ok: false });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    setMsg(null);
    try {
      await fetch("/api/smile-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      setMsg({ text: "Config saved", ok: true });
    } catch {
      setMsg({ text: "Save failed", ok: false });
    } finally {
      setLoading(false);
    }
  };

  const renderSlider = (f: ParamField) => (
    <div key={f.key} style={st.field}>
      <div style={st.label}>
        <span>{f.label}</span>
        <span style={st.value}>{config[f.key]}</span>
      </div>
      <input
        type="range"
        min={f.min}
        max={f.max}
        step={f.step}
        value={config[f.key]}
        onChange={(e) => update(f.key, parseFloat(e.target.value) as SmileConfigData[typeof f.key] & number)}
        style={st.slider}
      />
    </div>
  );

  return (
    <div style={st.page}>
      <h1 style={st.title}>Smile Annotation Config</h1>

      <div style={st.grid}>
        <div style={st.card}>
          <div style={st.sectionTitle}>Task Generation Parameters</div>
          {TASK_PARAMS.map(renderSlider)}
        </div>
        <div style={st.card}>
          <div style={st.sectionTitle}>Annotation UI Parameters</div>
          {UI_PARAMS.map(renderSlider)}
        </div>
      </div>

      <div style={st.btnRow}>
        <button
          style={{ ...st.btn, backgroundColor: "#6366f1" }}
          onClick={handlePreview}
          disabled={loading}
        >
          {loading ? "Working..." : "Preview Stats"}
        </button>
        <button
          style={{ ...st.btn, backgroundColor: "#f59e0b" }}
          onClick={handleGenerate}
          disabled={loading}
        >
          Generate Manifest
        </button>
        <button
          style={{ ...st.btn, backgroundColor: "#22c55e" }}
          onClick={handleSave}
          disabled={loading}
        >
          Save Config
        </button>
      </div>

      {msg && (
        <div style={{ ...st.msg, color: msg.ok ? "#22c55e" : "#f87171" }}>
          {msg.text}
        </div>
      )}

      {preview && (
        <div style={st.statsCard}>
          <div style={st.sectionTitle}>Preview</div>
          {([
            ["Total Tasks", preview.total_tasks],
            ["Videos With Tasks", preview.videos_with_tasks],
            ["Tasks/Video (mean)", preview.tasks_per_video_mean],
            ["Tasks/Video (median)", preview.tasks_per_video_median],
            ["Tasks/Video (max)", preview.tasks_per_video_max],
          ] as [string, number][]).map(([label, val]) => (
            <div key={label} style={st.statRow}>
              <span style={st.statLabel}>{label}</span>
              <span style={st.statValue}>{val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
