import { useState, useCallback } from "react";
import { useAnnotation } from "../hooks/useAnnotation";
import { EMOTION_COLORS, EMOTION_LABELS } from "../types";

const DARK_BG = "#1a1a2e";

interface AnnotationOverlayProps {
  currentTime: number;
  duration: number;
  videoId: string;
}

export default function AnnotationOverlay({
  currentTime,
  duration,
  videoId,
}: AnnotationOverlayProps) {
  const getCurrentTime = useCallback(() => currentTime, [currentTime]);
  const {
    annotations,
    annotator,
    setAnnotator,
    clearAnnotations,
    loadAnnotations,
  } = useAnnotation(getCurrentTime);

  const [status, setStatus] = useState<string | null>(null);

  const handleSave = async () => {
    setStatus(null);
    try {
      const res = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          annotator,
          annotations,
        }),
      });
      if (!res.ok) throw new Error("Save failed");
      setStatus("Saved successfully");
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : "Save failed"}`);
    }
  };

  const handleLoad = async () => {
    setStatus(null);
    try {
      const res = await fetch(
        `/api/annotations?video_id=${encodeURIComponent(videoId)}&annotator=${encodeURIComponent(annotator)}`
      );
      if (!res.ok) throw new Error("Load failed");
      const data = await res.json();
      loadAnnotations(data.annotations ?? []);
      setStatus("Loaded successfully");
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : "Load failed"}`);
    }
  };

  const handleClear = () => {
    clearAnnotations();
    setStatus("Annotations cleared");
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
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: "12px",
          marginBottom: "12px",
        }}
      >
        <label
          style={{
            color: "#94a3b8",
            fontSize: "13px",
          }}
        >
          Annotator:
          <input
            type="text"
            value={annotator}
            onChange={(e) => setAnnotator(e.target.value)}
            placeholder="Your name"
            style={{
              marginLeft: "8px",
              padding: "6px 10px",
              backgroundColor: "#16213e",
              border: "1px solid #334155",
              borderRadius: "4px",
              color: "#e2e8f0",
              fontSize: "13px",
            }}
          />
        </label>
        <span
          style={{
            color: "#94a3b8",
            fontSize: "13px",
          }}
        >
          {annotations.length} annotation{annotations.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div
        style={{
          width: "100%",
          height: "40px",
          backgroundColor: "#16213e",
          borderRadius: "4px",
          position: "relative",
          overflow: "hidden",
          marginBottom: "12px",
        }}
      >
        {annotations.map((a) => {
          const left =
            duration > 0 ? (a.start_sec / duration) * 100 : 0;
          const width =
            duration > 0
              ? ((a.end_sec - a.start_sec) / duration) * 100
              : 0;
          const color = EMOTION_COLORS[a.label] ?? "#94a3b8";

          return (
            <div
              key={`${a.start_sec}-${a.end_sec}-${a.key}`}
              style={{
                position: "absolute",
                left: `${left}%`,
                width: `${width}%`,
                height: "100%",
                backgroundColor: color,
                borderRadius: "2px",
              }}
            />
          );
        })}
      </div>

      <div
        style={{
          marginBottom: "12px",
          color: "#94a3b8",
          fontSize: "12px",
        }}
      >
        Hold 1-5 to annotate:{" "}
        {[1, 2, 3, 4, 5].map((k) => (
          <span
            key={k}
            style={{
              display: "inline-block",
              marginRight: "12px",
              padding: "2px 6px",
              backgroundColor: EMOTION_COLORS[EMOTION_LABELS[k]] ?? "#334155",
              borderRadius: "4px",
              color: "#1a1a2e",
              fontWeight: 600,
            }}
          >
            {k}: {EMOTION_LABELS[k]}
          </span>
        ))}
      </div>

      <div
        style={{
          display: "flex",
          gap: "8px",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={handleSave}
          style={{
            padding: "8px 16px",
            backgroundColor: "#22c55e",
            border: "none",
            borderRadius: "4px",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Save
        </button>
        <button
          onClick={handleLoad}
          style={{
            padding: "8px 16px",
            backgroundColor: "#3b82f6",
            border: "none",
            borderRadius: "4px",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Load
        </button>
        <button
          onClick={handleClear}
          style={{
            padding: "8px 16px",
            backgroundColor: "#ef4444",
            border: "none",
            borderRadius: "4px",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Clear
        </button>
      </div>

      {status && (
        <div
          style={{
            marginTop: "12px",
            color: status.startsWith("Error") ? "#f87171" : "#86efac",
            fontSize: "13px",
          }}
        >
          {status}
        </div>
      )}
    </div>
  );
}
