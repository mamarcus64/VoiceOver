import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useNavigate, Link } from "react-router-dom";
import TranscriptTrack from "./TranscriptTrack";
import type { Utterance } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const API = "/api";

interface RecallTask {
  task_number: number;
  video_id: string;
  segment_start: number;
  segment_end: number;
  total_tasks: number;
  available_tasks: number;
  video_downloaded: boolean;
}

interface RecallAnnotations {
  annotator: string;
  annotations: Record<string, { label: string; timestamp: string }>;
}

const BEFORE_SEC = 3;
const AFTER_SEC = 2;
const SPEEDS = [0.5, 1, 1.5, 2];

function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "8px 16px", maxWidth: "1400px", margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0", backgroundColor: "#0f172a", minHeight: "100vh",
  },
  topBar: {
    display: "flex", alignItems: "center", gap: "10px",
    padding: "7px 14px", backgroundColor: "#1e293b", borderRadius: "8px",
    marginBottom: "8px", flexWrap: "wrap" as const,
  },
  taskLabel: { fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc" },
  annotator: { fontSize: "0.8rem", color: "#94a3b8" },
  navBtn: {
    padding: "4px 12px", fontSize: "0.8rem", fontWeight: 600,
    border: "1px solid #475569", borderRadius: "5px", cursor: "pointer",
    backgroundColor: "#334155", color: "#e2e8f0",
  },
  jumpInput: {
    width: "60px", padding: "4px 6px", fontSize: "0.8rem",
    border: "1px solid #475569", borderRadius: "5px",
    backgroundColor: "#0f172a", color: "#e2e8f0", textAlign: "center" as const,
  },
  logoutBtn: {
    padding: "4px 10px", fontSize: "0.75rem", border: "1px solid #475569",
    borderRadius: "5px", cursor: "pointer", backgroundColor: "#334155", color: "#94a3b8",
  },
  mainLayout: { display: "flex", gap: "12px", alignItems: "flex-start" },
  leftPanel: { flex: "1 1 60%", minWidth: 0, display: "flex", flexDirection: "column" as const },
  rightPanel: { flex: "0 0 380px", display: "flex", flexDirection: "column" as const },
  video: { width: "100%", borderRadius: "8px", backgroundColor: "#000", maxHeight: "45vh" },
  controls: {
    display: "flex", alignItems: "center", gap: "6px", padding: "4px 8px",
    backgroundColor: "#1e293b", borderRadius: "6px", marginTop: "4px", flexWrap: "wrap" as const,
  },
  playBtn: {
    padding: "5px 12px", backgroundColor: "#3b82f6", color: "#fff",
    border: "none", borderRadius: "5px", cursor: "pointer", fontWeight: 600, fontSize: "0.8rem",
  },
  optBtn: {
    padding: "3px 6px", border: "1px solid #475569", borderRadius: "3px",
    cursor: "pointer", fontSize: "0.7rem", fontWeight: 500, backgroundColor: "#334155", color: "#e2e8f0",
  },
  optActive: { backgroundColor: "#3b82f6", borderColor: "#3b82f6", color: "#fff" },
  timeDisplay: { color: "#94a3b8", fontSize: "0.8rem", fontVariantNumeric: "tabular-nums" },
  notDownloaded: {
    display: "flex", alignItems: "center", justifyContent: "center", height: "200px",
    backgroundColor: "#1e293b", borderRadius: "8px", color: "#f59e0b", fontSize: "1rem", fontWeight: 600,
  },
  loading: { textAlign: "center" as const, padding: "48px", color: "#64748b", fontSize: "1.1rem" },
};

export default function RecallAnnotate() {
  const navigate = useNavigate();
  const annotator = localStorage.getItem(STORAGE_KEY);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);

  const [taskNum, setTaskNum] = useState<number | null>(null);
  const [task, setTask] = useState<RecallTask | null>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [annotations, setAnnotations] = useState<RecallAnnotations | null>(null);
  const [jumpVal, setJumpVal] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Init: fetch next incomplete + existing annotations
  useEffect(() => {
    if (!annotator) { navigate("/smile-login?next=/recall-annotate", { replace: true }); return; }
    (async () => {
      try {
        const [nextRes, annRes] = await Promise.all([
          fetch(`${API}/recall-tasks/next-incomplete?annotator=${encodeURIComponent(annotator)}`),
          fetch(`${API}/recall-annotations/${encodeURIComponent(annotator)}`),
        ]);
        const nextData = await nextRes.json();
        const annData = await annRes.json();
        setAnnotations(annData);
        setTaskNum(nextData.task_number ?? 1);
      } catch { setTaskNum(1); }
    })();
  }, [annotator, navigate]);

  // Load task data when taskNum changes
  useEffect(() => {
    if (taskNum === null) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const res = await fetch(`${API}/recall-tasks/${taskNum}`);
        if (!res.ok) throw new Error();
        const t: RecallTask = await res.json();
        if (cancelled) return;
        // Set task and clear loading in the same synchronous block so the video
        // element is in the DOM when the event-listener effect runs.
        setTask(t);
        setJumpVal(String(taskNum));
        setLoading(false);
        // Fetch transcript asynchronously after the video is already showing.
        try {
          const tRes = await fetch(`${API}/videos/${t.video_id}/transcript`);
          if (!cancelled) {
            if (tRes.ok) setUtterances(await tRes.json());
            else setUtterances([]);
          }
        } catch { if (!cancelled) setUtterances([]); }
      } catch {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [taskNum]);

  const playStart = useMemo(() => task ? Math.max(0, task.segment_start - BEFORE_SEC) : 0, [task]);
  const playEnd = useMemo(() => task ? task.segment_end + AFTER_SEC : 0, [task]);
  const segStart = task?.segment_start ?? 0;
  const segEnd = task?.segment_end ?? 0;
  const isInSegment = currentTime >= segStart && currentTime <= segEnd;

  // Video event listeners
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !task) return;
    const onTimeUpdate = () => {
      setCurrentTime(v.currentTime);
      if (v.currentTime >= playEnd) v.currentTime = playStart;
    };
    const onDurationChange = () => setDuration(v.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onRateChange = () => setSpeedState(v.playbackRate);
    const onLoadedData = () => { v.currentTime = playStart; v.play().catch(() => {}); };
    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("durationchange", onDurationChange);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ratechange", onRateChange);
    v.addEventListener("loadeddata", onLoadedData);
    return () => {
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("durationchange", onDurationChange);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ratechange", onRateChange);
      v.removeEventListener("loadeddata", onLoadedData);
    };
  }, [task, playStart, playEnd]);

  useEffect(() => {
    const v = videoRef.current;
    if (v && task?.video_downloaded) { v.currentTime = playStart; v.load(); }
  }, [task?.video_id, task?.task_number, playStart]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current; if (!v) return;
    if (v.paused) v.play(); else v.pause();
  }, []);

  const seek = useCallback((t: number) => {
    const v = videoRef.current; if (!v) return;
    v.currentTime = Math.max(playStart, Math.min(t, playEnd));
  }, [playStart, playEnd]);

  const setSpeed = useCallback((rate: number) => {
    const v = videoRef.current; if (v) v.playbackRate = rate;
  }, []);

  const goToTask = useCallback((num: number) => {
    if (!task || num < 1 || num > task.total_tasks) return;
    setTaskNum(num);
  }, [task]);

  const handleLabel = useCallback(async (label: string) => {
    if (!annotator || !task || saving) return;
    setSaving(true);
    try {
      await fetch(`${API}/recall-annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotator, task_number: task.task_number, label }),
      });
      setAnnotations((prev) => {
        const a = prev ? { ...prev } : { annotator, annotations: {} };
        a.annotations = {
          ...a.annotations,
          [String(task.task_number)]: { label, timestamp: new Date().toISOString() },
        };
        return a;
      });
      if (task.task_number < task.total_tasks) goToTask(task.task_number + 1);
    } catch { /* ignore */ }
    setSaving(false);
  }, [annotator, task, saving, goToTask]);

  const handleJump = useCallback(() => {
    const n = parseInt(jumpVal, 10);
    if (!isNaN(n)) goToTask(n);
  }, [jumpVal, goToTask]);

  const handleLogout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  }, [navigate]);

  const currentLabel = annotations?.annotations[String(taskNum)]?.label ?? null;

  const seekBarSegLeft = useMemo(() => {
    if (!task || playEnd <= playStart) return 0;
    return ((segStart - playStart) / (playEnd - playStart)) * 100;
  }, [task, segStart, playStart, playEnd]);
  const seekBarSegWidth = useMemo(() => {
    if (!task || playEnd <= playStart) return 0;
    return ((segEnd - segStart) / (playEnd - playStart)) * 100;
  }, [task, segStart, segEnd, playStart, playEnd]);

  if (!annotator) return null;
  if (loading || !task) return <div style={st.page}><div style={st.loading}>Loading task...</div></div>;

  return (
    <div style={st.page}>
      {/* Top bar */}
      <div style={st.topBar}>
        <span style={st.taskLabel}>Recall Task {task.task_number} / {task.available_tasks}</span>
        <span style={{ fontSize: "0.7rem", color: "#64748b" }}>({task.total_tasks} total)</span>
        <span style={st.annotator}>{annotator}</span>
        <button style={st.navBtn} onClick={() => goToTask(task.task_number - 1)} disabled={task.task_number <= 1}>Prev</button>
        <button style={st.navBtn} onClick={() => goToTask(task.task_number + 1)} disabled={task.task_number >= task.total_tasks}>Next</button>
        <input style={st.jumpInput} type="text" value={jumpVal}
          onChange={(e) => setJumpVal(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleJump()} onBlur={handleJump} />
        <button style={st.navBtn} onClick={handleJump}>Go</button>
        {currentLabel && (
          <span style={{
            fontSize: "0.8rem", fontWeight: 600, padding: "2px 8px",
            backgroundColor: "#0f172a", borderRadius: "5px",
            color: currentLabel === "smile" ? "#22c55e" : "#64748b",
          }}>
            {currentLabel === "smile" ? "Smile" : "Not a Smile"}
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "#fbbf24" }}>
          Is there a smile in the <strong>highlighted orange</strong> region?
        </span>
        <Link
          to="/recall-results"
          style={{
            ...st.logoutBtn,
            color: "#93c5fd",
            textDecoration: "none",
            display: "inline-flex",
            alignItems: "center",
          }}
        >
          View Results
        </Link>
        <button style={st.logoutBtn} onClick={handleLogout}>Logout</button>
      </div>

      <div style={st.mainLayout}>
        <div style={st.leftPanel}>
          {task.video_downloaded ? (
            <>
              <div style={{
                borderRadius: "10px", padding: "3px",
                backgroundColor: isInSegment ? "#f59e0b" : "transparent",
                transition: "background-color 0.2s",
              }}>
                <video ref={videoRef} src={`${API}/videos/${task.video_id}/stream`}
                  preload="auto" playsInline style={st.video} />
              </div>
              {/* Seek bar */}
              <div style={{
                position: "relative", height: "14px", marginTop: "4px",
                backgroundColor: "#1e293b", borderRadius: "3px", cursor: "pointer", overflow: "hidden",
              }} onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                seek(playStart + ((e.clientX - rect.left) / rect.width) * (playEnd - playStart));
              }}>
                <div style={{
                  position: "absolute", left: `${seekBarSegLeft}%`, width: `${seekBarSegWidth}%`,
                  top: 0, bottom: 0, backgroundColor: "#f59e0b55",
                  borderLeft: "2px solid #f59e0b", borderRight: "2px solid #f59e0b",
                }} />
                <div style={{
                  position: "absolute",
                  left: `${playEnd > playStart ? Math.max(0, Math.min(100, ((currentTime - playStart) / (playEnd - playStart)) * 100)) : 0}%`,
                  top: 0, bottom: 0, width: "3px", backgroundColor: "#3b82f6", borderRadius: "2px",
                  transition: "left 0.05s linear",
                }} />
              </div>
              {/* Controls */}
              <div style={st.controls}>
                <button style={st.playBtn} onClick={togglePlay}>{playing ? "\u23F8" : "\u25B6"}</button>
                <div style={{ display: "flex", gap: "2px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.65rem", color: "#64748b" }}>Spd</span>
                  {SPEEDS.map((r) => (
                    <button key={r} style={{ ...st.optBtn, ...(speed === r ? st.optActive : {}) }}
                      onClick={() => setSpeed(r)}>{r}x</button>
                  ))}
                </div>
                <span style={st.timeDisplay}>{fmtTime(currentTime)}/{fmtTime(duration)}</span>
                <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                  Segment {fmtTime(task.segment_start)}-{fmtTime(task.segment_end)}
                </span>
              </div>
            </>
          ) : (
            <div style={st.notDownloaded}>Video not yet downloaded</div>
          )}

          {/* Binary label buttons */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr",
            gap: "10px", marginTop: "10px",
          }}>
            {[
              { key: "smile", display: "Smile", color: "#22c55e", desc: "There is a visible smile in the highlighted segment." },
              { key: "not_a_smile", display: "Not a Smile", color: "#64748b", desc: "No smile is present — the detection was a false positive." },
            ].map((l) => {
              const isCurrent = currentLabel === l.key;
              return (
                <div key={l.key} style={{
                  display: "flex", flexDirection: "column", borderRadius: "10px", overflow: "hidden",
                  border: isCurrent ? `2px solid ${l.color}` : "2px solid #334155",
                  transition: "border-color 0.15s",
                }}>
                  <button style={{
                    padding: "12px 6px", fontSize: "1rem", fontWeight: 700,
                    border: "none", cursor: "pointer", color: "#0f172a",
                    backgroundColor: l.color, opacity: saving ? 0.6 : 1, textAlign: "center",
                  }} onClick={() => handleLabel(l.key)} disabled={saving}>
                    {l.display}
                  </button>
                  <div style={{
                    padding: "6px 10px", fontSize: "0.78rem", lineHeight: 1.5,
                    color: "#94a3b8", backgroundColor: "#1e293b", flex: 1, textAlign: "center",
                  }}>
                    {l.desc}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Transcript panel */}
        <div style={st.rightPanel}>
          <TranscriptTrack
            utterances={utterances}
            currentTimeMs={currentTime * 1000}
            onSeek={(ms) => seek(ms / 1000)}
            smileStartMs={task.segment_start * 1000}
            smileEndMs={task.segment_end * 1000}
            maxHeight="calc(100vh - 60px)"
            initialScrollToMs={task.segment_start * 1000}
          />
        </div>
      </div>
    </div>
  );
}
