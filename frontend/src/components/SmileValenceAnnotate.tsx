import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useNavigate, Link } from "react-router-dom";
import TranscriptTrack from "./TranscriptTrack";
import type { Utterance } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const CONTEXT_BEFORE_KEY = "smile_valence_context_before";
const CONTEXT_AFTER_KEY = "smile_valence_context_after";
const API = "/api";

const BEFORE_OPTIONS = [3, 5, 10, 15, 20, 30];
const AFTER_OPTIONS = [2, 5, 10, 15, 20, 30];
const SPEEDS = [0.5, 1, 1.5, 2];

type Valence = "negative" | "neutral" | "positive";
const VALENCE_OPTIONS: { value: Valence; label: string; color: string; activeColor: string }[] = [
  { value: "negative", label: "Negative", color: "#334155", activeColor: "#dc2626" },
  { value: "neutral",  label: "Neutral",  color: "#334155", activeColor: "#6366f1" },
  { value: "positive", label: "Positive", color: "#334155", activeColor: "#16a34a" },
];

function loadContextSeconds(key: string, fallback: number): number {
  try {
    const v = localStorage.getItem(key);
    if (v) return parseFloat(v);
  } catch { /* ignore */ }
  return fallback;
}

function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

interface ValenceTask {
  task_number: number;
  video_id: string;
  smile_start: number;
  smile_end: number;
  score: number | null;
  total_tasks: number;
  available_tasks: number;
  video_downloaded: boolean;
}

interface ValenceAnnotation {
  narrative_valence?: Valence;
  speaker_valence?: Valence;
  not_a_smile?: boolean;
  timestamp: string;
}

interface ValenceAnnotations {
  annotator: string;
  annotations: Record<string, ValenceAnnotation>;
}

interface TaskData {
  task: ValenceTask;
  utterances: Utterance[];
}

async function fetchTaskData(taskNum: number): Promise<TaskData> {
  const taskRes = await fetch(`${API}/smile-valence-tasks/${taskNum}`);
  if (!taskRes.ok) throw new Error(`Task ${taskNum}: ${taskRes.status}`);
  const task: ValenceTask = await taskRes.json();
  let utterances: Utterance[] = [];
  try {
    const tRes = await fetch(`${API}/videos/${task.video_id}/transcript`);
    if (tRes.ok) utterances = await tRes.json();
  } catch { /* transcript optional */ }
  return { task, utterances };
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

export default function SmileValenceAnnotate() {
  const navigate = useNavigate();
  const annotator = localStorage.getItem(STORAGE_KEY);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);

  const [ctxBefore, setCtxBefore] = useState(() => loadContextSeconds(CONTEXT_BEFORE_KEY, 5));
  const [ctxAfter, setCtxAfter] = useState(() => loadContextSeconds(CONTEXT_AFTER_KEY, 5));

  const [taskNum, setTaskNum] = useState<number | null>(null);
  const [taskData, setTaskData] = useState<TaskData | null>(null);
  const [annotations, setAnnotations] = useState<ValenceAnnotations | null>(null);
  const [jumpVal, setJumpVal] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Two-step selection state; reset on task change
  const [narrativeValence, setNarrativeValence] = useState<Valence | null>(null);
  const [speakerValence, setSpeakerValence] = useState<Valence | null>(null);

  const preloadRef = useRef<Map<number, Promise<TaskData>>>(new Map());

  useEffect(() => { localStorage.setItem(CONTEXT_BEFORE_KEY, String(ctxBefore)); }, [ctxBefore]);
  useEffect(() => { localStorage.setItem(CONTEXT_AFTER_KEY, String(ctxAfter)); }, [ctxAfter]);

  useEffect(() => {
    if (!annotator) {
      navigate("/smile-login?next=/smile-valence-annotate", { replace: true });
      return;
    }
    (async () => {
      try {
        const [nextRes, annRes] = await Promise.all([
          fetch(`${API}/smile-valence-tasks/next-incomplete?annotator=${encodeURIComponent(annotator)}`),
          fetch(`${API}/smile-valence-annotations/${encodeURIComponent(annotator)}`),
        ]);
        const nextData = await nextRes.json();
        const annData = await annRes.json();
        setAnnotations(annData);
        setTaskNum(nextData.task_number ?? 1);
      } catch {
        setTaskNum(1);
      }
    })();
  }, [annotator, navigate]);

  const preloadTask = useCallback((num: number) => {
    const cache = preloadRef.current;
    if (!cache.has(num)) {
      cache.set(num, fetchTaskData(num).catch(() => null as unknown as TaskData));
    }
    return cache.get(num)!;
  }, []);

  useEffect(() => {
    if (taskNum === null) return;
    let cancelled = false;
    setLoading(true);
    setNarrativeValence(null);
    setSpeakerValence(null);
    (async () => {
      try {
        const data = await preloadTask(taskNum);
        if (cancelled || !data) return;
        setTaskData(data);
        setJumpVal(String(taskNum));
        const total = data.task.total_tasks;
        for (let i = 1; i <= 5; i++) {
          if (taskNum + i <= total) preloadTask(taskNum + i);
        }
        if (taskNum > 1) preloadTask(taskNum - 1);
      } catch { /* handled */ }
      if (!cancelled) setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [taskNum, preloadTask]);

  const playStart = useMemo(() => {
    if (!taskData) return 0;
    return Math.max(0, taskData.task.smile_start - ctxBefore);
  }, [taskData, ctxBefore]);

  const playEnd = useMemo(() => {
    if (!taskData) return 0;
    return taskData.task.smile_end + ctxAfter;
  }, [taskData, ctxAfter]);

  const smileStart = taskData?.task.smile_start ?? 0;
  const smileEnd = taskData?.task.smile_end ?? 0;
  const isInSmile = currentTime >= smileStart && currentTime <= smileEnd;

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !taskData) return;
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
  }, [taskData, playStart, playEnd]);

  useEffect(() => {
    const v = videoRef.current;
    if (v && taskData?.task.video_downloaded) { v.currentTime = playStart; v.load(); }
  }, [taskData?.task.video_id, taskData?.task.task_number, playStart]);

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
    if (!taskData || num < 1 || num > taskData.task.total_tasks) return;
    preloadRef.current.delete(num);
    setTaskNum(num);
  }, [taskData]);

  const submitAnnotation = useCallback(async (payload: {
    narrative_valence?: Valence;
    speaker_valence?: Valence;
    not_a_smile?: boolean;
  }) => {
    if (!annotator || !taskData || saving) return;
    setSaving(true);
    try {
      await fetch(`${API}/smile-valence-annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotator, task_number: taskData.task.task_number, ...payload }),
      });
      setAnnotations((prev) => {
        const a = prev ? { ...prev } : { annotator, annotations: {} };
        a.annotations = {
          ...a.annotations,
          [String(taskData.task.task_number)]: {
            ...payload,
            timestamp: new Date().toISOString(),
          },
        };
        return a;
      });
      if (taskData.task.task_number < taskData.task.total_tasks) {
        goToTask(taskData.task.task_number + 1);
      }
    } catch { /* ignore */ }
    setSaving(false);
  }, [annotator, taskData, saving, goToTask]);

  const handleSelectSpeakerValence = useCallback((val: Valence) => {
    if (!narrativeValence) return;
    setSpeakerValence(val);
    submitAnnotation({ narrative_valence: narrativeValence, speaker_valence: val });
  }, [narrativeValence, submitAnnotation]);

  const handleNotASmile = useCallback(() => {
    submitAnnotation({ not_a_smile: true });
  }, [submitAnnotation]);

  const handleJump = useCallback(() => {
    const n = parseInt(jumpVal, 10);
    if (!isNaN(n)) goToTask(n);
  }, [jumpVal, goToTask]);

  const handleLogout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  }, [navigate]);

  const seekBarSmileLeft = useMemo(() => {
    if (!taskData || playEnd <= playStart) return 0;
    return ((smileStart - playStart) / (playEnd - playStart)) * 100;
  }, [taskData, smileStart, playStart, playEnd]);

  const seekBarSmileWidth = useMemo(() => {
    if (!taskData || playEnd <= playStart) return 0;
    return ((smileEnd - smileStart) / (playEnd - playStart)) * 100;
  }, [taskData, smileStart, smileEnd, playStart, playEnd]);

  const currentAnnotation = annotations?.annotations[String(taskNum)];

  if (!annotator) return null;
  if (loading || !taskData) return <div style={st.page}><div style={st.loading}>Loading task...</div></div>;

  const { task } = taskData;

  return (
    <div style={st.page}>
      {/* Top bar */}
      <div style={st.topBar}>
        <span style={st.taskLabel}>Task {task.task_number} / {task.available_tasks}</span>
        <span style={{ fontSize: "0.7rem", color: "#64748b" }}>({task.total_tasks} total)</span>
        <span style={st.annotator}>{annotator}</span>

        <button style={st.navBtn} onClick={() => goToTask(task.task_number - 1)} disabled={task.task_number <= 1}>Prev</button>
        <button style={st.navBtn} onClick={() => goToTask(task.task_number + 1)} disabled={task.task_number >= task.total_tasks}>Next</button>
        <input style={st.jumpInput} type="text" value={jumpVal}
          onChange={(e) => setJumpVal(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleJump()} onBlur={handleJump} />
        <button style={st.navBtn} onClick={handleJump}>Go</button>

        {currentAnnotation && (
          <span style={{
            fontSize: "0.8rem", fontWeight: 600, padding: "2px 8px",
            backgroundColor: "#0f172a", borderRadius: "5px",
            color: currentAnnotation.not_a_smile ? "#64748b" : "#22c55e",
          }}>
            {currentAnnotation.not_a_smile ? "Not a smile" : "Answered"}
          </span>
        )}

        <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "#fbbf24" }}>
          Rate the valence during the <strong>highlighted orange</strong> smile region
        </span>
        <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
          {(() => {
            const [subject, tape] = task.video_id.split(".");
            return tape
              ? <span>Subject <strong style={{ color: "#94a3b8" }}>{subject}</strong> · Tape <strong style={{ color: "#94a3b8" }}>{tape}</strong></span>
              : task.video_id;
          })()}
        </span>
        <Link to="/smile-valence-results" style={{ ...st.logoutBtn, color: "#93c5fd", textDecoration: "none" }}>
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
                backgroundColor: isInSmile ? "#f59e0b" : "transparent",
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
                const pct = (e.clientX - rect.left) / rect.width;
                seek(playStart + pct * (playEnd - playStart));
              }}>
                <div style={{
                  position: "absolute", left: `${seekBarSmileLeft}%`, width: `${seekBarSmileWidth}%`,
                  top: 0, bottom: 0, backgroundColor: "#f59e0b55",
                  borderLeft: "2px solid #f59e0b", borderRight: "2px solid #f59e0b",
                }} />
                <div style={{
                  position: "absolute",
                  left: `${playEnd > playStart ? ((currentTime - playStart) / (playEnd - playStart)) * 100 : 0}%`,
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
                    <button key={r} style={{ ...st.optBtn, ...(speed === r ? st.optActive : {}) }} onClick={() => setSpeed(r)}>
                      {r}x
                    </button>
                  ))}
                </div>
                <span style={{ color: "#334155", fontSize: "0.7rem" }}>|</span>
                <div style={{ display: "flex", gap: "2px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.65rem", color: "#64748b" }}>Before</span>
                  {BEFORE_OPTIONS.map((v) => (
                    <button key={v} style={{ ...st.optBtn, ...(ctxBefore === v ? st.optActive : {}) }} onClick={() => setCtxBefore(v)}>
                      {v}s
                    </button>
                  ))}
                </div>
                <span style={{ color: "#334155", fontSize: "0.7rem" }}>|</span>
                <div style={{ display: "flex", gap: "2px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.65rem", color: "#64748b" }}>After</span>
                  {AFTER_OPTIONS.map((v) => (
                    <button key={v} style={{ ...st.optBtn, ...(ctxAfter === v ? st.optActive : {}) }} onClick={() => setCtxAfter(v)}>
                      {v}s
                    </button>
                  ))}
                </div>
                <span style={st.timeDisplay}>{fmtTime(currentTime)}/{fmtTime(duration)}</span>
                <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                  Smile {fmtTime(task.smile_start)}-{fmtTime(task.smile_end)}
                </span>
              </div>
            </>
          ) : (
            <div style={st.notDownloaded}>Video not yet downloaded</div>
          )}

          {/* Annotation panel */}
          <div style={{ marginTop: "12px", display: "flex", flexDirection: "column", gap: "14px" }}>

            {/* Step 1: Narrative Valence */}
            <ValenceSelector
              label="Narrative Valence"
              description="What is the emotional valence of the story or content being discussed at this moment?"
              value={currentAnnotation?.narrative_valence ?? narrativeValence}
              locked={!!currentAnnotation || saving}
              onChange={setNarrativeValence}
            />

            {/* Step 2: Speaker Valence — only enabled after narrative is chosen */}
            <ValenceSelector
              label="Speaker's Current Valence"
              description="What is the speaker's personal emotional state during this moment?"
              value={currentAnnotation?.speaker_valence ?? speakerValence}
              locked={!!currentAnnotation || saving}
              disabled={!narrativeValence && !currentAnnotation}
              onChange={handleSelectSpeakerValence}
            />

            {/* Not a smile escape hatch */}
            {!currentAnnotation && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button
                  disabled={saving}
                  onClick={handleNotASmile}
                  style={{
                    padding: "6px 16px", fontSize: "0.78rem", fontWeight: 600,
                    border: "2px solid #475569", borderRadius: "6px", cursor: saving ? "default" : "pointer",
                    backgroundColor: "#1e293b", color: "#94a3b8",
                    opacity: saving ? 0.6 : 1,
                  }}
                >
                  Not a Smile
                </button>
              </div>
            )}
          </div>

          {/* Show existing annotation */}
          {currentAnnotation && (
            <div style={{
              marginTop: "8px", padding: "8px 12px",
              backgroundColor: "#1e293b", borderRadius: "8px",
              border: "1px solid #334155", fontSize: "0.8rem",
              display: "flex", gap: "20px",
            }}>
              <span style={{ color: "#64748b", fontWeight: 600 }}>Previous: </span>
              {currentAnnotation.not_a_smile ? (
                <span style={{ color: "#94a3b8", fontStyle: "italic" }}>Not a smile</span>
              ) : (
                <>
                  <span>
                    <span style={{ color: "#94a3b8" }}>Narrative: </span>
                    <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{currentAnnotation.narrative_valence}</span>
                  </span>
                  <span>
                    <span style={{ color: "#94a3b8" }}>Speaker: </span>
                    <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{currentAnnotation.speaker_valence}</span>
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Transcript panel */}
        <div style={st.rightPanel}>
          <TranscriptTrack
            utterances={taskData.utterances}
            currentTimeMs={currentTime * 1000}
            onSeek={(ms) => seek(ms / 1000)}
            smileStartMs={task.smile_start * 1000}
            smileEndMs={task.smile_end * 1000}
            maxHeight="calc(100vh - 60px)"
            initialScrollToMs={task.smile_start * 1000}
          />
        </div>
      </div>
    </div>
  );
}

// ── Sub-component ─────────────────────────────────────────────────────────────

interface ValenceSelectorProps {
  label: string;
  description: string;
  value: Valence | null | undefined;
  locked: boolean;
  disabled?: boolean;
  onChange: (v: Valence) => void;
}

function ValenceSelector({ label, description, value, locked, disabled = false, onChange }: ValenceSelectorProps) {
  return (
    <div style={{
      padding: "10px 14px", backgroundColor: "#1e293b", borderRadius: "8px",
      border: `1px solid ${disabled ? "#1e293b" : "#334155"}`,
      opacity: disabled ? 0.4 : 1,
      transition: "opacity 0.2s",
    }}>
      <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#f8fafc", marginBottom: "3px" }}>
        {label}
      </div>
      <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: "8px" }}>
        {description}
      </div>
      <div style={{ display: "flex", gap: "8px" }}>
        {VALENCE_OPTIONS.map((opt) => {
          const isSelected = value === opt.value;
          return (
            <button
              key={opt.value}
              disabled={disabled || locked}
              onClick={() => onChange(opt.value)}
              style={{
                flex: 1, padding: "8px 0", fontSize: "0.85rem", fontWeight: 600,
                border: `2px solid ${isSelected ? opt.activeColor : "#475569"}`,
                borderRadius: "6px", cursor: disabled || locked ? "default" : "pointer",
                backgroundColor: isSelected ? opt.activeColor : "#0f172a",
                color: isSelected ? "#fff" : "#94a3b8",
                transition: "background-color 0.15s, border-color 0.15s, color 0.15s",
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
