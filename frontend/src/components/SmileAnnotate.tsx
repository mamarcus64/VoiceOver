import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import TranscriptTrack from "./TranscriptTrack";
import type {
  SmileTask,
  SmileAnnotations,
  SmileConfigData,
  Utterance,
} from "../types";
import { SMILE_LABELS } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const CONTEXT_BEFORE_KEY = "smile_context_before";
const CONTEXT_AFTER_KEY = "smile_context_after";
const API = "/api";

const BEFORE_OPTIONS = [3, 5, 10, 15, 20];
const AFTER_OPTIONS = [2, 5, 10, 15];

function loadContextSeconds(key: string, fallback: number): number {
  try {
    const v = localStorage.getItem(key);
    if (v) return parseFloat(v);
  } catch { /* ignore */ }
  return fallback;
}

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "8px 16px",
    maxWidth: "1400px",
    margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  topBar: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "7px 14px",
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    marginBottom: "8px",
    flexWrap: "wrap" as const,
  },
  taskLabel: {
    fontSize: "1.1rem",
    fontWeight: 700,
    color: "#f8fafc",
  },
  annotator: {
    fontSize: "0.8rem",
    color: "#94a3b8",
  },
  navBtn: {
    padding: "4px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    border: "1px solid #475569",
    borderRadius: "5px",
    cursor: "pointer",
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
  jumpInput: {
    width: "60px",
    padding: "4px 6px",
    fontSize: "0.8rem",
    border: "1px solid #475569",
    borderRadius: "5px",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    textAlign: "center" as const,
  },
  logoutBtn: {
    padding: "4px 10px",
    fontSize: "0.75rem",
    border: "1px solid #475569",
    borderRadius: "5px",
    cursor: "pointer",
    backgroundColor: "#334155",
    color: "#94a3b8",
  },
  mainLayout: {
    display: "flex",
    gap: "12px",
    alignItems: "flex-start",
  },
  leftPanel: {
    flex: "1 1 60%",
    minWidth: 0,
    display: "flex",
    flexDirection: "column" as const,
  },
  rightPanel: {
    flex: "0 0 380px",
    display: "flex",
    flexDirection: "column" as const,
  },
  video: {
    width: "100%",
    borderRadius: "8px",
    backgroundColor: "#000",
    maxHeight: "45vh",
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "4px 8px",
    backgroundColor: "#1e293b",
    borderRadius: "6px",
    marginTop: "4px",
    flexWrap: "wrap" as const,
  },
  playBtn: {
    padding: "5px 12px",
    backgroundColor: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "0.8rem",
  },
  optBtn: {
    padding: "3px 6px",
    border: "1px solid #475569",
    borderRadius: "3px",
    cursor: "pointer",
    fontSize: "0.7rem",
    fontWeight: 500,
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
  optActive: {
    backgroundColor: "#3b82f6",
    borderColor: "#3b82f6",
    color: "#fff",
  },
  timeDisplay: {
    color: "#94a3b8",
    fontSize: "0.8rem",
    fontVariantNumeric: "tabular-nums",
  },
  notDownloaded: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "200px",
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    color: "#f59e0b",
    fontSize: "1rem",
    fontWeight: 600,
  },
  loading: {
    textAlign: "center" as const,
    padding: "48px",
    color: "#64748b",
    fontSize: "1.1rem",
  },
};

const SPEEDS = [0.5, 1, 1.5, 2];

function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

interface TaskData {
  task: SmileTask;
  utterances: Utterance[];
  config: SmileConfigData;
}

async function fetchTaskData(taskNum: number): Promise<TaskData> {
  const [taskRes, cfgRes] = await Promise.all([
    fetch(`${API}/smile-tasks/${taskNum}`),
    fetch(`${API}/smile-config`),
  ]);
  if (!taskRes.ok) throw new Error(`Task ${taskNum}: ${taskRes.status}`);
  const task: SmileTask = await taskRes.json();
  const config: SmileConfigData = await cfgRes.json();

  let utterances: Utterance[] = [];
  try {
    const tRes = await fetch(`${API}/videos/${task.video_id}/transcript`);
    if (tRes.ok) utterances = await tRes.json();
  } catch { /* transcript optional */ }

  return { task, utterances, config };
}

export default function SmileAnnotate() {
  const navigate = useNavigate();
  const annotator = localStorage.getItem(STORAGE_KEY);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);

  const [ctxBefore, setCtxBefore] = useState(() => loadContextSeconds(CONTEXT_BEFORE_KEY, 10));
  const [ctxAfter, setCtxAfter] = useState(() => loadContextSeconds(CONTEXT_AFTER_KEY, 5));

  const [taskNum, setTaskNum] = useState<number | null>(null);
  const [taskData, setTaskData] = useState<TaskData | null>(null);
  const [annotations, setAnnotations] = useState<SmileAnnotations | null>(null);
  const [jumpVal, setJumpVal] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [notes, setNotes] = useState("");
  const [twoLabelMode, setTwoLabelMode] = useState(false);
  const [pendingPrimary, setPendingPrimary] = useState<string | null>(null);

  const preloadRef = useRef<Map<number, Promise<TaskData>>>(new Map());

  useEffect(() => { localStorage.setItem(CONTEXT_BEFORE_KEY, String(ctxBefore)); }, [ctxBefore]);
  useEffect(() => { localStorage.setItem(CONTEXT_AFTER_KEY, String(ctxAfter)); }, [ctxAfter]);

  useEffect(() => {
    if (!annotator) {
      navigate("/smile-login", { replace: true });
      return;
    }
    (async () => {
      try {
        const [nextRes, annRes] = await Promise.all([
          fetch(`${API}/smile-tasks/next-incomplete?annotator=${encodeURIComponent(annotator)}`),
          fetch(`${API}/smile-annotations/${encodeURIComponent(annotator)}`),
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
    (async () => {
      try {
        const data = await preloadTask(taskNum);
        if (cancelled || !data) return;
        setTaskData(data);
        setJumpVal(String(taskNum));
        if (data.task.total_tasks > taskNum) {
          preloadTask(taskNum + 1);
        }
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
      if (v.currentTime >= playEnd) {
        v.currentTime = playStart;
      }
    };
    const onDurationChange = () => setDuration(v.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onRateChange = () => setSpeedState(v.playbackRate);
    const onLoadedData = () => {
      v.currentTime = playStart;
      v.play().catch(() => {});
    };

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
    if (v && taskData?.task.video_downloaded) {
      v.currentTime = playStart;
      v.load();
    }
  }, [taskData?.task.video_id, taskData?.task.task_number, playStart]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  }, []);

  const seek = useCallback((t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(playStart, Math.min(t, playEnd));
  }, [playStart, playEnd]);

  const setSpeed = useCallback((rate: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
  }, []);

  const goToTask = useCallback((num: number) => {
    if (!taskData || num < 1 || num > taskData.task.total_tasks) return;
    preloadRef.current.delete(num);
    setPendingPrimary(null);
    setTaskNum(num);
  }, [taskData]);

  const saveAnnotation = useCallback(async (label: string, runnerUp: string) => {
    if (!annotator || !taskData) return;
    await fetch(`${API}/smile-annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        annotator,
        task_number: taskData.task.task_number,
        label,
        notes,
        runner_up: runnerUp,
      }),
    });
    setAnnotations((prev) => {
      const a = prev ? { ...prev } : { annotator, annotations: {} };
      a.annotations = {
        ...a.annotations,
        [String(taskData.task.task_number)]: {
          label,
          timestamp: new Date().toISOString(),
          notes: notes || undefined,
          runner_up: runnerUp || undefined,
        },
      };
      return a;
    });
  }, [annotator, taskData, notes]);

  const cancelPrimary = useCallback(() => setPendingPrimary(null), []);

  const handleLabel = useCallback(async (label: string) => {
    if (!annotator || !taskData || saving) return;

    if (twoLabelMode && !pendingPrimary) {
      setPendingPrimary(label);
      return;
    }

    if (twoLabelMode && pendingPrimary && label === pendingPrimary) return;

    setSaving(true);
    try {
      if (twoLabelMode && pendingPrimary) {
        await saveAnnotation(pendingPrimary, label);
      } else {
        await saveAnnotation(label, "");
      }
      setPendingPrimary(null);
      if (taskData.task.task_number < taskData.task.total_tasks) {
        goToTask(taskData.task.task_number + 1);
      }
    } catch { /* ignore */ }
    setSaving(false);
  }, [annotator, taskData, saving, goToTask, twoLabelMode, pendingPrimary, saveAnnotation]);

  useEffect(() => {
    if (!pendingPrimary) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") cancelPrimary();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pendingPrimary, cancelPrimary]);

  const handleJump = useCallback(() => {
    const n = parseInt(jumpVal, 10);
    if (!isNaN(n)) goToTask(n);
  }, [jumpVal, goToTask]);

  const handleLogout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  }, [navigate]);

  const currentAnnotation = annotations?.annotations[String(taskNum)];
  const currentLabel = currentAnnotation?.label ?? null;
  const currentRunnerUp = currentAnnotation?.runner_up ?? null;

  useEffect(() => {
    setNotes(currentAnnotation?.notes ?? "");
    setPendingPrimary(null);
    setTwoLabelMode(!!currentAnnotation?.runner_up);
  }, [taskNum]);

  const seekBarSmileLeft = useMemo(() => {
    if (!taskData || playEnd <= playStart) return 0;
    return ((smileStart - playStart) / (playEnd - playStart)) * 100;
  }, [taskData, smileStart, playStart, playEnd]);

  const seekBarSmileWidth = useMemo(() => {
    if (!taskData || playEnd <= playStart) return 0;
    return ((smileEnd - smileStart) / (playEnd - playStart)) * 100;
  }, [taskData, smileStart, smileEnd, playStart, playEnd]);

  if (!annotator) return null;

  if (loading || !taskData) {
    return (
      <div style={st.page}>
        <div style={st.loading}>Loading task...</div>
      </div>
    );
  }

  const { task } = taskData;

  const promptText = twoLabelMode
    ? pendingPrimary
      ? `Primary: ${SMILE_LABELS.find((l) => l.key === pendingPrimary)?.display}. Now pick the runner-up.`
      : "Pick the primary label first."
    : null;

  return (
    <div style={st.page}>
      {/* Top bar */}
      <div style={st.topBar}>
        <span style={st.taskLabel}>
          Task {task.task_number} / {task.available_tasks}
        </span>
        <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
          ({task.total_tasks} total)
        </span>
        <span style={st.annotator}>{annotator}</span>

        <button style={st.navBtn} onClick={() => goToTask(task.task_number - 1)} disabled={task.task_number <= 1}>
          Prev
        </button>
        <button style={st.navBtn} onClick={() => goToTask(task.task_number + 1)} disabled={task.task_number >= task.total_tasks}>
          Next
        </button>
        <input
          style={st.jumpInput}
          type="text"
          value={jumpVal}
          onChange={(e) => setJumpVal(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleJump()}
          onBlur={handleJump}
        />
        <button style={st.navBtn} onClick={handleJump}>Go</button>

        {currentLabel && (
          <span style={{
            fontSize: "0.8rem", fontWeight: 600,
            color: SMILE_LABELS.find((l) => l.key === currentLabel)?.color ?? "#94a3b8",
            padding: "2px 8px", backgroundColor: "#0f172a", borderRadius: "5px",
          }}>
            {SMILE_LABELS.find((l) => l.key === currentLabel)?.display ?? currentLabel}
            {currentRunnerUp && (
              <span style={{ color: "#64748b", fontWeight: 400 }}>
                {" / "}
                {SMILE_LABELS.find((l) => l.key === currentRunnerUp)?.display ?? currentRunnerUp}
              </span>
            )}
          </span>
        )}

        <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "#fbbf24" }}>
          Label the <strong>highlighted</strong> smile. Ignore others.
        </span>
        <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
          {task.video_id}
        </span>
        <button style={st.logoutBtn} onClick={handleLogout}>Logout</button>
      </div>

      <div style={st.mainLayout}>
        <div style={st.leftPanel}>
          {task.video_downloaded ? (
            <>
              {/* Video with smile-region border glow */}
              <div style={{
                borderRadius: "10px",
                padding: "3px",
                backgroundColor: isInSmile ? "#f59e0b" : "transparent",
                transition: "background-color 0.2s",
              }}>
                <video
                  ref={videoRef}
                  src={`${API}/videos/${task.video_id}/stream`}
                  preload="auto"
                  playsInline
                  style={st.video}
                />
              </div>

              {/* Custom seek bar */}
              <div style={{
                position: "relative", height: "14px", marginTop: "4px",
                backgroundColor: "#1e293b", borderRadius: "3px", cursor: "pointer", overflow: "hidden",
              }}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const pct = (e.clientX - rect.left) / rect.width;
                  seek(playStart + pct * (playEnd - playStart));
                }}
              >
                <div style={{
                  position: "absolute",
                  left: `${seekBarSmileLeft}%`, width: `${seekBarSmileWidth}%`,
                  top: 0, bottom: 0,
                  backgroundColor: "#f59e0b55",
                  borderLeft: "2px solid #f59e0b", borderRight: "2px solid #f59e0b",
                }} />
                <div style={{
                  position: "absolute",
                  left: `${playEnd > playStart ? ((currentTime - playStart) / (playEnd - playStart)) * 100 : 0}%`,
                  top: 0, bottom: 0, width: "3px",
                  backgroundColor: "#3b82f6", borderRadius: "2px",
                  transition: "left 0.05s linear",
                }} />
              </div>

              {/* Controls */}
              <div style={st.controls}>
                <button style={st.playBtn} onClick={togglePlay}>
                  {playing ? "\u23F8" : "\u25B6"}
                </button>
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

          {/* Two-label mode prompt */}
          {promptText && (
            <div style={{
              marginTop: "6px", padding: "5px 12px",
              backgroundColor: "#1c1917", border: "1px solid #6366f155",
              borderRadius: "6px", fontSize: "0.8rem", color: "#a5b4fc",
              textAlign: "center", display: "flex", alignItems: "center",
              justifyContent: "center", gap: "10px",
            }}>
              <span>{promptText}</span>
              {pendingPrimary && (
                <button
                  onClick={cancelPrimary}
                  style={{
                    padding: "2px 10px", fontSize: "0.75rem", fontWeight: 600,
                    border: "1px solid #475569", borderRadius: "4px",
                    cursor: "pointer", backgroundColor: "#334155", color: "#e2e8f0",
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
          )}

          {/* Label cards: 4-column grid, button + description */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "8px",
            marginTop: "8px",
          }}>
            {SMILE_LABELS.map((l) => {
              const isPrimary = pendingPrimary === l.key;
              const isSaved = currentLabel === l.key;
              const isRunnerUp = currentRunnerUp === l.key;
              const highlighted = isPrimary || isSaved;

              return (
                <div key={l.key} style={{
                  display: "flex",
                  flexDirection: "column",
                  borderRadius: "10px",
                  overflow: "hidden",
                  border: highlighted ? `2px solid ${l.color}` : isRunnerUp ? "2px dashed #64748b" : "2px solid #334155",
                  transition: "border-color 0.15s",
                }}>
                  <button
                    style={{
                      padding: "10px 6px",
                      fontSize: "0.95rem",
                      fontWeight: 700,
                      border: "none",
                      cursor: "pointer",
                      color: "#0f172a",
                      backgroundColor: l.color,
                      opacity: saving ? 0.6 : 1,
                      textAlign: "center",
                    }}
                    onClick={() => handleLabel(l.key)}
                    disabled={saving}
                  >
                    {l.display}
                  </button>
                  <div style={{
                    padding: "6px 10px",
                    fontSize: "0.75rem",
                    lineHeight: 1.5,
                    color: "#94a3b8",
                    backgroundColor: "#1e293b",
                    flex: 1,
                    textAlign: "center",
                  }}>
                    {l.desc}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Notes + two-label toggle row */}
          <div style={{
            display: "flex",
            gap: "8px",
            alignItems: "center",
            marginTop: "8px",
          }}>
            <input
              type="text"
              placeholder="Notes (optional)"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              style={{
                flex: 1,
                padding: "6px 10px",
                fontSize: "0.8rem",
                border: "1px solid #475569",
                borderRadius: "5px",
                backgroundColor: "#1e293b",
                color: "#e2e8f0",
                outline: "none",
              }}
            />
            <button
              onClick={() => {
                setTwoLabelMode((v) => !v);
                setPendingPrimary(null);
              }}
              style={{
                padding: "6px 12px",
                fontSize: "0.75rem",
                fontWeight: 600,
                border: twoLabelMode ? "1px solid #6366f1" : "1px solidrgb(191, 149, 33)",
                borderRadius: "5px",
                cursor: "pointer",
                backgroundColor: twoLabelMode ? "#312e81" : "#334155",
                color: twoLabelMode ? "#a5b4fc" : "#94a3b8",
                whiteSpace: "nowrap" as const,
              }}
            >
              {twoLabelMode ? "This smile is one label" : "This smile could be two labels"}
            </button>
          </div>
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
