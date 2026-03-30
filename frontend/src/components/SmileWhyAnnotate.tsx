import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useNavigate, Link } from "react-router-dom";
import TranscriptTrack from "./TranscriptTrack";
import type { Utterance } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const CONTEXT_BEFORE_KEY = "smile_why_context_before";
const CONTEXT_AFTER_KEY = "smile_why_context_after";
const API = "/api";

const BEFORE_OPTIONS = [3, 5, 10, 15, 20, 30];
const AFTER_OPTIONS = [2, 5, 10, 15, 20, 30];
const SPEEDS = [0.5, 1, 1.5, 2];

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

interface WhyTask {
  task_number: number;
  video_id: string;
  smile_start: number;
  smile_end: number;
  score: number | null;
  stratum: string;
  score_tier: string;
  total_tasks: number;
  available_tasks: number;
  video_downloaded: boolean;
}

interface WhyAnnotations {
  annotator: string;
  annotations: Record<string, { response?: string; not_a_smile?: boolean; timestamp: string }>;
}

interface TaskData {
  task: WhyTask;
  utterances: Utterance[];
}

async function fetchTaskData(taskNum: number): Promise<TaskData> {
  const taskRes = await fetch(`${API}/smile-why-tasks/${taskNum}`);
  if (!taskRes.ok) throw new Error(`Task ${taskNum}: ${taskRes.status}`);
  const task: WhyTask = await taskRes.json();

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

export default function SmileWhyAnnotate() {
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
  const [annotations, setAnnotations] = useState<WhyAnnotations | null>(null);
  const [jumpVal, setJumpVal] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [responseText, setResponseText] = useState("");

  const preloadRef = useRef<Map<number, Promise<TaskData>>>(new Map());

  useEffect(() => { localStorage.setItem(CONTEXT_BEFORE_KEY, String(ctxBefore)); }, [ctxBefore]);
  useEffect(() => { localStorage.setItem(CONTEXT_AFTER_KEY, String(ctxAfter)); }, [ctxAfter]);

  useEffect(() => {
    if (!annotator) {
      navigate("/smile-login?next=/smile-why-annotate", { replace: true });
      return;
    }
    (async () => {
      try {
        const [nextRes, annRes] = await Promise.all([
          fetch(`${API}/smile-why-tasks/next-incomplete?annotator=${encodeURIComponent(annotator)}`),
          fetch(`${API}/smile-why-annotations/${encodeURIComponent(annotator)}`),
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

  const currentAnnotation = annotations?.annotations[String(taskNum)];

  useEffect(() => {
    setResponseText(currentAnnotation?.response ?? "");
  }, [taskNum]);

  const handleSubmit = useCallback(async (notASmile: boolean) => {
    if (!annotator || !taskData || saving) return;
    if (!notASmile && !responseText.trim()) return;

    setSaving(true);
    try {
      await fetch(`${API}/smile-why-annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotator,
          task_number: taskData.task.task_number,
          response: notASmile ? "" : responseText.trim(),
          not_a_smile: notASmile,
        }),
      });
      setAnnotations((prev) => {
        const a = prev ? { ...prev } : { annotator, annotations: {} };
        a.annotations = {
          ...a.annotations,
          [String(taskData.task.task_number)]: {
            response: notASmile ? undefined : responseText.trim(),
            not_a_smile: notASmile || undefined,
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
  }, [annotator, taskData, saving, responseText, goToTask]);

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
          Why did the subject smile in the <strong>highlighted orange</strong> region?
        </span>
        <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
          {(() => {
            const [subject, tape] = task.video_id.split(".");
            return tape
              ? <span>Subject <strong style={{ color: "#94a3b8" }}>{subject}</strong> · Tape <strong style={{ color: "#94a3b8" }}>{tape}</strong></span>
              : task.video_id;
          })()}
        </span>
        <Link to="/smile-why-results" style={{ ...st.logoutBtn, color: "#93c5fd", textDecoration: "none" }}>
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

          {/* Response area */}
          <div style={{ marginTop: "10px" }}>
            <div style={{
              fontSize: "0.9rem", fontWeight: 600, color: "#e2e8f0", marginBottom: "4px",
            }}>
              Why did the subject smile here?
            </div>
            <div style={{
              fontSize: "0.78rem", color: "#94a3b8", marginBottom: "6px", lineHeight: 1.5,
            }}>
              Pay attention to the subject matter, emotions, and whether the subject is directly
              engaging with the interviewer or if the smile occurred for a story-telling or narrative purpose.
            </div>
            <div style={{
              fontSize: "0.72rem", color: "#64748b", marginBottom: "8px", lineHeight: 1.6,
              padding: "6px 10px", backgroundColor: "#1e293b", borderRadius: "6px",
              border: "1px solid #334155",
            }}>
              <div style={{ fontWeight: 600, color: "#94a3b8", marginBottom: "2px" }}>Examples:</div>
              <div>&bull; "The subject smiled because he was telling a joke to the interviewer."</div>
              <div>&bull; "The subject smiled because she was fondly remembering meeting her husband."</div>
              <div>&bull; "The subject smiled because he was emphasizing a moment of surprise during his story."</div>
            </div>
            <textarea
              value={responseText}
              onChange={(e) => setResponseText(e.target.value)}
              placeholder="The subject smiled because..."
              rows={3}
              style={{
                width: "100%", padding: "10px 12px", fontSize: "0.85rem",
                border: "1px solid #475569", borderRadius: "8px",
                backgroundColor: "#1e293b", color: "#e2e8f0",
                resize: "vertical", outline: "none", fontFamily: "inherit",
                boxSizing: "border-box",
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  handleSubmit(false);
                }
              }}
            />
            <div style={{ display: "flex", gap: "10px", marginTop: "8px", alignItems: "center" }}>
              <button
                style={{
                  padding: "10px 24px", fontSize: "0.9rem", fontWeight: 700,
                  border: "none", borderRadius: "8px", cursor: "pointer",
                  backgroundColor: responseText.trim() ? "#3b82f6" : "#334155",
                  color: responseText.trim() ? "#fff" : "#64748b",
                  opacity: saving ? 0.6 : 1,
                  transition: "background-color 0.15s",
                }}
                onClick={() => handleSubmit(false)}
                disabled={saving || !responseText.trim()}
              >
                Submit
              </button>
              <span style={{ fontSize: "0.7rem", color: "#64748b" }}>Cmd+Enter to submit</span>
              <div style={{ marginLeft: "auto" }}>
                <button
                  style={{
                    padding: "8px 16px", fontSize: "0.8rem", fontWeight: 600,
                    border: "2px solid #475569", borderRadius: "8px", cursor: "pointer",
                    backgroundColor: "#1e293b", color: "#94a3b8",
                    opacity: saving ? 0.6 : 1,
                  }}
                  onClick={() => handleSubmit(true)}
                  disabled={saving}
                >
                  The subject was not smiling
                </button>
              </div>
            </div>
          </div>

          {/* Show existing annotation */}
          {currentAnnotation && (
            <div style={{
              marginTop: "8px", padding: "8px 12px",
              backgroundColor: "#1e293b", borderRadius: "8px",
              border: "1px solid #334155", fontSize: "0.8rem",
            }}>
              <span style={{ color: "#64748b", fontWeight: 600 }}>Your previous answer: </span>
              {currentAnnotation.not_a_smile ? (
                <span style={{ color: "#94a3b8", fontStyle: "italic" }}>Marked as not a smile</span>
              ) : (
                <span style={{ color: "#e2e8f0" }}>{currentAnnotation.response}</span>
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
