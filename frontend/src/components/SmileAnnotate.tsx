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
    padding: "12px 20px",
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
    gap: "12px",
    padding: "10px 16px",
    backgroundColor: "#1e293b",
    borderRadius: "10px",
    marginBottom: "12px",
    flexWrap: "wrap" as const,
  },
  taskLabel: {
    fontSize: "1.2rem",
    fontWeight: 700,
    color: "#f8fafc",
  },
  annotator: {
    fontSize: "0.85rem",
    color: "#94a3b8",
  },
  navBtn: {
    padding: "6px 14px",
    fontSize: "0.85rem",
    fontWeight: 600,
    border: "1px solid #475569",
    borderRadius: "6px",
    cursor: "pointer",
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
  jumpInput: {
    width: "70px",
    padding: "6px 8px",
    fontSize: "0.85rem",
    border: "1px solid #475569",
    borderRadius: "6px",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    textAlign: "center" as const,
  },
  logoutBtn: {
    padding: "6px 12px",
    fontSize: "0.8rem",
    border: "1px solid #475569",
    borderRadius: "6px",
    cursor: "pointer",
    backgroundColor: "#334155",
    color: "#94a3b8",
    marginLeft: "auto",
  },
  mainLayout: {
    display: "flex",
    gap: "16px",
    alignItems: "flex-start",
    marginBottom: "16px",
  },
  leftPanel: {
    flex: "1 1 60%",
    minWidth: 0,
    display: "flex",
    flexDirection: "column" as const,
  },
  rightPanel: {
    flex: "0 0 400px",
    display: "flex",
    flexDirection: "column" as const,
  },
  video: {
    width: "100%",
    borderRadius: "8px",
    backgroundColor: "#000",
    maxHeight: "60vh",
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "6px 10px",
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    marginTop: "6px",
    flexWrap: "wrap" as const,
  },
  playBtn: {
    padding: "8px 16px",
    backgroundColor: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "0.9rem",
  },
  optBtn: {
    padding: "5px 8px",
    border: "1px solid #475569",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "0.75rem",
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
    fontSize: "0.85rem",
    fontVariantNumeric: "tabular-nums",
  },
  labelRow: {
    display: "flex",
    gap: "10px",
    justifyContent: "center",
    marginTop: "16px",
    flexWrap: "wrap" as const,
  },
  labelBtn: {
    flex: "1 1 0",
    padding: "14px 8px",
    fontSize: "1rem",
    fontWeight: 700,
    border: "3px solid transparent",
    borderRadius: "12px",
    cursor: "pointer",
    color: "#fff",
    maxWidth: "200px",
    minWidth: "120px",
    transition: "transform 0.1s",
  },
  notDownloaded: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "300px",
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    color: "#f59e0b",
    fontSize: "1.1rem",
    fontWeight: 600,
  },
  sectionTitle: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#f8fafc",
    marginBottom: "6px",
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
    setTaskNum(num);
  }, [taskData]);

  const handleLabel = useCallback(async (label: string) => {
    if (!annotator || !taskData || saving) return;
    setSaving(true);
    try {
      await fetch(`${API}/smile-annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotator, task_number: taskData.task.task_number, label, notes }),
      });
      setAnnotations((prev) => {
        const a = prev ? { ...prev } : { annotator, annotations: {} };
        a.annotations = {
          ...a.annotations,
          [String(taskData.task.task_number)]: { label, timestamp: new Date().toISOString(), notes: notes || undefined },
        };
        return a;
      });
      if (taskData.task.task_number < taskData.task.total_tasks) {
        goToTask(taskData.task.task_number + 1);
      }
    } catch { /* ignore */ }
    setSaving(false);
  }, [annotator, taskData, saving, goToTask]);

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

  useEffect(() => {
    setNotes(currentAnnotation?.notes ?? "");
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

  return (
    <div style={st.page}>
      {/* Top bar */}
      <div style={st.topBar}>
        <span style={st.taskLabel}>
          Task {task.task_number} / {task.available_tasks}
        </span>
        <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
          ({task.total_tasks} total)
        </span>
        <span style={st.annotator}>{annotator}</span>

        <button
          style={st.navBtn}
          onClick={() => goToTask(task.task_number - 1)}
          disabled={task.task_number <= 1}
        >
          Prev
        </button>
        <button
          style={st.navBtn}
          onClick={() => goToTask(task.task_number + 1)}
          disabled={task.task_number >= task.total_tasks}
        >
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
            fontSize: "0.85rem",
            fontWeight: 600,
            color: SMILE_LABELS.find((l) => l.key === currentLabel)?.color ?? "#94a3b8",
            padding: "4px 10px",
            backgroundColor: "#0f172a",
            borderRadius: "6px",
          }}>
            Labeled: {SMILE_LABELS.find((l) => l.key === currentLabel)?.display ?? currentLabel}
          </span>
        )}

        <span style={{ ...st.annotator, marginLeft: "auto", fontSize: "0.8rem" }}>
          Video: {task.video_id}
        </span>
        <button style={st.logoutBtn} onClick={handleLogout}>Logout</button>
      </div>

      {/* Instruction */}
      <div style={{
        padding: "8px 16px",
        marginBottom: "10px",
        backgroundColor: "#1c1917",
        border: "1px solid #f59e0b44",
        borderRadius: "8px",
        fontSize: "0.85rem",
        color: "#fbbf24",
      }}>
        Provide the label for the <strong>highlighted smile</strong> (orange border on video, yellow bar on timeline).
        If there are other smiles in the video, ignore them.
      </div>

      <div style={st.mainLayout}>
        <div style={st.leftPanel}>
          {task.video_downloaded ? (
            <>
              {/* Video with smile-region border glow */}
              <div style={{
                borderRadius: "10px",
                padding: "4px",
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

              {/* Custom seek bar with smile highlight */}
              <div style={{
                position: "relative",
                height: "20px",
                marginTop: "6px",
                backgroundColor: "#1e293b",
                borderRadius: "4px",
                cursor: "pointer",
                overflow: "hidden",
              }}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const pct = (e.clientX - rect.left) / rect.width;
                  seek(playStart + pct * (playEnd - playStart));
                }}
              >
                {/* Smile region highlight */}
                <div style={{
                  position: "absolute",
                  left: `${seekBarSmileLeft}%`,
                  width: `${seekBarSmileWidth}%`,
                  top: 0,
                  bottom: 0,
                  backgroundColor: "#f59e0b55",
                  borderLeft: "2px solid #f59e0b",
                  borderRight: "2px solid #f59e0b",
                }} />
                {/* Playhead */}
                <div style={{
                  position: "absolute",
                  left: `${playEnd > playStart ? ((currentTime - playStart) / (playEnd - playStart)) * 100 : 0}%`,
                  top: 0,
                  bottom: 0,
                  width: "3px",
                  backgroundColor: "#3b82f6",
                  borderRadius: "2px",
                  transition: "left 0.05s linear",
                }} />
              </div>

              {/* Controls row */}
              <div style={st.controls}>
                <button style={st.playBtn} onClick={togglePlay}>
                  {playing ? "\u23F8 Pause" : "\u25B6 Play"}
                </button>

                {/* Speed */}
                <div style={{ display: "flex", gap: "3px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.7rem", color: "#64748b", marginRight: "2px" }}>Speed</span>
                  {SPEEDS.map((r) => (
                    <button
                      key={r}
                      style={{ ...st.optBtn, ...(speed === r ? st.optActive : {}) }}
                      onClick={() => setSpeed(r)}
                    >
                      {r}x
                    </button>
                  ))}
                </div>

                <span style={{ color: "#334155" }}>|</span>

                {/* Before seconds */}
                <div style={{ display: "flex", gap: "3px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.7rem", color: "#64748b", marginRight: "2px" }}>Before</span>
                  {BEFORE_OPTIONS.map((v) => (
                    <button
                      key={v}
                      style={{ ...st.optBtn, ...(ctxBefore === v ? st.optActive : {}) }}
                      onClick={() => setCtxBefore(v)}
                    >
                      {v}s
                    </button>
                  ))}
                </div>

                <span style={{ color: "#334155" }}>|</span>

                {/* After seconds */}
                <div style={{ display: "flex", gap: "3px", alignItems: "center" }}>
                  <span style={{ fontSize: "0.7rem", color: "#64748b", marginRight: "2px" }}>After</span>
                  {AFTER_OPTIONS.map((v) => (
                    <button
                      key={v}
                      style={{ ...st.optBtn, ...(ctxAfter === v ? st.optActive : {}) }}
                      onClick={() => setCtxAfter(v)}
                    >
                      {v}s
                    </button>
                  ))}
                </div>

                <span style={st.timeDisplay}>
                  {fmtTime(currentTime)} / {fmtTime(duration)}
                </span>
              </div>

              <div style={{
                display: "flex", gap: "12px", marginTop: "4px",
                fontSize: "0.8rem", color: "#64748b", padding: "0 4px",
              }}>
                <span>Smile: {fmtTime(task.smile_start)} - {fmtTime(task.smile_end)}</span>
                <span>Peak: {task.peak_r.toFixed(2)}</span>
                <span>Mean: {task.mean_r.toFixed(2)}</span>
              </div>
            </>
          ) : (
            <div style={st.notDownloaded}>Video not yet downloaded</div>
          )}

          {/* Notes */}
          <div style={{ marginTop: "12px" }}>
            <input
              type="text"
              placeholder="Notes (optional)"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                fontSize: "0.85rem",
                border: "1px solid #475569",
                borderRadius: "6px",
                backgroundColor: "#1e293b",
                color: "#e2e8f0",
                boxSizing: "border-box" as const,
                outline: "none",
              }}
            />
          </div>

          {/* Label buttons */}
          <div style={st.labelRow}>
            {SMILE_LABELS.map((l) => (
              <button
                key={l.key}
                style={{
                  ...st.labelBtn,
                  backgroundColor: l.color,
                  opacity: saving ? 0.6 : 1,
                  borderColor: currentLabel === l.key ? "#fff" : "transparent",
                }}
                onClick={() => handleLabel(l.key)}
                disabled={saving}
              >
                {l.display}
              </button>
            ))}
          </div>

          {/* Label Descriptions */}
          <div style={{
              marginTop: "8px",
              padding: "14px 18px",
              backgroundColor: "#1e293b",
              borderRadius: "10px",
              fontSize: "0.85rem",
              lineHeight: 1.6,
              color: "#cbd5e1",
            }}>
              <div style={{ marginBottom: "10px" }}>
                <strong style={{ color: "#22c55e" }}>Genuine Smile</strong>
                <div>A smile driven by true positive emotion. Some physical things to look for are wrinkled skin near corners of eyes, cheeks pushing upwards, eyebrows lowered, and straightening of the lowered eyelid. Genuine smiles build in intensity over time and usually do not fade instantly. Some behaviors to look for are &ldquo;sparkles&rdquo; in the eyes or remembering something fondly. These smiles may feel somewhat more natural and &ldquo;fluid&rdquo; than polite smiles.
              <br />
              
              Note: You should use behaviors only as helpful context, as polite and genuine smiles can have overlapping behaviors. </div>
              </div>
              <div style={{ marginBottom: "10px" }}>
                <strong style={{ color: "#3b82f6" }}>Polite Smile</strong>
                <div>A controlled smile meant for social etiquette, or oftentimes acknowledgement of an interviewer's question. Polite smiles can also indicate positive emotions sometimes: in these cases, the interview context and physical features should be considered. Some physical things to look for are eyes remaining relatively static and/or open even as the mouth moves. They often appear and disappear quicker than genuine smiles. If the subject laughs, consider whether it is from joy (genuine smile) or from acknowledgement, nervousness, or as a social/communicative function (polite smile).</div>
              </div>
              <div style={{ marginBottom: "10px" }}>
                <strong style={{ color: "#f59e0b" }}>Masking Smile</strong>
                <div>A smile used to convey a different underlying emotion than happiness. Look for micro-expressions in other parts of the face that betray the smile: typical emotions with masking smiles include sadness, anger, contempt, irony, or frustration. Also, see if the smile disappears quickly by looking at the corners of the mouth. If a smile disappears quickly, it is likely either a masking smile or a polite smile. In these cases, use behaviors, speech audio, and narrative context to make your decision.</div>
              </div>
              <div>
                <strong style={{ color: "#64748b" }}>Not a Smile</strong>
                <div>For segments where the subject is not smiling. If you're unsure, then assume it is a smile.</div>
              </div>
            </div>
        </div>

        {/* Transcript panel -- full transcript, scrollable */}
        <div style={st.rightPanel}>
          <h3 style={st.sectionTitle}>Transcript</h3>
          <TranscriptTrack
            utterances={taskData.utterances}
            currentTimeMs={currentTime * 1000}
            onSeek={(ms) => seek(ms / 1000)}
            smileStartMs={task.smile_start * 1000}
            smileEndMs={task.smile_end * 1000}
            maxHeight="calc(100vh - 180px)"
            initialScrollToMs={task.smile_start * 1000}
          />
        </div>
      </div>
    </div>
  );
}
