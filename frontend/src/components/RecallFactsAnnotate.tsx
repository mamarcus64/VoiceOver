import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";

const STORAGE_KEY = "recall_facts_annotator";

interface Task {
  id: number;
  type: "topic" | "memory_type";
  transcript_id: string;
  sentence_text: string;
  context_before: string[];
  context_after: string[];
  llm_memory_type: "internal" | "external";
  llm_topics: string[];
  topic_to_validate?: string;
}

interface TasksData {
  total_tasks: number;
  n_topic_tasks: number;
  n_memory_type_tasks: number;
  tasks: Task[];
}

// topic tasks use "yes" | "no" | "unsure"
// memory_type tasks use "internal" | "external" | "unsure"
type Answer = string;

const TOPIC_LIST = [
  "Captivity",
  "Daily life (childhood)",
  "Daily life (imprisonment)",
  "Feelings and thoughts",
  "Forced labor",
  "Government",
  "Health",
  "Liberation",
  "Post-conflict",
  "Refugee experiences",
  "Parents",
];

const MT_EXAMPLES = [
  { text: "It was a sunny day in Vancouver.", label: "internal" },
  { text: "It rains a lot in Vancouver.", label: "external" },
  { text: "I always loved Vancouver.", label: "external" },
  { text: "I danced so long that night.", label: "internal" },
  { text: "I was a dancer.", label: "external" },
  { text: "I was so angry when I saw him.", label: "internal" },
  { text: "I was more emotional back in those days.", label: "external" },
  { text: "I have two brothers.", label: "external" },
];

const st: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "32px 16px",
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: "12px",
    padding: "28px 32px",
    width: "100%",
    maxWidth: "720px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
    gap: "12px",
    flexWrap: "wrap" as const,
  },
  title: {
    fontSize: "1.2rem",
    fontWeight: 700,
    color: "#f8fafc",
    margin: 0,
  },
  annotatorBadge: {
    fontSize: "0.75rem",
    color: "#94a3b8",
    backgroundColor: "#0f172a",
    padding: "4px 10px",
    borderRadius: "20px",
    cursor: "pointer",
  },
  progressBar: {
    height: "6px",
    backgroundColor: "#334155",
    borderRadius: "3px",
    marginBottom: "6px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    borderRadius: "3px",
    transition: "width 0.3s ease",
  },
  phaseHeader: {
    borderRadius: "8px",
    padding: "14px 16px",
    marginBottom: "20px",
  },
  phaseTitle: {
    fontSize: "0.9rem",
    fontWeight: 700,
    marginBottom: "8px",
  },
  definitionBox: {
    fontSize: "0.82rem",
    lineHeight: 1.6,
    color: "#cbd5e1",
  },
  defRow: {
    display: "flex",
    gap: "6px",
    marginBottom: "4px",
    alignItems: "flex-start",
  },
  defLabel: {
    fontWeight: 700,
    whiteSpace: "nowrap" as const,
    minWidth: "76px",
  },
  examplesGrid: {
    marginTop: "10px",
    display: "grid",
    gap: "4px",
  },
  exampleRow: {
    display: "flex",
    gap: "8px",
    alignItems: "baseline",
    fontSize: "0.78rem",
  },
  contextBlock: {
    borderLeft: "3px solid #334155",
    paddingLeft: "12px",
    marginBottom: "6px",
    color: "#64748b",
    fontSize: "0.85rem",
    lineHeight: 1.5,
    fontStyle: "italic",
  },
  sentenceBlock: {
    backgroundColor: "#0f172a",
    border: "1px solid #475569",
    borderRadius: "8px",
    padding: "16px 20px",
    margin: "14px 0",
    fontSize: "1rem",
    lineHeight: 1.6,
    color: "#f1f5f9",
  },
  questionText: {
    fontSize: "1rem",
    fontWeight: 600,
    color: "#e2e8f0",
    margin: "18px 0 14px",
  },
  topicBadge: {
    display: "inline-block",
    backgroundColor: "#312e81",
    color: "#a5b4fc",
    border: "1px solid #4f46e5",
    borderRadius: "6px",
    padding: "3px 10px",
    fontSize: "0.85rem",
    fontWeight: 600,
  },
  answerRow: {
    display: "flex",
    gap: "10px",
    marginTop: "14px",
    flexWrap: "wrap" as const,
  },
  navRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: "24px",
    gap: "12px",
  },
  navBtn: {
    padding: "8px 18px",
    borderRadius: "7px",
    fontSize: "0.9rem",
    fontWeight: 600,
    cursor: "pointer",
    border: "1px solid #475569",
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
  input: {
    width: "100%",
    padding: "12px 16px",
    fontSize: "1rem",
    backgroundColor: "#0f172a",
    border: "1px solid #475569",
    borderRadius: "8px",
    color: "#f1f5f9",
    outline: "none",
    boxSizing: "border-box" as const,
    marginTop: "16px",
  },
  submitBtn: {
    marginTop: "16px",
    width: "100%",
    padding: "12px",
    fontSize: "1rem",
    fontWeight: 700,
    borderRadius: "8px",
    cursor: "pointer",
    border: "none",
    backgroundColor: "#6366f1",
    color: "#fff",
  },
};

function AnswerButton({
  selected,
  onClick,
  label,
  color,
}: {
  selected: boolean;
  onClick: () => void;
  label: string;
  color: string;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: "1 1 120px",
        padding: "13px 16px",
        borderRadius: "8px",
        fontSize: "0.92rem",
        fontWeight: 700,
        cursor: "pointer",
        border: selected ? `2px solid ${color}` : "2px solid #334155",
        backgroundColor: selected ? `${color}22` : "#0f172a",
        color: selected ? color : "#94a3b8",
        transition: "all 0.15s",
      }}
    >
      {label}
    </button>
  );
}

function PhaseHeader({ phase, taskInPhase, totalInPhase }: {
  phase: "topic" | "memory_type";
  taskInPhase: number;
  totalInPhase: number;
}) {
  const isTopic = phase === "topic";
  const bg = isTopic ? "#1e1b4b" : "#0c1a2e";
  const accent = isTopic ? "#a78bfa" : "#22d3ee";

  return (
    <div style={{ ...st.phaseHeader, backgroundColor: bg, border: `1px solid ${accent}22` }}>
      <div style={{ ...st.phaseTitle, color: accent }}>
        {isTopic
          ? `Phase 1 of 2 — Topic Validation (${taskInPhase} / ${totalInPhase})`
          : `Phase 2 of 2 — Memory Type Labeling (${taskInPhase} / ${totalInPhase})`}
      </div>
      <div style={st.definitionBox}>
        {isTopic ? (
          <>
            <div>
              You'll see a sentence that the AI tagged with a specific topic. Your job is to
              confirm whether that topic genuinely applies.
            </div>
            <div style={{ marginTop: "8px", color: "#94a3b8", fontSize: "0.78rem" }}>
              Topics: {TOPIC_LIST.join(" · ")}
            </div>
          </>
        ) : (
          <>
            <div style={{ ...st.defRow, marginBottom: "6px" }}>
              <span style={{ ...st.defLabel, color: "#34d399" }}>internal —</span>
              <span>references a <em>specific</em> time, place, or event: location, people, actions, thoughts, perceptual details involved in one particular moment.</span>
            </div>
            <div style={st.defRow}>
              <span style={{ ...st.defLabel, color: "#f87171" }}>external —</span>
              <span>factual/general information, habitual or repeated statements, things that don't require recalling one specific event.</span>
            </div>
            <div style={st.examplesGrid}>
              {MT_EXAMPLES.map((ex, i) => (
                <div key={i} style={st.exampleRow}>
                  <span style={{
                    color: ex.label === "internal" ? "#34d399" : "#f87171",
                    fontWeight: 700,
                    minWidth: "72px",
                    fontSize: "0.76rem",
                  }}>
                    {ex.label}
                  </span>
                  <span style={{ color: "#94a3b8", fontStyle: "italic" }}>"{ex.text}"</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function RecallFactsAnnotate() {
  const navigate = useNavigate();
  const [annotator, setAnnotator] = useState<string>(() =>
    localStorage.getItem(STORAGE_KEY) || ""
  );
  const [nameInput, setNameInput] = useState("");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [answers, setAnswers] = useState<Record<number, Answer>>({});
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!annotator) return;
    setLoading(true);
    Promise.all([
      fetch("/api/recall-facts/tasks").then((r) => r.json()),
      fetch(`/api/recall-facts/my-annotations?annotator=${encodeURIComponent(annotator)}`).then((r) => r.json()),
    ])
      .then(([taskData, annData]: [TasksData, { annotations: Record<string, { answer: Answer }> }]) => {
        setTasks(taskData.tasks);
        const saved: Record<number, Answer> = {};
        for (const [k, v] of Object.entries(annData.annotations || {})) {
          saved[parseInt(k)] = v.answer;
        }
        setAnswers(saved);
        const firstUnanswered = taskData.tasks.findIndex((t) => !(t.id in saved));
        setCurrentIdx(firstUnanswered === -1 ? taskData.tasks.length : firstUnanswered);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [annotator]);

  const saveAnswer = useCallback(
    async (taskId: number, answer: Answer) => {
      setSaving(true);
      try {
        await fetch("/api/recall-facts/annotation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ annotator, task_id: taskId, answer }),
        });
        setAnswers((prev) => ({ ...prev, [taskId]: answer }));
        // Auto-advance to next task
        setCurrentIdx((i) => Math.min(i + 1, tasks.length - 1));
      } catch (e) {
        console.error("Save failed:", e);
      } finally {
        setSaving(false);
      }
    },
    [annotator, tasks.length]
  );

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    const name = nameInput.trim();
    if (!name) return;
    localStorage.setItem(STORAGE_KEY, name);
    setAnnotator(name);
  };

  const topicTasks = tasks.filter((t) => t.type === "topic");
  const mtTasks = tasks.filter((t) => t.type === "memory_type");
  const answered = Object.keys(answers).length;
  const total = tasks.length;
  const allDone = total > 0 && answered >= total;
  const currentTask = tasks[currentIdx];
  const currentAnswer = currentTask ? answers[currentTask.id] : undefined;

  // ── Login screen ─────────────────────────────────────────────────────────
  if (!annotator) {
    return (
      <div style={st.page}>
        <div style={{ ...st.card, maxWidth: "440px" }}>
          <h1 style={{ ...st.title, fontSize: "1.4rem", marginBottom: "8px", textAlign: "center" }}>
            Recall &amp; Topic Annotation
          </h1>
          <p style={{ color: "#94a3b8", marginBottom: "8px", fontSize: "0.88rem", textAlign: "center" }}>
            You'll complete two phases:
          </p>
          <div style={{ backgroundColor: "#0f172a", borderRadius: "8px", padding: "14px 16px", marginBottom: "20px", fontSize: "0.85rem" }}>
            <div style={{ color: "#a78bfa", fontWeight: 700, marginBottom: "6px" }}>Phase 1 — Topic Validation (100 tasks)</div>
            <div style={{ color: "#94a3b8" }}>Confirm whether an AI-assigned topic genuinely applies to the sentence.</div>
            <div style={{ color: "#22d3ee", fontWeight: 700, marginBottom: "6px", marginTop: "10px" }}>Phase 2 — Memory Type Labeling (100 tasks)</div>
            <div style={{ color: "#94a3b8" }}>Independently label each sentence as <em>internal</em> or <em>external</em> memory detail.</div>
          </div>
          <form onSubmit={handleLogin}>
            <input
              style={st.input}
              type="text"
              placeholder="Your name or initials"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              autoFocus
            />
            <button style={st.submitBtn} type="submit" disabled={!nameInput.trim()}>
              Start →
            </button>
          </form>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={st.page}>
        <div style={st.card}>
          <p style={{ color: "#94a3b8" }}>Loading tasks…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={st.page}>
        <div style={st.card}>
          <p style={{ color: "#f87171" }}>Error: {error}</p>
          <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>
            Make sure the backend is running and tasks have been generated.
          </p>
        </div>
      </div>
    );
  }

  // ── Done ──────────────────────────────────────────────────────────────────
  if (allDone) {
    return (
      <div style={st.page}>
        <div style={{ ...st.card, textAlign: "center", padding: "48px 32px" }}>
          <div style={{ fontSize: "3rem", marginBottom: "16px" }}>✓</div>
          <h2 style={{ ...st.title, fontSize: "1.5rem", marginBottom: "12px" }}>
            All {total} tasks complete!
          </h2>
          <p style={{ color: "#94a3b8", marginBottom: "28px" }}>
            Thank you, <strong style={{ color: "#e2e8f0" }}>{annotator}</strong>.
          </p>
          <button
            style={{ ...st.submitBtn, width: "auto", padding: "12px 32px" }}
            onClick={() => navigate("/recall-facts-agreement")}
          >
            View Agreement Results →
          </button>
        </div>
      </div>
    );
  }

  if (!currentTask) return null;

  const isTopic = currentTask.type === "topic";
  const accent = isTopic ? "#a78bfa" : "#22d3ee";

  // Phase progress
  const phaseTasks = isTopic ? topicTasks : mtTasks;
  const phaseAnswered = phaseTasks.filter((t) => t.id in answers).length;
  const phaseTotal = phaseTasks.length;
  const phaseIdxInList = phaseTasks.findIndex((t) => t.id === currentTask.id);
  const phasePct = phaseTotal > 0 ? (phaseAnswered / phaseTotal) * 100 : 0;

  return (
    <div style={st.page}>
      <div style={st.card}>
        {/* Header */}
        <div style={st.header}>
          <h1 style={st.title}>Recall &amp; Topic Annotation</h1>
          <span
            style={st.annotatorBadge}
            onClick={() => { localStorage.removeItem(STORAGE_KEY); setAnnotator(""); }}
            title="Click to switch annotator"
          >
            {annotator} ×
          </span>
        </div>

        {/* Overall progress */}
        <div style={{ marginBottom: "4px", display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Overall: {answered} / {total}</span>
          <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
            Phase {isTopic ? "1" : "2"} of 2
          </span>
        </div>
        <div style={st.progressBar}>
          <div style={{
            ...st.progressFill,
            width: `${(answered / total) * 100}%`,
            backgroundColor: "#6366f1",
          }} />
        </div>

        {/* Phase header with definitions */}
        <div style={{ marginBottom: "16px" }} />
        <PhaseHeader
          phase={currentTask.type}
          taskInPhase={phaseIdxInList + 1}
          totalInPhase={phaseTotal}
        />

        {/* Phase progress bar */}
        <div style={{ marginBottom: "4px", display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: "0.75rem", color: "#64748b" }}>
            Task {phaseIdxInList + 1} of {phaseTotal}
          </span>
          <span style={{ fontSize: "0.75rem", color: "#64748b" }}>{phaseAnswered} answered</span>
        </div>
        <div style={{ ...st.progressBar, marginBottom: "20px" }}>
          <div style={{ ...st.progressFill, width: `${phasePct}%`, backgroundColor: accent }} />
        </div>

        {/* Context before */}
        {currentTask.context_before.length > 0 && (
          <div style={st.contextBlock}>
            {currentTask.context_before.map((s, i) => <div key={i}>{s}</div>)}
          </div>
        )}

        {/* Target sentence */}
        <div style={st.sentenceBlock}>{currentTask.sentence_text}</div>

        {/* Context after */}
        {currentTask.context_after.length > 0 && (
          <div style={st.contextBlock}>
            {currentTask.context_after.map((s, i) => <div key={i}>{s}</div>)}
          </div>
        )}

        {/* Question + answer buttons */}
        {isTopic ? (
          <>
            <p style={st.questionText}>
              Is this sentence genuinely about{" "}
              <span style={st.topicBadge}>{currentTask.topic_to_validate}</span>?
            </p>
            <div style={st.answerRow}>
              <AnswerButton selected={currentAnswer === "yes"} onClick={() => saveAnswer(currentTask.id, "yes")} label="✓  Yes" color="#4ade80" />
              <AnswerButton selected={currentAnswer === "no"} onClick={() => saveAnswer(currentTask.id, "no")} label="✗  No" color="#f87171" />
              <AnswerButton selected={currentAnswer === "unsure"} onClick={() => saveAnswer(currentTask.id, "unsure")} label="?  Unsure" color="#fbbf24" />
            </div>
          </>
        ) : (
          <>
            <p style={st.questionText}>Label this sentence:</p>
            <div style={st.answerRow}>
              <AnswerButton
                selected={currentAnswer === "internal"}
                onClick={() => saveAnswer(currentTask.id, "internal")}
                label="internal"
                color="#34d399"
              />
              <AnswerButton
                selected={currentAnswer === "external"}
                onClick={() => saveAnswer(currentTask.id, "external")}
                label="external"
                color="#f87171"
              />
              <AnswerButton
                selected={currentAnswer === "unsure"}
                onClick={() => saveAnswer(currentTask.id, "unsure")}
                label="?  Unsure"
                color="#fbbf24"
              />
            </div>
          </>
        )}

        {/* Navigation */}
        <div style={st.navRow}>
          <button
            style={{ ...st.navBtn, opacity: currentIdx === 0 ? 0.4 : 1 }}
            onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
            disabled={currentIdx === 0}
          >
            ← Prev
          </button>
          <span style={{ fontSize: "0.8rem", color: saving ? "#fbbf24" : "#475569" }}>
            {saving ? "Saving…" : currentAnswer ? "✓ Saved" : ""}
          </span>
          <button
            style={{
              ...st.navBtn,
              backgroundColor: currentAnswer ? "#4f46e5" : "#1e293b",
              borderColor: currentAnswer ? "#6366f1" : "#475569",
              color: currentAnswer ? "#c7d2fe" : "#64748b",
            }}
            onClick={() => { if (currentIdx < tasks.length - 1) setCurrentIdx((i) => i + 1); }}
          >
            Next →
          </button>
        </div>

        {answered < total && (
          <div style={{ marginTop: "14px", textAlign: "center" }}>
            <button
              style={{ background: "none", border: "none", color: "#6366f1", cursor: "pointer", fontSize: "0.8rem", textDecoration: "underline" }}
              onClick={() => {
                const next = tasks.findIndex((t) => !(t.id in answers));
                if (next !== -1) setCurrentIdx(next);
              }}
            >
              Jump to next unanswered
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
