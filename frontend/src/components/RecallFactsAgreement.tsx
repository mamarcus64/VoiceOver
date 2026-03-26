import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const STORAGE_KEY = "recall_facts_annotator";

interface AnnotatorRow {
  annotator: string;
  completed: number;
  topic_done: number;
  topic_agree: number;
  topic_disagree: number;
  topic_unsure: number;
  topic_pct_agree: number | null;
  mt_done: number;
  mt_agree: number;
  mt_disagree: number;
  mt_unsure: number;
  mt_pct_agree: number | null;
}

interface AgreementData {
  annotators: AnnotatorRow[];
  inter_annotator_agreement_pct: number | null;
  total_tasks: number;
  n_topic_tasks: number;
  n_memory_type_tasks: number;
}

interface TaskDetail {
  id: number;
  type: string;
  sentence: string;
  transcript_id: string;
  topic: string | null;
  llm_answer: string;
  responses: Record<string, string | undefined>;
}

interface DetailData {
  tasks: TaskDetail[];
  annotators: string[];
}

const st: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    padding: "32px 16px",
  },
  container: {
    maxWidth: "1000px",
    margin: "0 auto",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: "28px",
    gap: "16px",
    flexWrap: "wrap" as const,
  },
  title: {
    fontSize: "1.4rem",
    fontWeight: 700,
    color: "#f8fafc",
    margin: 0,
  },
  subtitle: {
    fontSize: "0.85rem",
    color: "#64748b",
    marginTop: "4px",
  },
  section: {
    backgroundColor: "#1e293b",
    borderRadius: "10px",
    padding: "24px",
    marginBottom: "20px",
  },
  sectionTitle: {
    fontSize: "0.75rem",
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase" as const,
    color: "#64748b",
    marginBottom: "16px",
  },
  statRow: {
    display: "flex",
    gap: "16px",
    flexWrap: "wrap" as const,
    marginBottom: "16px",
  },
  statBox: {
    backgroundColor: "#0f172a",
    borderRadius: "8px",
    padding: "14px 20px",
    flex: "1 1 160px",
    textAlign: "center" as const,
  },
  statNum: {
    fontSize: "1.8rem",
    fontWeight: 700,
    color: "#f8fafc",
    lineHeight: 1,
    marginBottom: "4px",
  },
  statLabel: {
    fontSize: "0.75rem",
    color: "#64748b",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: "0.88rem",
  },
  th: {
    textAlign: "left" as const,
    padding: "8px 12px",
    color: "#64748b",
    fontWeight: 600,
    fontSize: "0.75rem",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
    borderBottom: "1px solid #334155",
  },
  td: {
    padding: "10px 12px",
    borderBottom: "1px solid #1e293b",
    verticalAlign: "top" as const,
  },
  pill: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: "12px",
    fontSize: "0.75rem",
    fontWeight: 600,
  },
  btn: {
    padding: "8px 18px",
    borderRadius: "7px",
    fontSize: "0.85rem",
    fontWeight: 600,
    cursor: "pointer",
    border: "1px solid #475569",
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
};

function AgreePct({ pct }: { pct: number | null }) {
  if (pct === null) return <span style={{ color: "#475569" }}>—</span>;
  const color = pct >= 80 ? "#4ade80" : pct >= 60 ? "#fbbf24" : "#f87171";
  return (
    <span style={{ color, fontWeight: 700 }}>
      {pct.toFixed(1)}%
    </span>
  );
}

function AnswerPill({ answer }: { answer: string | undefined }) {
  if (!answer) return <span style={{ color: "#475569" }}>—</span>;
  const map: Record<string, [string, string]> = {
    yes: ["#4ade8022", "#4ade80"],
    no: ["#f8717122", "#f87171"],
    unsure: ["#fbbf2422", "#fbbf24"],
    internal: ["#34d39922", "#34d399"],
    external: ["#f8717122", "#f87171"],
  };
  const [bg, color] = map[answer] ?? ["#33415522", "#64748b"];
  return (
    <span style={{ ...st.pill, backgroundColor: bg, color }}>{answer}</span>
  );
}

export default function RecallFactsAgreement() {
  const navigate = useNavigate();
  const me = localStorage.getItem(STORAGE_KEY) || "";

  const [agreement, setAgreement] = useState<AgreementData | null>(null);
  const [detail, setDetail] = useState<DetailData | null>(null);
  const [showDetail, setShowDetail] = useState(false);
  const [filterType, setFilterType] = useState<"all" | "memory_type" | "topic">("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/recall-facts/agreement").then((r) => r.json()),
      fetch("/api/recall-facts/task-detail").then((r) => r.json()),
    ])
      .then(([ag, det]) => {
        setAgreement(ag);
        setDetail(det);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={st.page}>
        <div style={st.container}>
          <p style={{ color: "#94a3b8" }}>Loading…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={st.page}>
        <div style={st.container}>
          <p style={{ color: "#f87171" }}>Error: {error}</p>
        </div>
      </div>
    );
  }

  const annotators = detail?.annotators ?? [];
  const filteredTasks = (detail?.tasks ?? []).filter(
    (t) => filterType === "all" || t.type === filterType
  );

  return (
    <div style={st.page}>
      <div style={st.container}>
        {/* Header */}
        <div style={st.header}>
          <div>
            <h1 style={st.title}>Recall &amp; Topic Agreement</h1>
            <p style={st.subtitle}>
              Human annotations vs. LLM baseline ·{" "}
              {agreement?.total_tasks ?? 0} tasks (
              {agreement?.n_topic_tasks} topic + {agreement?.n_memory_type_tasks} memory-type)
            </p>
          </div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            {me && (
              <button
                style={st.btn}
                onClick={() => navigate("/recall-facts-annotation")}
              >
                ← Annotate
              </button>
            )}
          </div>
        </div>

        {/* Summary stats */}
        {agreement && (
          <div style={st.section}>
            <div style={st.sectionTitle}>Summary</div>
            <div style={st.statRow}>
              <div style={st.statBox}>
                <div style={st.statNum}>{agreement.annotators.length}</div>
                <div style={st.statLabel}>Annotators</div>
              </div>
              {agreement.inter_annotator_agreement_pct !== null && (
                <div style={st.statBox}>
                  <div style={{ ...st.statNum, color: "#a5b4fc" }}>
                    {agreement.inter_annotator_agreement_pct.toFixed(1)}%
                  </div>
                  <div style={st.statLabel}>Inter-annotator agreement</div>
                </div>
              )}
              {agreement.annotators.length > 0 && (
                <>
                  <div style={st.statBox}>
                    <div style={{ ...st.statNum, color: "#a78bfa" }}>
                      {(
                        agreement.annotators
                          .filter((a) => a.topic_pct_agree !== null)
                          .reduce((s, a) => s + (a.topic_pct_agree ?? 0), 0) /
                          Math.max(1, agreement.annotators.filter((a) => a.topic_pct_agree !== null).length)
                      ).toFixed(1)}%
                    </div>
                    <div style={st.statLabel}>Avg topic validation w/ LLM</div>
                  </div>
                  <div style={st.statBox}>
                    <div style={{ ...st.statNum, color: "#22d3ee" }}>
                      {(
                        agreement.annotators
                          .filter((a) => a.mt_pct_agree !== null)
                          .reduce((s, a) => s + (a.mt_pct_agree ?? 0), 0) /
                          Math.max(1, agreement.annotators.filter((a) => a.mt_pct_agree !== null).length)
                      ).toFixed(1)}%
                    </div>
                    <div style={st.statLabel}>Avg memory-type agree w/ LLM</div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Per-annotator table */}
        {agreement && agreement.annotators.length > 0 && (
          <div style={st.section}>
            <div style={st.sectionTitle}>Per-annotator breakdown</div>
            <div style={{ overflowX: "auto" }}>
              <table style={st.table}>
                <thead>
                    <tr>
                        <th style={st.th}>Annotator</th>
                        <th style={st.th}>Done</th>
                        <th style={st.th}>🏷 Topic agree w/ LLM</th>
                        <th style={st.th}>↳ disagree / unsure</th>
                        <th style={st.th}>🧠 Memory-type agree w/ LLM</th>
                        <th style={st.th}>↳ disagree / unsure</th>
                      </tr>
                </thead>
                <tbody>
                  {agreement.annotators.map((row) => (
                    <tr key={row.annotator} style={{ backgroundColor: row.annotator === me ? "#1e3a5f22" : undefined }}>
                      <td style={st.td}>
                        <span style={{ fontWeight: row.annotator === me ? 700 : 400, color: row.annotator === me ? "#93c5fd" : "#e2e8f0" }}>
                          {row.annotator}
                          {row.annotator === me ? " (you)" : ""}
                        </span>
                      </td>
                      <td style={st.td}>{row.completed} / {agreement.total_tasks}</td>
                      <td style={st.td}>
                        <AgreePct pct={row.topic_pct_agree} />
                        {row.topic_done > 0 && (
                          <span style={{ color: "#475569", fontSize: "0.75rem", marginLeft: "6px" }}>
                            ({row.topic_agree}/{row.topic_done})
                          </span>
                        )}
                      </td>
                      <td style={{ ...st.td, color: "#64748b", fontSize: "0.82rem" }}>
                        {row.topic_done > 0
                          ? `${row.topic_disagree} disagree, ${row.topic_unsure} unsure`
                          : "—"}
                      </td>
                      <td style={st.td}>
                        <AgreePct pct={row.mt_pct_agree} />
                        {row.mt_done > 0 && (
                          <span style={{ color: "#475569", fontSize: "0.75rem", marginLeft: "6px" }}>
                            ({row.mt_agree}/{row.mt_done})
                          </span>
                        )}
                      </td>
                      <td style={{ ...st.td, color: "#64748b", fontSize: "0.82rem" }}>
                        {row.mt_done > 0
                          ? `${row.mt_disagree} disagree, ${row.mt_unsure} unsure`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {agreement && agreement.annotators.length === 0 && (
          <div style={st.section}>
            <p style={{ color: "#64748b", textAlign: "center", margin: 0 }}>
              No annotations yet.{" "}
              <button
                style={{ background: "none", border: "none", color: "#6366f1", cursor: "pointer", textDecoration: "underline" }}
                onClick={() => navigate("/recall-facts-annotation")}
              >
                Be the first to annotate →
              </button>
            </p>
          </div>
        )}

        {/* Per-task detail */}
        {detail && detail.tasks.length > 0 && (
          <div style={st.section}>
            <div
              style={{ ...st.sectionTitle, cursor: "pointer", userSelect: "none", display: "flex", justifyContent: "space-between" }}
              onClick={() => setShowDetail((v) => !v)}
            >
              <span>Per-task detail ({filteredTasks.length} tasks)</span>
              <span style={{ color: "#6366f1" }}>{showDetail ? "▲ Hide" : "▼ Show"}</span>
            </div>

            {showDetail && (
              <>
                {/* Filter buttons */}
                <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
                  {(["all", "topic", "memory_type"] as const).map((f) => (
                    <button
                      key={f}
                      style={{
                        ...st.btn,
                        backgroundColor: filterType === f ? "#4f46e5" : "#334155",
                        borderColor: filterType === f ? "#6366f1" : "#475569",
                        color: filterType === f ? "#c7d2fe" : "#e2e8f0",
                        padding: "5px 14px",
                        fontSize: "0.8rem",
                      }}
                      onClick={() => setFilterType(f)}
                    >
                      {f === "all" ? "All" : f === "topic" ? "🏷 Topic" : "🧠 Memory-type"}
                    </button>
                  ))}
                </div>

                <div style={{ overflowX: "auto" }}>
                  <table style={st.table}>
                    <thead>
                      <tr>
                        <th style={{ ...st.th, width: "40px" }}>#</th>
                        <th style={st.th}>Type</th>
                        <th style={st.th}>Sentence</th>
                        {annotators.map((a) => (
                          <th key={a} style={{ ...st.th, whiteSpace: "nowrap" }}>{a}</th>
                        ))}
                        <th style={st.th}>LLM</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTasks.map((task) => (
                        <tr key={task.id}>
                          <td style={{ ...st.td, color: "#475569", fontSize: "0.75rem" }}>{task.id}</td>
                          <td style={{ ...st.td, whiteSpace: "nowrap" }}>
                            <span
                              style={{
                                fontSize: "0.7rem",
                                padding: "2px 6px",
                                borderRadius: "4px",
                                backgroundColor: task.type === "memory_type" ? "#22d3ee18" : "#a78bfa18",
                                color: task.type === "memory_type" ? "#22d3ee" : "#a78bfa",
                              }}
                            >
                              {task.type === "memory_type" ? "🧠 mem-type" : `🏷 ${task.topic ?? "topic"}`}
                            </span>
                          </td>
                          <td style={{ ...st.td, maxWidth: "340px", fontSize: "0.82rem", color: "#cbd5e1" }}>
                            {task.sentence}
                          </td>
                          {annotators.map((a) => (
                            <td key={a} style={{ ...st.td, textAlign: "center" }}>
                              <AnswerPill answer={task.responses[a]} />
                            </td>
                          ))}
                          <td style={{ ...st.td, textAlign: "center" }}>
                            <AnswerPill answer={task.llm_answer} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
