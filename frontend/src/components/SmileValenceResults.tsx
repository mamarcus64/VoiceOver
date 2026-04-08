import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const API = "/api";

type Valence = "negative" | "neutral" | "positive";
const VALENCE_LABELS: Valence[] = ["negative", "neutral", "positive"];
const VALENCE_COLOR: Record<Valence, string> = {
  negative: "#dc2626",
  neutral: "#6366f1",
  positive: "#16a34a",
};

interface HumanAnnotation {
  annotator: string;
  narrative_valence?: Valence;
  speaker_valence?: Valence;
  not_a_smile?: boolean;
  timestamp: string;
}

interface LLMAnnotation {
  narrative_valence: Valence;
  speaker_valence: Valence;
  content_domain: string | null;
  narrative_structure: string | null;
}

interface TaskResult {
  task_number: number;
  video_id: string;
  smile_start: number;
  smile_end: number;
  score: number | null;
  human_annotations: HumanAnnotation[];
  llm: LLMAnnotation | null;
}

interface IAAStats {
  overall: { n_pairs: number; pct_agree: number | null; kappa: number | null };
  pairs: { annotator_1: string; annotator_2: string; n: number; pct_agree: number; kappa: number | null }[];
}

interface ConfMatrix {
  [humanLabel: string]: { [llmLabel: string]: number };
}

interface LLMAlignmentField {
  n: number;
  accuracy: number | null;
  confusion_matrix: ConfMatrix;
}

interface ResultsData {
  total_tasks: number;
  total_annotations: number;
  annotators: string[];
  iaa: { narrative_valence: IAAStats; speaker_valence: IAAStats };
  llm_alignment: { narrative_valence: LLMAlignmentField; speaker_valence: LLMAlignmentField };
  llm_tasks_count: number;
  results: TaskResult[];
}

// ── Styles ────────────────────────────────────────────────────────────────────
const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "20px 28px", maxWidth: "1300px", margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0", backgroundColor: "#0f172a", minHeight: "100vh",
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    marginBottom: "20px", flexWrap: "wrap" as const, gap: "10px",
  },
  title: { fontSize: "1.4rem", fontWeight: 700, color: "#f8fafc" },
  subtitle: { fontSize: "0.8rem", color: "#64748b", marginTop: "2px" },
  backLink: {
    fontSize: "0.85rem", color: "#93c5fd", textDecoration: "none",
    padding: "6px 14px", border: "1px solid #475569", borderRadius: "6px",
    backgroundColor: "#1e293b",
  },
  section: {
    marginBottom: "24px", padding: "16px 20px",
    backgroundColor: "#1e293b", borderRadius: "10px", border: "1px solid #334155",
  },
  sectionTitle: { fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "12px" },
  statGrid: { display: "flex", gap: "16px", flexWrap: "wrap" as const },
  statBox: {
    flex: "1 1 130px", padding: "10px 14px", borderRadius: "8px",
    backgroundColor: "#0f172a", border: "1px solid #334155", textAlign: "center" as const,
  },
  statVal: { fontSize: "1.6rem", fontWeight: 700, color: "#f8fafc", lineHeight: 1 },
  statLabel: { fontSize: "0.72rem", color: "#64748b", marginTop: "4px" },
  table: { width: "100%", borderCollapse: "collapse" as const, fontSize: "0.8rem" },
  th: {
    padding: "6px 10px", textAlign: "left" as const, color: "#64748b", fontWeight: 600,
    borderBottom: "1px solid #334155", fontSize: "0.75rem",
  },
  td: {
    padding: "6px 10px", borderBottom: "1px solid #1e293b", verticalAlign: "top" as const,
  },
  vBadge: {
    display: "inline-block", padding: "1px 8px", borderRadius: "4px",
    fontSize: "0.72rem", fontWeight: 700, color: "#fff",
  },
  notSmileBadge: {
    display: "inline-block", padding: "1px 8px", borderRadius: "4px",
    fontSize: "0.72rem", fontWeight: 600, color: "#64748b",
    backgroundColor: "#1e293b", border: "1px solid #334155",
  },
  loading: { textAlign: "center" as const, padding: "60px", color: "#64748b" },
  confCell: {
    padding: "4px 8px", textAlign: "center" as const, fontSize: "0.8rem",
    borderRadius: "4px", minWidth: "36px",
  },
  filterRow: { display: "flex", gap: "8px", marginBottom: "12px", flexWrap: "wrap" as const },
  filterBtn: {
    padding: "4px 12px", fontSize: "0.78rem", fontWeight: 600, cursor: "pointer",
    border: "1px solid #475569", borderRadius: "5px",
    backgroundColor: "#334155", color: "#94a3b8",
  },
  filterBtnActive: {
    backgroundColor: "#3b82f6", borderColor: "#3b82f6", color: "#fff",
  },
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtTime(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return "--";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function pct(v: number | null): string {
  if (v === null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function kappaLabel(k: number | null): string {
  if (k === null) return "—";
  const s = k.toFixed(3);
  if (k >= 0.8) return `${s} (almost perfect)`;
  if (k >= 0.6) return `${s} (substantial)`;
  if (k >= 0.4) return `${s} (moderate)`;
  if (k >= 0.2) return `${s} (fair)`;
  return `${s} (slight)`;
}

function ValencePill({ v }: { v: Valence | undefined | null }) {
  if (!v) return <span style={st.notSmileBadge}>—</span>;
  return <span style={{ ...st.vBadge, backgroundColor: VALENCE_COLOR[v] }}>{v}</span>;
}

function ConfusionMatrix({ data, label }: { data: ConfMatrix; label: string }) {
  const total = Object.values(data).flatMap((row) => Object.values(row)).reduce((a, b) => a + b, 0);
  if (total === 0) return <span style={{ color: "#64748b", fontSize: "0.8rem" }}>No data yet</span>;
  return (
    <div>
      <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginBottom: "6px" }}>
        Rows = human majority · Cols = LLM · {label}
      </div>
      <table style={{ borderCollapse: "collapse", fontSize: "0.78rem" }}>
        <thead>
          <tr>
            <th style={{ ...st.confCell, color: "#64748b" }}>↓human / LLM→</th>
            {VALENCE_LABELS.map((l) => (
              <th key={l} style={{ ...st.confCell, color: VALENCE_COLOR[l], fontWeight: 700 }}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {VALENCE_LABELS.map((human) => {
            const row = data[human] ?? {};
            const rowTotal = Object.values(row).reduce((a, b) => a + b, 0);
            return (
              <tr key={human}>
                <td style={{ ...st.confCell, color: VALENCE_COLOR[human], fontWeight: 700 }}>{human}</td>
                {VALENCE_LABELS.map((llm) => {
                  const val = row[llm] ?? 0;
                  const isMatch = human === llm;
                  const intensity = rowTotal > 0 ? val / rowTotal : 0;
                  return (
                    <td key={llm} style={{
                      ...st.confCell,
                      backgroundColor: val === 0 ? "#0f172a"
                        : isMatch ? `rgba(22,163,74,${0.2 + intensity * 0.6})`
                        : `rgba(220,38,38,${0.1 + intensity * 0.4})`,
                      color: val === 0 ? "#334155" : "#e2e8f0",
                      fontWeight: isMatch ? 700 : 400,
                    }}>
                      {val}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
type FilterMode = "all" | "annotated" | "not_a_smile" | "llm";

export default function SmileValenceResults() {
  const [data, setData] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterMode>("annotated");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/smile-valence-results`);
        if (res.ok) setData(await res.json());
      } catch { /* ignore */ }
      setLoading(false);
    })();
  }, []);

  if (loading) return <div style={st.page}><div style={st.loading}>Loading results...</div></div>;
  if (!data) return <div style={st.page}><div style={st.loading}>Failed to load results.</div></div>;

  const { annotators, iaa, llm_alignment } = data;
  const hasMultipleAnnotators = annotators.length >= 2;

  const results = data.results ?? [];
  const filtered = filter === "annotated"
    ? results.filter((r) => r.human_annotations.some((a) => !a.not_a_smile))
    : filter === "not_a_smile"
      ? results.filter((r) => r.human_annotations.some((a) => a.not_a_smile))
      : filter === "llm"
        ? results.filter((r) => r.llm !== null)
        : results;

  // Annotator completion counts
  const completionByAnnotator: Record<string, { answered: number; not_a_smile: number }> = {};
  for (const ann of annotators) completionByAnnotator[ann] = { answered: 0, not_a_smile: 0 };
  for (const r of results) {
    for (const a of r.human_annotations) {
      if (!completionByAnnotator[a.annotator]) completionByAnnotator[a.annotator] = { answered: 0, not_a_smile: 0 };
      if (a.not_a_smile) completionByAnnotator[a.annotator].not_a_smile += 1;
      else completionByAnnotator[a.annotator].answered += 1;
    }
  }

  return (
    <div style={st.page}>
      {/* Header */}
      <div style={st.header}>
        <div>
          <div style={st.title}>Smile Valence Results</div>
          <div style={st.subtitle}>
            {data.total_annotations} annotations · {annotators.length} annotator{annotators.length !== 1 ? "s" : ""} · {data.total_tasks} tasks · {data.llm_tasks_count} LLM-annotated
          </div>
        </div>
        <Link to="/smile-valence-annotate" style={st.backLink}>Back to Annotation</Link>
      </div>

      {/* Summary stats */}
      <div style={st.section}>
        <div style={st.sectionTitle}>Summary</div>
        <div style={st.statGrid}>
          <div style={st.statBox}>
            <div style={st.statVal}>{data.total_annotations}</div>
            <div style={st.statLabel}>Total Annotations</div>
          </div>
          <div style={st.statBox}>
            <div style={st.statVal}>{annotators.length}</div>
            <div style={st.statLabel}>Annotators</div>
          </div>
          <div style={st.statBox}>
            <div style={st.statVal}>{results.length}</div>
            <div style={st.statLabel}>Tasks w/ Any Data</div>
          </div>
          <div style={st.statBox}>
            <div style={st.statVal}>{data.llm_tasks_count}</div>
            <div style={st.statLabel}>Tasks w/ LLM Labels</div>
          </div>
        </div>

        {/* Per-annotator completion */}
        {annotators.length > 0 && (
          <div style={{ marginTop: "14px" }}>
            <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "#94a3b8", marginBottom: "6px" }}>
              Per-Annotator Completion
            </div>
            <table style={st.table}>
              <thead>
                <tr>
                  <th style={st.th}>Annotator</th>
                  <th style={st.th}>Answered</th>
                  <th style={st.th}>Not a Smile</th>
                  <th style={st.th}>Total</th>
                  <th style={st.th}>% of Tasks</th>
                </tr>
              </thead>
              <tbody>
                {annotators.map((ann) => {
                  const c = completionByAnnotator[ann];
                  const total = c.answered + c.not_a_smile;
                  return (
                    <tr key={ann}>
                      <td style={st.td}>{ann}</td>
                      <td style={st.td}>{c.answered}</td>
                      <td style={st.td}>{c.not_a_smile}</td>
                      <td style={{ ...st.td, fontWeight: 700 }}>{total}</td>
                      <td style={st.td}>{pct(total / data.total_tasks)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* IAA */}
      <div style={st.section}>
        <div style={st.sectionTitle}>Inter-Annotator Agreement (IAA)</div>
        {!hasMultipleAnnotators ? (
          <div style={{ color: "#64748b", fontSize: "0.85rem" }}>
            Need ≥ 2 annotators to compute IAA. Currently {annotators.length} annotator{annotators.length !== 1 ? "s" : ""}.
          </div>
        ) : (
          <>
            {(["narrative_valence", "speaker_valence"] as const).map((field) => {
              const stats = iaa[field];
              const label = field === "narrative_valence" ? "Narrative Valence" : "Speaker's Current Valence";
              return (
                <div key={field} style={{ marginBottom: "16px" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#f8fafc", marginBottom: "8px" }}>{label}</div>
                  <div style={st.statGrid}>
                    <div style={st.statBox}>
                      <div style={st.statVal}>{pct(stats.overall.pct_agree)}</div>
                      <div style={st.statLabel}>Pairwise Agreement</div>
                    </div>
                    <div style={st.statBox}>
                      <div style={{ ...st.statVal, fontSize: "1.1rem", marginTop: "4px" }}>
                        {kappaLabel(stats.overall.kappa)}
                      </div>
                      <div style={st.statLabel}>Cohen's κ (pooled)</div>
                    </div>
                    <div style={st.statBox}>
                      <div style={st.statVal}>{stats.overall.n_pairs}</div>
                      <div style={st.statLabel}>Annotation Pairs</div>
                    </div>
                  </div>

                  {/* Per-pair breakdown */}
                  {stats.pairs.length > 0 && (
                    <table style={{ ...st.table, marginTop: "10px" }}>
                      <thead>
                        <tr>
                          <th style={st.th}>Annotator 1</th>
                          <th style={st.th}>Annotator 2</th>
                          <th style={st.th}>N Tasks</th>
                          <th style={st.th}>% Agree</th>
                          <th style={st.th}>Cohen's κ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {stats.pairs.map((p) => (
                          <tr key={`${p.annotator_1}-${p.annotator_2}`}>
                            <td style={st.td}>{p.annotator_1}</td>
                            <td style={st.td}>{p.annotator_2}</td>
                            <td style={st.td}>{p.n}</td>
                            <td style={st.td}>{pct(p.pct_agree)}</td>
                            <td style={st.td}>{kappaLabel(p.kappa)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              );
            })}
          </>
        )}
      </div>

      {/* LLM Alignment */}
      <div style={st.section}>
        <div style={st.sectionTitle}>LLM Alignment (Human Majority = Ground Truth)</div>
        {data.llm_tasks_count === 0 ? (
          <div style={{ color: "#64748b", fontSize: "0.85rem" }}>
            No LLM annotations overlap with the current task manifest yet. Run the LLM annotation script on the valence manifest to populate this section.
          </div>
        ) : (
          <div style={{ display: "flex", gap: "32px", flexWrap: "wrap" as const }}>
            {(["narrative_valence", "speaker_valence"] as const).map((field) => {
              const al = llm_alignment[field];
              const label = field === "narrative_valence" ? "Narrative Valence" : "Speaker's Current Valence";
              return (
                <div key={field} style={{ flex: "1 1 400px" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#f8fafc", marginBottom: "8px" }}>
                    {label}
                  </div>
                  <div style={{ ...st.statGrid, marginBottom: "12px" }}>
                    <div style={st.statBox}>
                      <div style={st.statVal}>{pct(al.accuracy)}</div>
                      <div style={st.statLabel}>LLM Accuracy</div>
                    </div>
                    <div style={st.statBox}>
                      <div style={st.statVal}>{al.n}</div>
                      <div style={st.statLabel}>Comparable Tasks</div>
                    </div>
                  </div>
                  <ConfusionMatrix data={al.confusion_matrix} label={label} />
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Task table */}
      <div style={st.section}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
          <div style={st.sectionTitle} >Per-Task Results</div>
          <div style={st.filterRow}>
            {(["all", "annotated", "not_a_smile", "llm"] as const).map((f) => (
              <button key={f} onClick={() => setFilter(f)} style={{
                ...st.filterBtn,
                ...(filter === f ? st.filterBtnActive : {}),
              }}>
                {f === "all" ? "All" : f === "annotated" ? "Annotated" : f === "not_a_smile" ? "Not a Smile" : "Has LLM"}
              </button>
            ))}
            <span style={{ fontSize: "0.75rem", color: "#64748b", alignSelf: "center" }}>
              {filtered.length} tasks
            </span>
          </div>
        </div>

        {filtered.length === 0 ? (
          <div style={{ color: "#64748b", fontSize: "0.85rem", textAlign: "center", padding: "20px" }}>No tasks match this filter.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={st.table}>
              <thead>
                <tr>
                  <th style={st.th}>#</th>
                  <th style={st.th}>Video</th>
                  <th style={st.th}>Time</th>
                  <th style={st.th}>Score</th>
                  {annotators.map((a) => (
                    <>
                      <th key={`${a}-nv`} style={{ ...st.th, color: "#94a3b8" }}>{a} · Narrative</th>
                      <th key={`${a}-sv`} style={{ ...st.th, color: "#94a3b8" }}>{a} · Speaker</th>
                    </>
                  ))}
                  <th style={{ ...st.th, color: "#6366f1" }}>LLM · Narrative</th>
                  <th style={{ ...st.th, color: "#6366f1" }}>LLM · Speaker</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => {
                  const annByName = Object.fromEntries(r.human_annotations.map((a) => [a.annotator, a]));
                  return (
                    <tr key={r.task_number}>
                      <td style={{ ...st.td, color: "#64748b" }}>{r.task_number}</td>
                      <td style={{ ...st.td, fontVariantNumeric: "tabular-nums" }}>{r.video_id}</td>
                      <td style={{ ...st.td, color: "#64748b", fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" as const }}>
                        {fmtTime(r.smile_start)}–{fmtTime(r.smile_end)}
                      </td>
                      <td style={{ ...st.td, color: "#94a3b8" }}>{r.score?.toFixed(3) ?? "—"}</td>
                      {annotators.map((a) => {
                        const ann = annByName[a];
                        return (
                          <>
                            <td key={`${a}-nv`} style={st.td}>
                              {ann?.not_a_smile
                                ? <span style={st.notSmileBadge}>not a smile</span>
                                : <ValencePill v={ann?.narrative_valence} />}
                            </td>
                            <td key={`${a}-sv`} style={st.td}>
                              {ann?.not_a_smile
                                ? <span style={st.notSmileBadge}>not a smile</span>
                                : <ValencePill v={ann?.speaker_valence} />}
                            </td>
                          </>
                        );
                      })}
                      <td style={st.td}><ValencePill v={r.llm?.narrative_valence} /></td>
                      <td style={st.td}><ValencePill v={r.llm?.speaker_valence} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
