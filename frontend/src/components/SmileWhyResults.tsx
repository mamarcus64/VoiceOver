import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const API = "/api";

interface ResultEntry {
  task_number: number;
  annotator: string;
  response: string;
  not_a_smile: boolean;
  timestamp: string;
  video_id: string;
  smile_start: number | null;
  smile_end: number | null;
  stratum: string;
  score_tier: string;
}

interface ResultsData {
  total_annotations: number;
  total_tasks: number;
  results: ResultEntry[];
}

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "24px 32px", maxWidth: "1100px", margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0", backgroundColor: "#0f172a", minHeight: "100vh",
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    marginBottom: "20px", flexWrap: "wrap" as const, gap: "10px",
  },
  title: { fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc" },
  stats: { fontSize: "0.85rem", color: "#94a3b8" },
  backLink: {
    fontSize: "0.85rem", color: "#93c5fd", textDecoration: "none",
    padding: "6px 14px", border: "1px solid #475569", borderRadius: "6px",
    backgroundColor: "#1e293b",
  },
  card: {
    padding: "14px 18px", marginBottom: "10px",
    backgroundColor: "#1e293b", borderRadius: "10px",
    border: "1px solid #334155",
  },
  cardHeader: {
    display: "flex", alignItems: "center", gap: "12px", marginBottom: "6px",
    fontSize: "0.8rem", color: "#94a3b8", flexWrap: "wrap" as const,
  },
  response: { fontSize: "0.9rem", color: "#e2e8f0", lineHeight: 1.5 },
  notSmile: { fontSize: "0.85rem", color: "#64748b", fontStyle: "italic" },
  badge: {
    display: "inline-block", padding: "1px 7px",
    borderRadius: "4px", fontSize: "0.7rem", fontWeight: 600,
  },
  empty: { textAlign: "center" as const, padding: "60px", color: "#64748b", fontSize: "1rem" },
  loading: { textAlign: "center" as const, padding: "60px", color: "#64748b", fontSize: "1rem" },
};

function fmtTime(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return "--";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function SmileWhyResults() {
  const [data, setData] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "responses" | "not_a_smile">("all");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/smile-why-results`);
        if (res.ok) setData(await res.json());
      } catch { /* ignore */ }
      setLoading(false);
    })();
  }, []);

  if (loading) return <div style={st.page}><div style={st.loading}>Loading results...</div></div>;

  const results = data?.results ?? [];
  const filtered = filter === "all"
    ? results
    : filter === "responses"
      ? results.filter((r) => r.response && !r.not_a_smile)
      : results.filter((r) => r.not_a_smile);

  const annotators = [...new Set(results.map((r) => r.annotator))];

  return (
    <div style={st.page}>
      <div style={st.header}>
        <div>
          <div style={st.title}>Smile-Why Annotation Results</div>
          <div style={st.stats}>
            {data?.total_annotations ?? 0} annotations across {annotators.length} annotator{annotators.length !== 1 ? "s" : ""}
            {" "}/ {data?.total_tasks ?? 0} total tasks
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {(["all", "responses", "not_a_smile"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)} style={{
              padding: "5px 12px", fontSize: "0.8rem", fontWeight: 600,
              border: "1px solid #475569", borderRadius: "5px", cursor: "pointer",
              backgroundColor: filter === f ? "#3b82f6" : "#334155",
              color: filter === f ? "#fff" : "#94a3b8",
            }}>
              {f === "all" ? "All" : f === "responses" ? "Responses" : "Not a smile"}
            </button>
          ))}
          <Link to="/smile-why-annotate" style={st.backLink}>Back to Annotation</Link>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div style={st.empty}>No annotations yet.</div>
      ) : (
        filtered.map((r, i) => (
          <div key={`${r.task_number}-${r.annotator}-${i}`} style={st.card}>
            <div style={st.cardHeader}>
              <span style={{ fontWeight: 700, color: "#f8fafc" }}>Task {r.task_number}</span>
              <span>Video {r.video_id}</span>
              <span>{fmtTime(r.smile_start)}-{fmtTime(r.smile_end)}</span>
              <span style={{ ...st.badge, backgroundColor: r.stratum === "narrative" ? "#1e3a5f" : "#3b2f1e", color: r.stratum === "narrative" ? "#93c5fd" : "#fbbf24" }}>
                {r.stratum}
              </span>
              <span style={{ ...st.badge, backgroundColor: "#1e293b", border: "1px solid #475569", color: "#94a3b8" }}>
                {r.score_tier}
              </span>
              <span style={{ marginLeft: "auto", fontSize: "0.7rem", color: "#64748b" }}>
                {r.annotator} &middot; {r.timestamp ? new Date(r.timestamp).toLocaleString() : ""}
              </span>
            </div>
            {r.not_a_smile ? (
              <div style={st.notSmile}>Subject was not smiling</div>
            ) : (
              <div style={st.response}>{r.response}</div>
            )}
          </div>
        ))
      )}
    </div>
  );
}
