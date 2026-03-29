import { useEffect, useMemo, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, CartesianGrid,
} from "recharts";

const API = "/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface InterviewSummary {
  int_code: number;
  name: string | null;
  gender: string | null;
  label: string | null;
  country_of_birth: string | null;
  city_of_birth: string | null;
  interview_date: string | null;
  interview_length: string | null;
  num_segments: number;
}

interface GroupStats {
  total: number;
  gender_counts: Record<string, number>;
  label_counts: Record<string, number>;
  country_counts: Record<string, number>;
  city_counts: Record<string, number>;
  keyword_freq: Record<string, number>;
  interview_years: Record<string, number>;
  interviews: InterviewSummary[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
  "#06b6d4", "#84cc16", "#f97316", "#a78bfa", "#34d399",
];

function toChartData(obj: Record<string, number>, limit?: number) {
  const entries = Object.entries(obj).sort((a, b) => b[1] - a[1]);
  const sliced = limit ? entries.slice(0, limit) : entries;
  return sliced.map(([name, value]) => ({ name, value }));
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { value: number }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      backgroundColor: "#1e293b",
      border: "1px solid #334155",
      borderRadius: 6,
      padding: "6px 12px",
      fontSize: 12,
      color: "#e2e8f0",
    }}>
      <div style={{ color: "#94a3b8", marginBottom: 2 }}>{label}</div>
      <div style={{ fontWeight: 600 }}>{payload[0].value.toLocaleString()}</div>
    </div>
  );
};

// ── StatCard ──────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }: {
  label: string; value: string; sub?: string; color?: string;
}) {
  return (
    <div style={st.card}>
      <div style={{ ...st.cardValue, color: color ?? "#f8fafc" }}>{value}</div>
      <div style={st.cardLabel}>{label}</div>
      {sub && <div style={st.cardSub}>{sub}</div>}
    </div>
  );
}

// ── ChartSection ──────────────────────────────────────────────────────────────

function ChartSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={st.chartSection}>
      <div style={st.chartTitle}>{title}</div>
      {children}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function TranscriptGroupView() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<GroupStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<keyof InterviewSummary>("int_code");
  const [sortDir, setSortDir] = useState<1 | -1>(1);
  const [labelFilter, setLabelFilter] = useState<string>("all");
  const [genderFilter, setGenderFilter] = useState<string>("all");

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API}/metadata/group-stats`);
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        setStats(await res.json());
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filteredInterviews = useMemo(() => {
    if (!stats) return [];
    let list = stats.interviews;
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        i =>
          i.name?.toLowerCase().includes(q) ||
          String(i.int_code).includes(q) ||
          i.country_of_birth?.toLowerCase().includes(q) ||
          i.city_of_birth?.toLowerCase().includes(q) ||
          i.label?.toLowerCase().includes(q)
      );
    }
    if (labelFilter !== "all") list = list.filter(i => i.label === labelFilter);
    if (genderFilter !== "all") list = list.filter(i => i.gender === genderFilter);
    return [...list].sort((a, b) => {
      const av = a[sortKey] ?? "";
      const bv = b[sortKey] ?? "";
      if (av < bv) return -sortDir;
      if (av > bv) return sortDir;
      return 0;
    });
  }, [stats, search, sortKey, sortDir, labelFilter, genderFilter]);

  function toggleSort(key: keyof InterviewSummary) {
    if (sortKey === key) setSortDir(d => (d === 1 ? -1 : 1));
    else { setSortKey(key); setSortDir(1); }
  }

  const SortTh = ({ col, label }: { col: keyof InterviewSummary; label: string }) => (
    <th
      style={{ ...st.th, cursor: "pointer", userSelect: "none" }}
      onClick={() => toggleSort(col)}
    >
      {label}{sortKey === col ? (sortDir === 1 ? " ▲" : " ▼") : ""}
    </th>
  );

  if (loading) return (
    <div style={st.page}>
      <div style={st.msg}>
        Loading group statistics across all interviews…
        <div style={{ fontSize: 12, color: "#475569", marginTop: 8 }}>
          (Parsing ~1000 XML files — may take a few seconds)
        </div>
      </div>
    </div>
  );

  if (error) return (
    <div style={st.page}>
      <div style={{ ...st.msg, color: "#f87171" }}>{error}</div>
    </div>
  );

  if (!stats) return null;

  const yearData = toChartData(stats.interview_years);
  const countryData = toChartData(stats.country_counts, 25);
  const keywordData = toChartData(stats.keyword_freq, 30);
  const labelData = toChartData(stats.label_counts);
  const genderData = toChartData(stats.gender_counts);

  const uniqueLabels = Object.keys(stats.label_counts);
  const uniqueGenders = Object.keys(stats.gender_counts);

  const avgSegments = stats.interviews.length
    ? (stats.interviews.reduce((s, i) => s + i.num_segments, 0) / stats.interviews.length).toFixed(1)
    : "—";

  return (
    <div style={st.page}>
      {/* Header */}
      <div style={st.header}>
        <button style={st.backBtn} onClick={() => navigate("/")}>← Home</button>
        <h1 style={st.title}>VHA Testimony Archive — Group View</h1>
        <div style={{ marginLeft: "auto", fontSize: 12, color: "#64748b" }}>
          {stats.total.toLocaleString()} subjects · hover charts for details
        </div>
      </div>

      {/* Top stat cards */}
      <div style={st.cardRow}>
        <StatCard label="Subjects" value={stats.total.toLocaleString()} color="#3b82f6"
          sub="unique survivors/witnesses" />
        {genderData.map((g, i) => (
          <StatCard key={g.name} label={g.name} value={g.value.toLocaleString()}
            sub={`${((g.value / stats.total) * 100).toFixed(1)}%`}
            color={COLORS[(i + 1) % COLORS.length]} />
        ))}
        <StatCard label="Avg. VHA Segments / Subject" value={avgSegments} color="#f59e0b"
          sub="1-min indexed segments" />
        <StatCard label="Unique Keywords" value={Object.keys(stats.keyword_freq).length.toLocaleString()} color="#a78bfa" />
      </div>

      {/* Charts row 1 */}
      <div style={st.chartsRow}>
        <ChartSection title="Interviews by Year">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={yearData} margin={{ top: 4, right: 8, bottom: 24, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }}
                angle={-45} textAnchor="end" interval={2} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={36} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                {yearData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartSection>

        <ChartSection title="Interviewee Classification">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={labelData} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 140 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} width={140} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                {labelData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartSection>
      </div>

      {/* Country distribution */}
      <ChartSection title="Country of Birth (Top 25)">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={countryData} margin={{ top: 4, right: 8, bottom: 60, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }}
              angle={-40} textAnchor="end" interval={0} />
            <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={40} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" radius={[2, 2, 0, 0]}>
              {countryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Top Keywords */}
      <ChartSection title="Top 30 VHA Segment Keywords (across all testimonies)">
        <ResponsiveContainer width="100%" height={460}>
          <BarChart data={keywordData} layout="vertical"
            margin={{ top: 4, right: 60, bottom: 4, left: 280 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} width={280} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" radius={[0, 2, 2, 0]}>
              {keywordData.map((_, i) => <Cell key={i} fill={COLORS[i % 5]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Interview table */}
      <div style={st.chartSection}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
          <div style={st.chartTitle}>All Subjects</div>
          <span style={{ fontSize: 11, color: "#475569" }}>
            Each row = one survivor/witness. "Total Length" is the full interview across all tapes.
            Click "View" to open that subject's transcript with tape navigation.
          </span>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search name, code, country…"
            style={st.searchInput}
          />
          <select value={labelFilter} onChange={e => setLabelFilter(e.target.value)} style={st.select}>
            <option value="all">All labels</option>
            {uniqueLabels.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
          <select value={genderFilter} onChange={e => setGenderFilter(e.target.value)} style={st.select}>
            <option value="all">All genders</option>
            {uniqueGenders.map(g => <option key={g} value={g}>{g}</option>)}
          </select>
          <span style={{ color: "#64748b", fontSize: 12, marginLeft: "auto" }}>
            {filteredInterviews.length.toLocaleString()} shown
          </span>
        </div>

        <div style={{ overflowX: "auto", borderRadius: 8, border: "1px solid #334155" }}>
          <table style={st.table}>
            <thead>
              <tr>
                <SortTh col="int_code" label="Code" />
                <SortTh col="name" label="Name" />
                <SortTh col="gender" label="Gender" />
                <SortTh col="label" label="Classification" />
                <SortTh col="country_of_birth" label="Country" />
                <SortTh col="city_of_birth" label="City" />
                <SortTh col="interview_date" label="Date" />
                <SortTh col="interview_length" label="Total Length" />
                <SortTh col="num_segments" label="VHA Segs" />
                <th style={st.th}>View</th>
              </tr>
            </thead>
            <tbody>
              {filteredInterviews.slice(0, 500).map((iv) => (
                <tr key={iv.int_code} style={st.tr}>
                  <td style={{ ...st.td, color: "#64748b", fontVariantNumeric: "tabular-nums" }}>{iv.int_code}</td>
                  <td style={{ ...st.td, fontWeight: 500, color: "#f8fafc" }}>{iv.name ?? "—"}</td>
                  <td style={{ ...st.td, color: "#94a3b8" }}>{iv.gender ?? "—"}</td>
                  <td style={{ ...st.td, color: "#94a3b8", fontSize: 11 }}>{iv.label ?? "—"}</td>
                  <td style={st.td}>{iv.country_of_birth ?? "—"}</td>
                  <td style={st.td}>{iv.city_of_birth ?? "—"}</td>
                  <td style={{ ...st.td, fontVariantNumeric: "tabular-nums", color: "#64748b", fontSize: 12 }}>
                    {iv.interview_date ?? "—"}
                  </td>
                  <td style={{ ...st.td, fontVariantNumeric: "tabular-nums", color: "#64748b", fontSize: 12 }}>
                    {iv.interview_length ?? "—"}
                  </td>
                  <td style={{ ...st.td, textAlign: "center", color: "#94a3b8" }}>{iv.num_segments}</td>
                  <td style={st.td}>
                    <Link to={`/transcript/${iv.int_code}`} style={st.viewLink}>
                      View →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredInterviews.length > 500 && (
            <div style={{ padding: "10px 16px", color: "#64748b", fontSize: 12 }}>
              Showing first 500 of {filteredInterviews.length} results. Refine search to see more.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "20px 24px",
    maxWidth: 1600,
    margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    marginBottom: 20,
    flexWrap: "wrap",
  },
  backBtn: {
    padding: "8px 14px",
    backgroundColor: "#1e293b",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  title: {
    fontSize: "1.4rem",
    fontWeight: 700,
    color: "#f8fafc",
    margin: 0,
  },
  cardRow: {
    display: "flex",
    gap: 12,
    flexWrap: "wrap",
    marginBottom: 24,
  },
  card: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 10,
    padding: "12px 18px",
    minWidth: 130,
    flex: "1 1 130px",
  },
  cardValue: {
    fontSize: "1.5rem",
    fontWeight: 700,
    color: "#f8fafc",
  },
  cardLabel: {
    fontSize: "0.75rem",
    color: "#64748b",
    marginTop: 3,
  },
  cardSub: {
    fontSize: "0.7rem",
    color: "#475569",
    marginTop: 1,
  },
  chartsRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 16,
    marginBottom: 16,
  },
  chartSection: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 10,
    padding: "16px 20px",
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: 12,
  },
  searchInput: {
    padding: "7px 12px",
    fontSize: "0.9rem",
    borderRadius: 6,
    border: "1px solid #334155",
    backgroundColor: "#0f172a",
    color: "#f8fafc",
    minWidth: 240,
    outline: "none",
  },
  select: {
    padding: "7px 10px",
    fontSize: "0.85rem",
    borderRadius: 6,
    border: "1px solid #334155",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    cursor: "pointer",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.875rem",
  },
  th: {
    textAlign: "left",
    padding: "10px 14px",
    fontWeight: 600,
    color: "#64748b",
    fontSize: "0.8rem",
    borderBottom: "1px solid #334155",
    backgroundColor: "#0f172a",
    whiteSpace: "nowrap",
  },
  td: {
    padding: "10px 14px",
    borderBottom: "1px solid #1e293b",
    color: "#cbd5e1",
  },
  tr: {
    transition: "background-color 0.1s ease",
  },
  viewLink: {
    color: "#3b82f6",
    textDecoration: "none",
    fontSize: 12,
    fontWeight: 500,
  },
  msg: {
    textAlign: "center",
    padding: "60px",
    color: "#64748b",
    fontSize: "1rem",
  },
};
