import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SmileAgreementStats } from "../types";
import { ALL_SMILE_LABELS } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const API = "/api";

const LABEL_DISPLAY: Record<string, string> = Object.fromEntries(
  ALL_SMILE_LABELS.map((x) => [x.key, x.display])
);
const LABEL_COLOR: Record<string, string> = {
  ...Object.fromEntries(ALL_SMILE_LABELS.map((x) => [x.key, x.color])),
  positive: "#60a5fa",
  masking: "#f59e0b",
  not_a_smile: "#64748b",
};
const COARSE_DISPLAY: Record<string, string> = {
  positive: "Positive (genuine + polite)",
  masking: "Masking",
  not_a_smile: "Not a smile",
};

const ANNOTATOR_PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"];

const KAPPA_BANDS = [
  { min: 0.8, label: "Almost perfect", color: "#22c55e" },
  { min: 0.6, label: "Substantial", color: "#84cc16" },
  { min: 0.4, label: "Moderate", color: "#eab308" },
  { min: 0.2, label: "Fair", color: "#f97316" },
  { min: -Infinity, label: "Slight / poor", color: "#f87171" },
];

function kappaColor(k: number | null): string {
  if (k == null) return "#64748b";
  const band = KAPPA_BANDS.find((b) => k >= b.min);
  return band?.color ?? "#f87171";
}

function kappaLabel(k: number | null): string {
  if (k == null) return "—";
  const band = KAPPA_BANDS.find((b) => k >= b.min);
  return band?.label ?? "Slight / poor";
}

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "16px 20px",
    maxWidth: "1100px",
    margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0",
    backgroundColor: "#0f172a",
    minHeight: "100vh",
  },
  top: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    flexWrap: "wrap" as const,
    gap: "12px",
    marginBottom: "20px",
  },
  h1: { fontSize: "1.35rem", fontWeight: 700, color: "#f8fafc", margin: 0 },
  sub: { fontSize: "0.8rem", color: "#94a3b8", marginTop: "4px" },
  nav: { display: "flex", gap: "8px" },
  btn: {
    padding: "6px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    border: "1px solid #475569",
    borderRadius: "6px",
    cursor: "pointer",
    backgroundColor: "#334155",
    color: "#e2e8f0",
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: "10px",
    padding: "16px 18px",
    marginBottom: "16px",
    border: "1px solid #334155",
  },
  cardTitle: {
    fontSize: "0.95rem",
    fontWeight: 600,
    color: "#f1f5f9",
    marginBottom: "14px",
  },
  toggles: { display: "flex", flexWrap: "wrap" as const, gap: "10px 16px" },
  toggle: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "0.85rem",
    color: "#cbd5e1",
    cursor: "pointer",
    userSelect: "none" as const,
  },
  statGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
    gap: "10px",
  },
  statBox: {
    backgroundColor: "#0f172a",
    borderRadius: "8px",
    padding: "12px 14px",
    border: "1px solid #334155",
  },
  statVal: { fontSize: "1.3rem", fontWeight: 700, color: "#f8fafc", lineHeight: 1.2 },
  statSub: { fontSize: "0.7rem", fontWeight: 400, color: "#64748b", display: "block" },
  statLab: { fontSize: "0.7rem", color: "#94a3b8", textTransform: "uppercase" as const, letterSpacing: "0.04em", marginTop: "4px" },
  segBtnBase: {
    padding: "4px 12px",
    fontSize: "0.78rem",
    fontWeight: 600,
    border: "1px solid #475569",
    borderRadius: "5px",
    cursor: "pointer",
  } as React.CSSProperties,
  err: { color: "#f87171", fontSize: "0.9rem" },
};

function fmtK(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}
function fmtPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v.toFixed(1)}%`;
}

function confusionMax(conf: number[][]): number {
  let m = 1;
  for (const row of conf) for (const c of row) m = Math.max(m, c);
  return m;
}

function cellBg(count: number, maxVal: number, isDiag: boolean): string {
  if (count === 0) return "rgba(15,23,42,0.9)";
  const t = maxVal > 0 ? count / maxVal : 0;
  const alpha = 0.2 + 0.6 * t;
  return isDiag
    ? `rgba(34,197,94,${alpha})`
    : `rgba(248,113,113,${alpha * 0.85})`;
}

interface KappaBarDatum {
  name: string;
  fine: number;
  coarse: number;
  fineRaw: number | null;
  coarseRaw: number | null;
  n: number;
}

export default function SmileAgreement() {
  const navigate = useNavigate();
  const me = localStorage.getItem(STORAGE_KEY);
  const [allNames, setAllNames] = useState<string[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [stats, setStats] = useState<SmileAgreementStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pairIdx, setPairIdx] = useState(0);
  const [confMode, setConfMode] = useState<"fine" | "coarse">("coarse");

  useEffect(() => {
    if (!me) navigate("/smile-login?next=/agreement", { replace: true });
  }, [me, navigate]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/smile-agreement/annotators`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const names: string[] = data.annotators ?? [];
        setAllNames(names);
        setSelected(Object.fromEntries(names.map((n) => [n, true])));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load annotators");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const activeList = useMemo(
    () => allNames.filter((n) => selected[n]),
    [allNames, selected]
  );

  const loadStats = useCallback(async () => {
    if (activeList.length === 0) {
      setStats(null);
      setError(null);
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(
        `${API}/smile-agreement/stats?annotators=${encodeURIComponent(activeList.join(","))}`
      );
      if (!res.ok) {
        const j = await res.json().catch(() => null);
        throw new Error(j?.detail ?? `HTTP ${res.status}`);
      }
      setStats(await res.json());
      setPairIdx(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load stats");
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, [activeList]);

  useEffect(() => {
    if (allNames.length === 0) return;
    void loadStats();
  }, [allNames.length, loadStats]);

  const toggle = (name: string) => setSelected((s) => ({ ...s, [name]: !s[name] }));

  const kappaBarData = useMemo((): KappaBarDatum[] => {
    if (!stats) return [];
    return stats.pairwise
      .filter((p) => p.n_tasks > 0)
      .map((p) => ({
        name: `${p.annotator_a} ↔ ${p.annotator_b}`,
        fine: p.cohen_kappa ?? 0,
        coarse: p.coarse_cohen_kappa ?? 0,
        fineRaw: p.cohen_kappa,
        coarseRaw: p.coarse_cohen_kappa,
        n: p.n_tasks,
      }));
  }, [stats]);

  // Stacked % bar data for annotator distributions
  const distData = useMemo(() => {
    if (!stats) return [];
    return stats.annotators.map((a) => {
      const counts = stats.per_annotator_counts[a] ?? {};
      const total = Object.values(counts).reduce((s, v) => s + v, 0) || 1;
      const row: Record<string, string | number> = { annotator: a };
      for (const lab of stats.valid_labels) {
        row[lab] = (((counts[lab] ?? 0) / total) * 100);
      }
      row["total"] = total;
      return row;
    });
  }, [stats]);

  const pair = stats?.pairwise[pairIdx];
  const activeConf = confMode === "coarse" ? pair?.coarse_confusion : pair?.confusion;
  const activeLabels = confMode === "coarse"
    ? (stats?.coarse_labels ?? [])
    : (stats?.valid_labels ?? []);
  const activeLabelDisplay = confMode === "coarse" ? COARSE_DISPLAY : LABEL_DISPLAY;
  const pairMax = activeConf ? confusionMax(activeConf) : 1;

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  };

  if (!me) return null;

  return (
    <div style={st.page}>
      <div style={st.top}>
        <div>
          <h1 style={st.h1}>Inter-annotator agreement</h1>
          <div style={st.sub}>
            Logged in as <strong style={{ color: "#e2e8f0" }}>{me}</strong>
          </div>
        </div>
        <div style={st.nav}>
          <button type="button" style={st.btn} onClick={() => navigate("/smile-annotate")}>
            Annotate
          </button>
          <button type="button" style={st.btn} onClick={() => void loadStats()} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button type="button" style={st.btn} onClick={logout}>Log out</button>
        </div>
      </div>

      {/* Annotator selector */}
      <div style={st.card}>
        <div style={st.cardTitle}>Annotators</div>
        <div style={st.toggles}>
          {allNames.map((n, i) => (
            <label key={n} style={st.toggle}>
              <input type="checkbox" checked={!!selected[n]} onChange={() => toggle(n)} />
              <span style={{ color: ANNOTATOR_PALETTE[i % ANNOTATOR_PALETTE.length], fontWeight: 600 }}>{n}</span>
            </label>
          ))}
          {allNames.length === 0 && !loading && (
            <span style={{ color: "#64748b", fontSize: "0.85rem" }}>No annotation files found.</span>
          )}
        </div>
      </div>

      {error && <div style={st.err}>{error}</div>}

      {stats && (
        <>
          {/* Summary */}
          <div style={st.card}>
            <div style={st.cardTitle}>Summary</div>
            <div style={st.statGrid}>
              <div style={st.statBox}>
                <div style={st.statLab}>Tasks annotated</div>
                <div style={st.statVal}>{stats.tasks_fully_labeled}
                  <span style={st.statSub}>of {stats.tasks_with_any_label} labeled</span>
                </div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Fleiss κ (4-class)</div>
                <div style={{ ...st.statVal, color: kappaColor(stats.fleiss_kappa) }}>
                  {fmtK(stats.fleiss_kappa)}
                  <span style={{ ...st.statSub, color: kappaColor(stats.fleiss_kappa) }}>{kappaLabel(stats.fleiss_kappa)}</span>
                </div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Fleiss κ (Pos+/Mask/No)</div>
                <div style={{ ...st.statVal, color: kappaColor(stats.coarse_fleiss_kappa) }}>
                  {fmtK(stats.coarse_fleiss_kappa)}
                  <span style={{ ...st.statSub, color: kappaColor(stats.coarse_fleiss_kappa) }}>{kappaLabel(stats.coarse_fleiss_kappa)}</span>
                </div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Unanimous (4-class)</div>
                <div style={st.statVal}>{fmtPct(stats.percent_full_agreement)}</div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Unanimous (3-class)</div>
                <div style={st.statVal}>{fmtPct(stats.coarse_percent_full_agreement)}</div>
              </div>
            </div>
            {stats.annotators.length < 2 && (
              <p style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "12px", marginBottom: 0 }}>
                Fleiss κ needs at least two annotators.
              </p>
            )}
          </div>

          {/* Label distribution: stacked % per annotator */}
          {stats.annotators.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Label distribution per annotator</div>
              <div style={{ width: "100%", height: Math.max(140, stats.annotators.length * 52 + 40) }}>
                <ResponsiveContainer>
                  <BarChart
                    layout="vertical"
                    data={distData}
                    margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical />
                    <XAxis
                      type="number"
                      domain={[0, 100]}
                      unit="%"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <YAxis
                      type="category"
                      dataKey="annotator"
                      width={90}
                      tick={{ fill: "#cbd5e1", fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }}
                      formatter={(value: number, name: string, props) => {
                        const tot = (props.payload as { total?: number }).total ?? 1;
                        return [`${value.toFixed(1)}%  (n=${Math.round(value * tot / 100)})`, LABEL_DISPLAY[name] ?? name];
                      }}
                    />
                    <Legend formatter={(key) => LABEL_DISPLAY[key] ?? key} />
                    {(stats.valid_labels as string[]).map((lab) => (
                      <Bar key={lab} dataKey={lab} stackId="a" fill={LABEL_COLOR[lab] ?? "#94a3b8"} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Pairwise κ comparison */}
          {kappaBarData.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Pairwise Cohen's κ — fine-grained vs. coarse (Positive / Masking / Not a smile)</div>
              <p style={{ fontSize: "0.8rem", color: "#94a3b8", margin: "0 0 12px" }}>
                Coarse κ merges genuine + polite into "Positive." A higher coarse κ means the pair's disagreement is mostly genuine↔polite. A lower coarse κ means deeper confusion.
              </p>
              <div style={{ width: "100%", height: Math.max(160, kappaBarData.length * 64 + 40) }}>
                <ResponsiveContainer>
                  <BarChart
                    layout="vertical"
                    data={kappaBarData}
                    margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical />
                    <XAxis type="number" domain={[-0.2, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={160} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                    <ReferenceLine x={0} stroke="#475569" strokeDasharray="4 2" />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }}
                      formatter={(_v: number, name: string, props) => {
                        const raw = name === "Fine (4-class)"
                          ? (props.payload as KappaBarDatum).fineRaw
                          : (props.payload as KappaBarDatum).coarseRaw;
                        return [raw != null ? `${raw.toFixed(3)} (${kappaLabel(raw)})` : "—", name];
                      }}
                      labelFormatter={(_, payload) => {
                        const p = (payload?.[0]?.payload as KappaBarDatum | undefined);
                        return p ? `Shared tasks: ${p.n}` : "";
                      }}
                    />
                    <Legend />
                    <Bar dataKey="fine" name="Fine (4-class)" radius={[0, 3, 3, 0]}>
                      {kappaBarData.map((d, i) => (
                        <Cell key={i} fill={kappaColor(d.fineRaw)} fillOpacity={0.7} />
                      ))}
                    </Bar>
                    <Bar dataKey="coarse" name="Coarse (3-class)" radius={[0, 3, 3, 0]}>
                      {kappaBarData.map((d, i) => (
                        <Cell key={i} fill={kappaColor(d.coarseRaw)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Kappa interpretation scale */}
              <div style={{ display: "flex", flexWrap: "wrap" as const, gap: "8px", marginTop: "14px" }}>
                {KAPPA_BANDS.map((b) => (
                  <span key={b.label} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "0.72rem", color: "#94a3b8" }}>
                    <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, backgroundColor: b.color }} />
                    {b.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Confusion matrix with fine/coarse toggle */}
          {stats.pairwise.length > 0 && (
            <div style={st.card}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px", flexWrap: "wrap" as const, gap: "8px" }}>
                <span style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>Confusion matrix</span>
                <div style={{ display: "flex", gap: "4px" }}>
                  <button type="button" style={{ ...st.segBtnBase, backgroundColor: confMode === "coarse" ? "#334155" : "transparent", color: confMode === "coarse" ? "#f1f5f9" : "#64748b" }} onClick={() => setConfMode("coarse")}>
                    3-class
                  </button>
                  <button type="button" style={{ ...st.segBtnBase, backgroundColor: confMode === "fine" ? "#334155" : "transparent", color: confMode === "fine" ? "#f1f5f9" : "#64748b" }} onClick={() => setConfMode("fine")}>
                    4-class
                  </button>
                </div>
              </div>

              <select
                style={{
                  marginBottom: "12px", padding: "6px 10px", borderRadius: "6px",
                  border: "1px solid #475569", backgroundColor: "#0f172a",
                  color: "#e2e8f0", fontSize: "0.85rem",
                }}
                value={pairIdx}
                onChange={(e) => setPairIdx(Number(e.target.value))}
              >
                {stats.pairwise.map((p, i) => (
                  <option key={i} value={i}>
                    {p.annotator_a} vs {p.annotator_b} — n={p.n_tasks}
                  </option>
                ))}
              </select>

              {pair && activeConf && (
                <>
                  <p style={{ fontSize: "0.78rem", color: "#64748b", margin: "0 0 10px" }}>
                    Rows: <strong style={{ color: "#94a3b8" }}>{pair.annotator_a}</strong> &nbsp;·&nbsp; Columns: <strong style={{ color: "#94a3b8" }}>{pair.annotator_b}</strong>
                  </p>
                  <div style={{ overflowX: "auto" as const }}>
                    <table style={{ borderCollapse: "collapse", fontSize: "0.8rem" }}>
                      <thead>
                        <tr>
                          <th style={{ padding: 6, color: "#64748b" }} />
                          {activeLabels.map((lab) => (
                            <th key={lab} style={{ padding: "6px 10px", color: "#94a3b8", fontWeight: 600, textAlign: "center" as const, maxWidth: 110, fontSize: "0.76rem" }}>
                              {activeLabelDisplay[lab] ?? lab}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {activeLabels.map((rowLab, ri) => (
                          <tr key={rowLab}>
                            <td style={{ padding: "6px 10px", color: "#94a3b8", fontWeight: 600, fontSize: "0.76rem", maxWidth: 110 }}>
                              {activeLabelDisplay[rowLab] ?? rowLab}
                            </td>
                            {activeLabels.map((colLab, ci) => {
                              const cnt = activeConf[ri]?.[ci] ?? 0;
                              return (
                                <td
                                  key={colLab}
                                  style={{
                                    padding: "10px 16px",
                                    textAlign: "center" as const,
                                    fontWeight: 700,
                                    fontSize: "1rem",
                                    color: "#f8fafc",
                                    backgroundColor: cellBg(cnt, pairMax, ri === ci),
                                    border: "1px solid #334155",
                                    minWidth: 56,
                                  }}
                                  title={`${pair.annotator_a}: ${rowLab}, ${pair.annotator_b}: ${colLab} → ${cnt}`}
                                >
                                  {cnt}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ marginTop: 12, fontSize: "0.8rem", color: "#94a3b8", display: "flex", gap: "16px", flexWrap: "wrap" as const }}>
                    <span>
                      Cohen κ ={" "}
                      <strong style={{ color: kappaColor(confMode === "coarse" ? pair.coarse_cohen_kappa : pair.cohen_kappa) }}>
                        {fmtK(confMode === "coarse" ? pair.coarse_cohen_kappa : pair.cohen_kappa)}
                      </strong>
                    </span>
                    <span>
                      Exact match ={" "}
                      <strong style={{ color: "#e2e8f0" }}>
                        {fmtPct(confMode === "coarse" ? pair.coarse_percent_agreement : pair.percent_agreement)}
                      </strong>
                    </span>
                    <span>n = {pair.n_tasks}</span>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}

      {!stats && !error && loading && allNames.length > 0 && (
        <p style={{ color: "#94a3b8" }}>Loading…</p>
      )}
    </div>
  );
}
