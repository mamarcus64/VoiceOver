import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SmileAgreementStats } from "../types";
import { SMILE_LABELS } from "../types";

const STORAGE_KEY = "smile_annotator_name";
const API = "/api";

const LABEL_DISPLAY: Record<string, string> = Object.fromEntries(
  SMILE_LABELS.map((x) => [x.key, x.display])
);

const ANNOTATOR_PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#f87171"];

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "16px 20px",
    maxWidth: "1200px",
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
  nav: { display: "flex", gap: "8px", flexWrap: "wrap" as const },
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
    marginBottom: "12px",
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
    gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
    gap: "12px",
  },
  statBox: {
    backgroundColor: "#0f172a",
    borderRadius: "8px",
    padding: "12px",
    border: "1px solid #334155",
  },
  statVal: { fontSize: "1.25rem", fontWeight: 700, color: "#f8fafc" },
  statLab: { fontSize: "0.7rem", color: "#94a3b8", textTransform: "uppercase" as const, letterSpacing: "0.04em" },
  select: {
    marginBottom: "12px",
    padding: "8px 10px",
    borderRadius: "6px",
    border: "1px solid #475569",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    fontSize: "0.85rem",
    maxWidth: "100%",
  },
  heatWrap: { overflowX: "auto" as const },
  err: { color: "#f87171", fontSize: "0.9rem" },
};

function fmtKappa(v: number | null | undefined): string {
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
  const alpha = 0.25 + 0.55 * t;
  if (isDiag) return `rgba(34,197,94,${alpha})`;
  return `rgba(248,113,113,${alpha * 0.85})`;
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

  useEffect(() => {
    if (!me) {
      navigate("/smile-login?next=/agreement", { replace: true });
    }
  }, [me, navigate]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/smile-agreement/annotators`);
        if (!res.ok) throw new Error(`Annotators ${res.status}`);
        const data = await res.json();
        const names: string[] = data.annotators ?? [];
        setAllNames(names);
        const init: Record<string, boolean> = {};
        for (const n of names) init[n] = true;
        setSelected(init);
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
      const q = encodeURIComponent(activeList.join(","));
      const res = await fetch(`${API}/smile-agreement/stats?annotators=${q}`);
      if (!res.ok) {
        const j = await res.json().catch(() => null);
        throw new Error(j?.detail ?? `Stats ${res.status}`);
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

  const toggle = (name: string) => {
    setSelected((s) => ({ ...s, [name]: !s[name] }));
  };

  const selectAll = () => {
    const next: Record<string, boolean> = {};
    for (const n of allNames) next[n] = true;
    setSelected(next);
  };

  const selectNone = () => {
    const next: Record<string, boolean> = {};
    for (const n of allNames) next[n] = false;
    setSelected(next);
  };

  const distributionData = useMemo(() => {
    if (!stats) return [];
    return stats.valid_labels.map((lab) => {
      const row: Record<string, string | number> = {
        key: lab,
        short: LABEL_DISPLAY[lab] ?? lab,
      };
      for (const a of stats.annotators) {
        row[a] = stats.per_annotator_counts[a]?.[lab] ?? 0;
      }
      return row;
    });
  }, [stats]);

  const kappaBarData = useMemo(() => {
    if (!stats) return [];
    return stats.pairwise
      .filter((p) => p.n_tasks > 0)
      .map((p) => ({
        name: `${p.annotator_a} ↔ ${p.annotator_b}`,
        kappa: p.cohen_kappa != null ? Math.max(-1, Math.min(1, p.cohen_kappa)) : 0,
        kappaRaw: p.cohen_kappa,
        n: p.n_tasks,
      }));
  }, [stats]);

  const pair = stats?.pairwise[pairIdx];
  const pairMax = pair ? confusionMax(pair.confusion) : 1;

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
            Logged in as <strong style={{ color: "#e2e8f0" }}>{me}</strong> · Smile labels across selected annotators
          </div>
        </div>
        <div style={st.nav}>
          <button type="button" style={st.btn} onClick={() => navigate("/smile-annotate")}>
            Annotate
          </button>
          <button type="button" style={st.btn} onClick={() => void loadStats()} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button type="button" style={st.btn} onClick={logout}>
            Log out
          </button>
        </div>
      </div>

      <div style={st.card}>
        <div style={st.cardTitle}>Annotators</div>
        <p style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: 0, marginBottom: "12px" }}>
          Uncheck anyone to exclude them from agreement metrics. Distribution counts still reflect each person’s total labels.
        </p>
        <div style={{ ...st.toggles, marginBottom: "12px" }}>
          <button type="button" style={st.btn} onClick={selectAll}>
            All
          </button>
          <button type="button" style={st.btn} onClick={selectNone}>
            None
          </button>
        </div>
        <div style={st.toggles}>
          {allNames.map((n) => (
            <label key={n} style={st.toggle}>
              <input
                type="checkbox"
                checked={!!selected[n]}
                onChange={() => toggle(n)}
              />
              {n}
            </label>
          ))}
        </div>
        {allNames.length === 0 && !loading && (
          <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>No annotation files found under data/smile_annotations.</p>
        )}
      </div>

      {error && <div style={st.err}>{error}</div>}

      {stats && (
        <>
          <div style={st.card}>
            <div style={st.cardTitle}>Summary</div>
            <div style={st.statGrid}>
              <div style={st.statBox}>
                <div style={st.statLab}>Tasks (any label)</div>
                <div style={st.statVal}>{stats.tasks_with_any_label}</div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Fully labeled</div>
                <div style={st.statVal}>{stats.tasks_fully_labeled}</div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Fleiss κ</div>
                <div style={st.statVal}>{fmtKappa(stats.fleiss_kappa)}</div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Unanimous (full overlap)</div>
                <div style={st.statVal}>{fmtPct(stats.percent_full_agreement)}</div>
              </div>
            </div>
            {stats.annotators.length < 2 && (
              <p style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: "12px", marginBottom: 0 }}>
                Fleiss κ and unanimous rate need at least two annotators selected. Pairwise metrics appear below when two or more overlap on tasks.
              </p>
            )}
          </div>

          <div style={st.card}>
            <div style={st.cardTitle}>Labels per annotator</div>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={distributionData} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="short" tick={{ fill: "#94a3b8", fontSize: 11 }} interval={0} angle={-18} textAnchor="end" height={70} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }}
                    labelStyle={{ color: "#f1f5f9" }}
                  />
                  <Legend />
                  {stats.annotators.map((a, i) => (
                    <Bar
                      key={a}
                      dataKey={a}
                      name={a}
                      fill={ANNOTATOR_PALETTE[i % ANNOTATOR_PALETTE.length]}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {kappaBarData.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Pairwise Cohen&apos;s κ (ordered tasks)</div>
              <div style={{ width: "100%", height: Math.max(200, kappaBarData.length * 44) }}>
                <ResponsiveContainer>
                  <BarChart layout="vertical" data={kappaBarData} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" domain={[-1, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={200} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: 8 }}
                      formatter={(_value: number, _n, props) => {
                        const raw = (props.payload as { kappaRaw: number | null }).kappaRaw;
                        return [raw != null ? raw.toFixed(3) : "—", "κ"];
                      }}
                      labelFormatter={(_, payload) => {
                        const p = payload?.[0]?.payload as { n?: number } | undefined;
                        return p?.n != null ? `Overlapping tasks: ${p.n}` : "";
                      }}
                    />
                    <Bar dataKey="kappa" name="Cohen κ" radius={[0, 4, 4, 0]}>
                      {kappaBarData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={
                            entry.kappaRaw == null
                              ? "#64748b"
                              : entry.kappaRaw >= 0.6
                                ? "#22c55e"
                                : entry.kappaRaw >= 0.4
                                  ? "#eab308"
                                  : "#f87171"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {stats.pairwise.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Pairwise confusion matrix</div>
              <p style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: 0, marginBottom: "8px" }}>
                Rows: first annotator, columns: second. Diagonal counts are agreements.
              </p>
              <select
                style={st.select}
                value={pairIdx}
                onChange={(e) => setPairIdx(Number(e.target.value))}
              >
                {stats.pairwise.map((p, i) => (
                  <option key={i} value={i}>
                    {p.annotator_a} vs {p.annotator_b} ({p.n_tasks} tasks)
                  </option>
                ))}
              </select>
              {pair && (
                <div style={st.heatWrap}>
                  <table style={{ borderCollapse: "collapse", fontSize: "0.8rem" }}>
                    <thead>
                      <tr>
                        <th style={{ padding: 6, color: "#64748b" }} />
                        {stats.valid_labels.map((lab) => (
                          <th key={lab} style={{ padding: 6, color: "#94a3b8", fontWeight: 600, maxWidth: 100 }}>
                            {LABEL_DISPLAY[lab] ?? lab}
                            <div style={{ fontSize: "0.65rem", fontWeight: 400, color: "#64748b" }}>{pair.annotator_b}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {stats.valid_labels.map((rowLab, ri) => (
                        <tr key={rowLab}>
                          <td style={{ padding: 6, color: "#94a3b8", fontWeight: 600, maxWidth: 120, verticalAlign: "middle" }}>
                            <div>{LABEL_DISPLAY[rowLab] ?? rowLab}</div>
                            <div style={{ fontSize: "0.65rem", fontWeight: 400, color: "#64748b" }}>{pair.annotator_a}</div>
                          </td>
                          {stats.valid_labels.map((colLab, ci) => {
                            const cnt = pair.confusion[ri]?.[ci] ?? 0;
                            const isDiag = ri === ci;
                            return (
                              <td
                                key={colLab}
                                style={{
                                  padding: 10,
                                  textAlign: "center" as const,
                                  fontWeight: 600,
                                  color: "#f8fafc",
                                  backgroundColor: cellBg(cnt, pairMax, isDiag),
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
                  <div style={{ marginTop: 12, fontSize: "0.8rem", color: "#94a3b8" }}>
                    Cohen κ = {fmtKappa(pair.cohen_kappa)} · Exact match = {fmtPct(pair.percent_agreement)} · n = {pair.n_tasks}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {!stats && !error && loading && allNames.length > 0 && (
        <p style={{ color: "#94a3b8" }}>Loading agreement…</p>
      )}
    </div>
  );
}
