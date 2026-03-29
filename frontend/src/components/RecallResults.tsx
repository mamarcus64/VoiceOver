import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";

const STORAGE_KEY = "smile_annotator_name";
const API = "/api";
const OPERATING_THRESHOLD = 0.636;

interface BinStat {
  bin: number;
  score_min: number;
  score_max: number;
  population: number;
  sampled: number;
  labeled: number;
  smile_count: number;
  not_smile_count: number;
  smile_rate: number | null;
  passes_threshold: boolean;
}

interface AnnotatorCounts {
  smile: number;
  not_a_smile: number;
  total: number;
}

interface RecallResults {
  operating_threshold: number;
  annotators: string[];
  total_tasks: number;
  population_size: number;
  completed_tasks: number;
  bins: BinStat[];
  recall_estimate: number | null;
  recall_ci_low: number | null;
  recall_ci_high: number | null;
  ci_method: string | null;
  bins_with_data: number;
  min_bins_for_estimate: number;
  per_annotator_counts: Record<string, AnnotatorCounts>;
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
  statGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
    gap: "10px",
  },
  statBox: {
    backgroundColor: "#0f172a",
    borderRadius: "8px",
    padding: "12px 14px",
    border: "1px solid #334155",
  },
  statVal: { fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", lineHeight: 1.2 },
  statSub: { fontSize: "0.7rem", fontWeight: 400, color: "#64748b", display: "block" },
  statLab: {
    fontSize: "0.68rem",
    color: "#94a3b8",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
    marginTop: "4px",
  },
  err: { color: "#f87171", fontSize: "0.9rem" },
  note: {
    fontSize: "0.78rem",
    color: "#94a3b8",
    margin: "0 0 12px",
    lineHeight: 1.6,
  },
};

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtN(v: number): string {
  return v.toLocaleString();
}

const ANNOTATOR_PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"];

interface ChartRow {
  binLabel: string;
  smile_rate: number | null;
  labeled: number;
  sampled: number;
  population: number;
  passes_threshold: boolean;
  score_min: number;
  score_max: number;
}


export default function RecallResults() {
  const navigate = useNavigate();
  const me = localStorage.getItem(STORAGE_KEY);

  const [data, setData] = useState<RecallResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/recall-tasks/results`);
      if (!res.ok) {
        const j = await res.json().catch(() => null);
        throw new Error(j?.detail ?? `HTTP ${res.status}`);
      }
      setData(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load results");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!me) {
      navigate("/smile-login?next=/recall-results", { replace: true });
      return;
    }
    void load();
  }, [me, navigate, load]);

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  };

  if (!me) return null;

  const chartRows: ChartRow[] = (data?.bins ?? []).map((b) => ({
    binLabel: `${(b.score_min * 100).toFixed(0)}–${(b.score_max * 100).toFixed(0)}`,
    smile_rate: b.smile_rate,
    labeled: b.labeled,
    sampled: b.sampled,
    population: b.population,
    passes_threshold: b.passes_threshold,
    score_min: b.score_min,
    score_max: b.score_max,
  }));

  // Find where the threshold falls on the chart x-axis
  const thresholdBinIndex = data
    ? data.bins.findIndex(
        (b) => b.score_min < OPERATING_THRESHOLD && b.score_max >= OPERATING_THRESHOLD
      )
    : -1;

  const hasEstimate =
    data?.recall_estimate != null &&
    (data?.bins_with_data ?? 0) >= (data?.min_bins_for_estimate ?? 5);

  const completionPct =
    data && data.total_tasks > 0
      ? data.completed_tasks / data.total_tasks
      : null;

  return (
    <div style={st.page}>
      <div style={st.top}>
        <div>
          <h1 style={st.h1}>Recall Estimation Results</h1>
          <div style={st.sub}>
            {me ? (
              <>Logged in as <strong style={{ color: "#e2e8f0" }}>{me}</strong></>
            ) : null}
          </div>
        </div>
        <div style={st.nav}>
          <button type="button" style={st.btn} onClick={() => navigate("/recall-annotate")}>
            Annotate
          </button>
          <button
            type="button"
            style={st.btn}
            onClick={() => void load()}
            disabled={loading}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button type="button" style={st.btn} onClick={logout}>
            Log out
          </button>
        </div>
      </div>

      {error && <div style={st.err}>{error}</div>}

      {data && (
        <>
          {/* Summary stat cards */}
          <div style={st.card}>
            <div style={st.cardTitle}>Recall Estimate at θ = {OPERATING_THRESHOLD}</div>
            <div style={st.statGrid}>
              <div style={st.statBox}>
                <div style={st.statLab}>Recall (HT estimate)</div>
                <div
                  style={{
                    ...st.statVal,
                    fontSize: "2rem",
                    color: hasEstimate ? "#22c55e" : "#64748b",
                  }}
                >
                  {hasEstimate ? fmtPct(data.recall_estimate, 1) : "—"}
                  {hasEstimate &&
                    data.recall_ci_low != null &&
                    data.recall_ci_high != null && (
                      <span
                        style={{
                          ...st.statSub,
                          fontSize: "0.75rem",
                          color: "#94a3b8",
                        }}
                      >
                        95% CI:{" "}
                        {fmtPct(data.recall_ci_low, 1)}–
                        {fmtPct(data.recall_ci_high, 1)}
                        {data.ci_method ? ` (${data.ci_method.replace("_", " ")})` : ""}
                      </span>
                    )}
                </div>
                {!hasEstimate && (
                  <div style={{ ...st.statSub, fontSize: "0.72rem", color: "#f59e0b", marginTop: 4 }}>
                    Need ≥{data.min_bins_for_estimate} bins with ≥5 labels (
                    {data.bins_with_data} so far)
                  </div>
                )}
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Tasks labeled</div>
                <div style={st.statVal}>
                  {fmtN(data.completed_tasks)}
                  <span style={st.statSub}>
                    of {fmtN(data.total_tasks)} sampled
                    {completionPct != null
                      ? ` · ${fmtPct(completionPct, 0)}`
                      : ""}
                  </span>
                </div>
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Population</div>
                <div style={st.statVal}>
                  {fmtN(data.population_size)}
                  <span style={st.statSub}>segments (AU12 &gt; 1.0)</span>
                </div>
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Annotators</div>
                <div style={st.statVal}>
                  {data.annotators.length}
                  <span style={st.statSub}>
                    {data.annotators.join(", ") || "none yet"}
                  </span>
                </div>
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Bins with data</div>
                <div style={st.statVal}>
                  {data.bins_with_data}
                  <span style={st.statSub}>of 10 decile bins</span>
                </div>
              </div>
            </div>
          </div>

          {/* Per-bin smile rate chart */}
          <div style={st.card}>
            <div style={st.cardTitle}>Smile rate by logistic-score decile</div>
            <p style={st.note}>
              Each bar shows the fraction of labeled segments in that decile bin
              rated as a smile. Bins to the <strong style={{ color: "#3b82f6" }}>
              left of the dashed line</strong> (θ = {OPERATING_THRESHOLD}) are
              rejected by the model — these contain the false negatives that drive
              recall down. The recall estimate is the weighted fraction of true
              smiles that fall at or above θ.
            </p>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <BarChart
                  data={chartRows}
                  margin={{ top: 8, right: 24, left: 8, bottom: 32 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis
                    dataKey="binLabel"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    interval={0}
                    label={{
                      value: "Logistic score range (%)",
                      position: "insideBottom",
                      offset: -20,
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                    angle={-30}
                    textAnchor="end"
                    height={52}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                    tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    width={44}
                    label={{
                      value: "P(smile)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                  />
                  {thresholdBinIndex >= 0 && (
                    <ReferenceLine
                      x={chartRows[thresholdBinIndex]?.binLabel}
                      stroke="#f59e0b"
                      strokeDasharray="5 3"
                      strokeWidth={2}
                      label={{
                        value: `θ=${OPERATING_THRESHOLD}`,
                        position: "top",
                        fill: "#f59e0b",
                        fontSize: 11,
                      }}
                    />
                  )}
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "1px solid #475569",
                      borderRadius: 8,
                    }}
                    content={(props: TooltipProps<number, string>) => {
                      if (!props.payload?.length) return null;
                      const d = (props.payload[0] as { payload: ChartRow }).payload;
                      return (
                        <div
                          style={{
                            padding: "8px 12px",
                            fontSize: "0.78rem",
                            color: "#e2e8f0",
                          }}
                        >
                          <div
                            style={{
                              fontWeight: 600,
                              marginBottom: 5,
                              color: "#f1f5f9",
                            }}
                          >
                            Score {d.score_min.toFixed(3)}–{d.score_max.toFixed(3)}
                          </div>
                          <div>
                            Smile rate:{" "}
                            <strong>
                              {d.smile_rate != null
                                ? `${(d.smile_rate * 100).toFixed(1)}%`
                                : "no data"}
                            </strong>
                          </div>
                          <div style={{ color: "#94a3b8" }}>
                            Labeled: {d.labeled} / {d.sampled} sampled
                          </div>
                          <div style={{ color: "#64748b" }}>
                            Population: {fmtN(d.population)}
                          </div>
                          <div
                            style={{
                              color: d.passes_threshold ? "#22c55e" : "#64748b",
                              marginTop: 4,
                            }}
                          >
                            {d.passes_threshold
                              ? "✓ Above threshold (TP region)"
                              : "✗ Below threshold (FN region)"}
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="smile_rate" name="Smile rate" radius={[3, 3, 0, 0]}>
                    {chartRows.map((row, i) => (
                      <Cell
                        key={i}
                        fill={
                          row.labeled === 0
                            ? "#1e293b"
                            : row.passes_threshold
                            ? "#22c55e"
                            : "#3b82f6"
                        }
                        fillOpacity={row.labeled === 0 ? 0.3 : 0.85}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div
              style={{
                display: "flex",
                gap: "16px",
                marginTop: "8px",
                fontSize: "0.74rem",
                color: "#94a3b8",
              }}
            >
              {[
                { color: "#3b82f6", label: "Below threshold (FN region)" },
                { color: "#22c55e", label: "Above threshold (TP region)" },
                { color: "#1e293b", label: "Not yet labeled" },
              ].map(({ color, label }) => (
                <span
                  key={label}
                  style={{ display: "flex", alignItems: "center", gap: 5 }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      backgroundColor: color,
                      border: "1px solid #475569",
                    }}
                  />
                  {label}
                </span>
              ))}
            </div>
          </div>

          {/* Per-annotator progress table */}
          {data.annotators.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Per-annotator progress</div>
              <div style={{ overflowX: "auto" as const }}>
                <table
                  style={{
                    borderCollapse: "collapse",
                    fontSize: "0.82rem",
                    width: "100%",
                  }}
                >
                  <thead>
                    <tr>
                      {["Annotator", "Labeled", "Smile", "Not a Smile", "% Smile"].map(
                        (h) => (
                          <th
                            key={h}
                            style={{
                              padding: "6px 12px",
                              textAlign: "left" as const,
                              color: "#94a3b8",
                              fontWeight: 600,
                              borderBottom: "1px solid #334155",
                              fontSize: "0.72rem",
                              textTransform: "uppercase" as const,
                              letterSpacing: "0.04em",
                            }}
                          >
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {data.annotators.map((name, i) => {
                      const c = data.per_annotator_counts[name] ?? {
                        smile: 0,
                        not_a_smile: 0,
                        total: 0,
                      };
                      const smilePct =
                        c.total > 0 ? c.smile / c.total : null;
                      return (
                        <tr key={name}>
                          <td
                            style={{
                              padding: "8px 12px",
                              fontWeight: 600,
                              color:
                                ANNOTATOR_PALETTE[i % ANNOTATOR_PALETTE.length],
                            }}
                          >
                            {name}
                          </td>
                          <td style={{ padding: "8px 12px", color: "#e2e8f0" }}>
                            {c.total}
                            <span style={{ color: "#475569", fontSize: "0.7rem" }}>
                              {" "}
                              / {data.total_tasks}
                            </span>
                          </td>
                          <td
                            style={{
                              padding: "8px 12px",
                              color: "#22c55e",
                              fontWeight: 600,
                            }}
                          >
                            {c.smile}
                          </td>
                          <td style={{ padding: "8px 12px", color: "#64748b" }}>
                            {c.not_a_smile}
                          </td>
                          <td
                            style={{
                              padding: "8px 12px",
                              color: "#f1f5f9",
                              fontWeight: 600,
                            }}
                          >
                            {smilePct != null
                              ? `${(smilePct * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Method note */}
          <div style={{ ...st.card, borderColor: "#1e3a5f" }}>
            <div style={{ ...st.cardTitle, color: "#93c5fd" }}>
              Methodology
            </div>
            <p style={st.note}>
              <strong style={{ color: "#e2e8f0" }}>Population:</strong> All AU12&#8202;&gt;&#8202;1.0
              candidate segments from Stage 1 extraction (194,670 total).
              Stratified into 10 equal-count decile bins by 17-AU logistic
              score; 75 segments sampled per bin (750 total tasks, seed 42).
              Tasks were globally shuffled and presented blinded (no score or bin
              shown).
            </p>
            <p style={{ ...st.note, marginBottom: 0 }}>
              <strong style={{ color: "#e2e8f0" }}>Estimator:</strong> Horvitz–Thompson
              weighted recall at θ&#8202;=&#8202;{OPERATING_THRESHOLD}:{" "}
              <em>
                R̂ = Σ&#8202;(N&#8202;/&#8202;n&#8202;·&#8202;Σ&#8202;1[s&#8202;≥&#8202;θ]&#8202;·&#8202;y) &#8202;/&#8202; Σ&#8202;(N&#8202;/&#8202;n&#8202;·&#8202;Σ&#8202;y)
              </em>
              . Confidence intervals from stratified bootstrap (10,000 resamples,
              2.5th–97.5th percentile). Recall is undefined until at least{" "}
              {data.min_bins_for_estimate} bins each have ≥5 labeled tasks.
            </p>
          </div>
        </>
      )}

      {!data && !error && loading && (
        <p style={{ color: "#94a3b8" }}>Loading…</p>
      )}
    </div>
  );
}
