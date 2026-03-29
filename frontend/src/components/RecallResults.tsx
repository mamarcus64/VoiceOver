import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";

const STORAGE_KEY = "smile_annotator_name";
const API = "/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type ModelId = "logistic" | "au12";

interface PRPoint {
  threshold: number;
  recall: number | null;
  precision: number | null;
  recall_ci_low: number | null;
  recall_ci_high: number | null;
  precision_ci_low: number | null;
  precision_ci_high: number | null;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

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

interface ModelResults {
  model: ModelId;
  cache_key: string;
  computed_at: string;
  is_cached: boolean;
  score_key: string;
  operating_threshold: number;
  score_range: [number, number];
  sources: { recall_manifest: number; main_study: number; pilot_study: number };
  total_labeled: number;
  total_recall_tasks: number;
  population_size: number;
  pr_curve: PRPoint[];
  bins: BinStat[];
  annotators: string[];
  per_annotator_counts: Record<string, AnnotatorCounts>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}
function fmtN(v: number): string {
  return v.toLocaleString();
}
function fmtScore(v: number, model: ModelId): string {
  return model === "au12" ? v.toFixed(3) : v.toFixed(3);
}

function findNearest<T extends { threshold: number }>(arr: T[], t: number): T | null {
  if (!arr.length) return null;
  return arr.reduce((best, p) =>
    Math.abs(p.threshold - t) < Math.abs(best.threshold - t) ? p : best
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

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

const ANNOTATOR_PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"];

const MODEL_META: Record<ModelId, { label: string; scoreLabel: string }> = {
  logistic: { label: "17-AU Logistic", scoreLabel: "Logistic score" },
  au12: { label: "AU12 Threshold", scoreLabel: "AU12 mean_r" },
};

// ── PR Curve Tooltip ──────────────────────────────────────────────────────────

function PRTooltip({ active, payload, model }: TooltipProps<number, string> & { model: ModelId }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as PRPoint;
  const recallVal = d.recall ?? null;
  const precVal = d.precision ?? null;
  return (
    <div
      style={{
        backgroundColor: "#1e293b",
        border: "1px solid #475569",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: "0.78rem",
        color: "#e2e8f0",
        minWidth: 180,
      }}
    >
      <div style={{ fontWeight: 600, color: "#f1f5f9", marginBottom: 6 }}>
        {MODEL_META[model].scoreLabel} = {d.threshold.toFixed(3)}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <span style={{ color: "#38bdf8" }}>
          Recall:{" "}
          <strong>
            {recallVal != null ? fmtPct(recallVal) : "—"}
          </strong>
          {d.recall_ci_low != null && (
            <span style={{ color: "#64748b" }}>
              {" "}
              [{fmtPct(d.recall_ci_low)}–{fmtPct(d.recall_ci_high)}]
            </span>
          )}
        </span>
        <span style={{ color: "#34d399" }}>
          Precision:{" "}
          <strong>
            {precVal != null ? fmtPct(precVal) : "—"}
          </strong>
          {d.precision_ci_low != null && (
            <span style={{ color: "#64748b" }}>
              {" "}
              [{fmtPct(d.precision_ci_low)}–{fmtPct(d.precision_ci_high)}]
            </span>
          )}
        </span>

        <span style={{ color: "#64748b", marginTop: 4, fontSize: "0.72rem" }}>
          TP={d.tp} FP={d.fp} FN={d.fn} TN={d.tn}
        </span>
        {d.tp != null && (
          <span style={{ color: "#64748b", fontSize: "0.72rem" }}>
            F1={d.tp > 0 ? fmtPct((2 * d.tp) / (2 * d.tp + d.fp + d.fn)) : "—"}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Bin chart row type ────────────────────────────────────────────────────────

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

// ── Main component ────────────────────────────────────────────────────────────

export default function RecallResults() {
  const navigate = useNavigate();
  const me = localStorage.getItem(STORAGE_KEY);

  const [activeModel, setActiveModel] = useState<ModelId>("logistic");
  const [data, setData] = useState<ModelResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.636);

  const load = useCallback(
    async (model: ModelId) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API}/recall-tasks/results?model=${model}`);
        if (!res.ok) {
          const j = await res.json().catch(() => null);
          throw new Error((j as { detail?: string } | null)?.detail ?? `HTTP ${res.status}`);
        }
        const d: ModelResults = await res.json();
        setData(d);
        setThreshold(d.operating_threshold);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load results");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (!me) {
      navigate("/smile-login?next=/recall-results", { replace: true });
      return;
    }
    void load(activeModel);
  }, [me, navigate, load, activeModel]);

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    navigate("/smile-login", { replace: true });
  };

  if (!me) return null;

  // ── Derived data at current threshold ──────────────────────────────────────

  const mergedCurve = useMemo(() => data?.pr_curve ?? [], [data]);

  const currentPR = useMemo(
    () => (data ? findNearest(data.pr_curve, threshold) : null),
    [data, threshold]
  );

  const f1 =
    currentPR && currentPR.tp != null
      ? (2 * currentPR.tp) / (2 * currentPR.tp + currentPR.fp + currentPR.fn) || null
      : null;

  // ── Bin chart ──────────────────────────────────────────────────────────────

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

  const LOGISTIC_OP_THRESHOLD = 0.636;
  const thresholdBinIndex = data
    ? data.bins.findIndex(
        (b) => b.score_min < LOGISTIC_OP_THRESHOLD && b.score_max >= LOGISTIC_OP_THRESHOLD
      )
    : -1;

  const completionPct =
    data && data.total_recall_tasks > 0
      ? (data.sources.recall_manifest / data.total_recall_tasks)
      : null;

  // ── Score range for slider ─────────────────────────────────────────────────

  const scoreMin = data?.score_range[0] ?? 0;
  const scoreMax = data?.score_range[1] ?? 1;
  const sliderStep = (scoreMax - scoreMin) / 400;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={st.page}>
      {/* Header */}
      <div style={st.top}>
        <div>
          <h1 style={st.h1}>Recall Estimation Results</h1>
          <div style={st.sub}>
            {me && (
              <>
                Logged in as <strong style={{ color: "#e2e8f0" }}>{me}</strong>
              </>
            )}
          </div>
        </div>
        <div style={st.nav}>
          <button type="button" style={st.btn} onClick={() => navigate("/recall-annotate")}>
            Annotate
          </button>
          <button
            type="button"
            style={st.btn}
            onClick={() => void load(activeModel)}
            disabled={loading}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button type="button" style={st.btn} onClick={logout}>
            Log out
          </button>
        </div>
      </div>

      {/* Model selector */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
        {(["logistic", "au12"] as ModelId[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setActiveModel(m)}
            style={{
              padding: "7px 16px",
              fontSize: "0.82rem",
              fontWeight: 600,
              border: `1px solid ${activeModel === m ? "#3b82f6" : "#475569"}`,
              borderRadius: "6px",
              cursor: "pointer",
              backgroundColor: activeModel === m ? "#1d4ed8" : "#1e293b",
              color: activeModel === m ? "#eff6ff" : "#94a3b8",
              transition: "all 0.15s",
            }}
          >
            {MODEL_META[m].label}
          </button>
        ))}
        {data && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: "0.72rem",
              color: data.is_cached ? "#22c55e" : "#f59e0b",
              alignSelf: "center",
            }}
          >
            {data.is_cached ? "✓ cached" : "⟳ freshly computed"} ·{" "}
            {new Date(data.computed_at).toLocaleTimeString()}
          </span>
        )}
      </div>

      {error && <div style={st.err}>{error}</div>}

      {data && (
        <>
          {/* ── PR Curve + Threshold Slider ── */}
          <div style={st.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 14,
              }}
            >
              <div style={st.cardTitle} title="Threshold slider below controls the vertical line">
                Precision / Recall vs Threshold
              </div>
              {/* Inline threshold control */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  backgroundColor: "#0f172a",
                  padding: "6px 12px",
                  borderRadius: 8,
                  border: "1px solid #334155",
                }}
              >
                <span style={{ fontSize: "0.74rem", color: "#94a3b8" }}>θ&nbsp;=</span>
                <input
                  type="range"
                  min={scoreMin}
                  max={scoreMax}
                  step={sliderStep}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  style={{ width: 160, accentColor: "#f59e0b", cursor: "pointer" }}
                />
                <span
                  style={{
                    fontSize: "0.82rem",
                    fontWeight: 700,
                    color: "#f59e0b",
                    minWidth: 44,
                  }}
                >
                  {fmtScore(threshold, activeModel)}
                </span>
                <button
                  type="button"
                  onClick={() => setThreshold(data.operating_threshold)}
                  title="Reset to operating threshold"
                  style={{
                    fontSize: "0.7rem",
                    padding: "2px 7px",
                    border: "1px solid #475569",
                    borderRadius: 4,
                    backgroundColor: "#334155",
                    color: "#94a3b8",
                    cursor: "pointer",
                  }}
                >
                  reset
                </button>
              </div>
            </div>

            {/* P/R curve chart */}
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <ComposedChart
                  data={mergedCurve}
                  margin={{ top: 8, right: 24, left: 4, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={true} />
                  <XAxis
                    dataKey="threshold"
                    type="number"
                    domain={[scoreMin, scoreMax]}
                    tickFormatter={(v: number) => v.toFixed(2)}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    label={{
                      value: MODEL_META[activeModel].scoreLabel,
                      position: "insideBottom",
                      offset: -4,
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                    height={36}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                    tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    width={40}
                  />
                  {/* Recall (all data) */}
                  <Line
                    dataKey="recall"
                    name="Recall (all data)"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  {/* Precision */}
                  <Line
                    dataKey="precision"
                    name="Precision"
                    stroke="#34d399"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  {/* Current threshold marker */}
                  <ReferenceLine
                    x={threshold}
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    label={{
                      value: `θ=${threshold.toFixed(2)}`,
                      position: "top",
                      fill: "#f59e0b",
                      fontSize: 10,
                    }}
                  />
                  {/* Operating threshold marker */}
                  {Math.abs(threshold - data.operating_threshold) > 0.005 && (
                    <ReferenceLine
                      x={data.operating_threshold}
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="2 4"
                      label={{
                        value: `op.θ=${data.operating_threshold}`,
                        position: "insideTopRight",
                        fill: "#475569",
                        fontSize: 9,
                      }}
                    />
                  )}
                  <Tooltip
                    content={
                      (props: TooltipProps<number, string>) => (
                        <PRTooltip {...props} model={activeModel} />
                      )
                    }
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Chart legend */}
            <div
              style={{
                display: "flex",
                gap: 16,
                marginTop: 8,
                fontSize: "0.74rem",
                color: "#94a3b8",
                flexWrap: "wrap",
              }}
            >
              {[
                { color: "#38bdf8", dash: false, label: "Recall (all data)" },
                { color: "#34d399", dash: false, label: "Precision (all data)" },
                { color: "#f59e0b", dash: true, label: "Current threshold" },
              ].map(({ color, dash, label }) => (
                <span key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <svg width="18" height="4">
                    <line
                      x1="0"
                      y1="2"
                      x2="18"
                      y2="2"
                      stroke={color}
                      strokeWidth={2}
                      strokeDasharray={dash ? "4 2" : "none"}
                    />
                  </svg>
                  {label}
                </span>
              ))}
            </div>
          </div>

          {/* ── Stats at selected threshold ── */}
          <div style={st.card}>
            <div style={st.cardTitle}>
              At θ = {fmtScore(threshold, activeModel)}
              {" · "}
              <span style={{ fontWeight: 400, color: "#94a3b8", fontSize: "0.82rem" }}>
                {MODEL_META[activeModel].label}
              </span>
            </div>
            <div style={st.statGrid}>
              <div style={{ ...st.statBox, borderColor: "#38bdf8" + "40" }}>
                <div style={st.statLab}>Recall (all data)</div>
                <div style={{ ...st.statVal, fontSize: "1.8rem", color: "#38bdf8" }}>
                  {fmtPct(currentPR?.recall)}
                </div>
                {currentPR?.recall_ci_low != null && (
                  <div style={{ ...st.statSub, marginTop: 3 }}>
                    95% CI: {fmtPct(currentPR.recall_ci_low)}–{fmtPct(currentPR.recall_ci_high)}
                  </div>
                )}
              </div>

              <div style={{ ...st.statBox, borderColor: "#34d399" + "40" }}>
                <div style={st.statLab}>Precision (all data)</div>
                <div style={{ ...st.statVal, fontSize: "1.8rem", color: "#34d399" }}>
                  {fmtPct(currentPR?.precision)}
                </div>
                {currentPR?.precision_ci_low != null && (
                  <div style={{ ...st.statSub, marginTop: 3 }}>
                    95% CI: {fmtPct(currentPR.precision_ci_low)}–{fmtPct(currentPR.precision_ci_high)}
                  </div>
                )}
              </div>

              <div style={{ ...st.statBox, borderColor: "#a78bfa40" }}>
                <div style={st.statLab}>F1 score</div>
                <div style={{ ...st.statVal, fontSize: "1.8rem", color: "#a78bfa" }}>
                  {fmtPct(f1 ?? null)}
                </div>
                {currentPR && (
                  <div style={{ ...st.statSub, marginTop: 3 }}>
                    TP={currentPR.tp} FP={currentPR.fp} FN={currentPR.fn}
                  </div>
                )}
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Total labeled</div>
                <div style={st.statVal}>{fmtN(data.total_labeled)}</div>
                <div style={{ ...st.statSub, marginTop: 3 }}>
                  across all sources
                </div>
              </div>

              <div style={st.statBox}>
                <div style={st.statLab}>Population</div>
                <div style={st.statVal}>{fmtN(data.population_size)}</div>
                <div style={{ ...st.statSub, marginTop: 3 }}>segments (AU12 &gt; 1.0)</div>
              </div>
            </div>
          </div>

          {/* ── Data sources ── */}
          <div style={st.card}>
            <div style={st.cardTitle}>Data sources</div>
            <div style={st.statGrid}>
              <div style={st.statBox}>
                <div style={st.statLab}>Recall manifest</div>
                <div style={st.statVal}>{fmtN(data.sources.recall_manifest)}</div>
                <div style={{ ...st.statSub, marginTop: 3 }}>
                  {fmtPct(
                    data.total_recall_tasks > 0
                      ? data.sources.recall_manifest / data.total_recall_tasks
                      : null
                  )}{" "}
                  of {fmtN(data.total_recall_tasks)} tasks · stratified sample
                </div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Main study</div>
                <div style={st.statVal}>{fmtN(data.sources.main_study)}</div>
                <div style={{ ...st.statSub, marginTop: 3 }}>
                  high-score bias (logistic ≥ 0.636)
                </div>
              </div>
              {activeModel === "au12" && (
                <div style={st.statBox}>
                  <div style={st.statLab}>Pilot study</div>
                  <div style={st.statVal}>{fmtN(data.sources.pilot_study)}</div>
                  <div style={{ ...st.statSub, marginTop: 3 }}>
                    high-score bias (AU12 ≥ 1.5) · included for AU12 only
                  </div>
                </div>
              )}
              <div style={st.statBox}>
                <div style={st.statLab}>Annotators (recall)</div>
                <div style={st.statVal}>{data.annotators.length}</div>
                <div style={{ ...st.statSub, marginTop: 3 }}>
                  {data.annotators.join(", ") || "none yet"}
                </div>
              </div>
              <div style={st.statBox}>
                <div style={st.statLab}>Recall task progress</div>
                <div style={st.statVal}>
                  {completionPct != null ? fmtPct(completionPct, 0) : "—"}
                </div>
                <div style={{ ...st.statSub, marginTop: 3 }}>
                  {fmtN(data.sources.recall_manifest)} / {fmtN(data.total_recall_tasks)} labeled
                </div>
              </div>
            </div>
          </div>

          {/* ── Per-bin smile rate chart (recall manifest, logistic score deciles) ── */}
          <div style={st.card}>
            <div style={st.cardTitle}>
              Smile rate by logistic-score decile
              <span
                style={{
                  fontWeight: 400,
                  color: "#64748b",
                  fontSize: "0.78rem",
                  marginLeft: 8,
                }}
              >
                (recall manifest · {fmtN(data.sources.recall_manifest)} labeled)
              </span>
            </div>
            <p style={st.note}>
              Fraction of recall-manifest segments labeled "smile" per logistic-score decile bin.
              Bins right of the dashed line (θ = {LOGISTIC_OP_THRESHOLD}) pass the logistic
              operating threshold. The false-negative mass lives in the left bins.
            </p>
            <div style={{ width: "100%", height: 300 }}>
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
                        value: `θ=${LOGISTIC_OP_THRESHOLD}`,
                        position: "top",
                        fill: "#f59e0b",
                        fontSize: 11,
                      }}
                    />
                  )}
                  <Tooltip
                    content={(props: TooltipProps<number, string>) => {
                      if (!props.payload?.length) return null;
                      const d = (props.payload[0] as { payload: ChartRow }).payload;
                      return (
                        <div
                          style={{
                            padding: "8px 12px",
                            fontSize: "0.78rem",
                            color: "#e2e8f0",
                            backgroundColor: "#1e293b",
                            border: "1px solid #475569",
                            borderRadius: 8,
                          }}
                        >
                          <div style={{ fontWeight: 600, marginBottom: 5, color: "#f1f5f9" }}>
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
                          <div style={{ color: d.passes_threshold ? "#22c55e" : "#64748b", marginTop: 4 }}>
                            {d.passes_threshold
                              ? "✓ Above logistic threshold (TP region)"
                              : "✗ Below logistic threshold (FN region)"}
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
                <span key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
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

          {/* ── Per-annotator table ── */}
          {data.annotators.length > 0 && (
            <div style={st.card}>
              <div style={st.cardTitle}>Per-annotator progress (recall tasks)</div>
              <div style={{ overflowX: "auto" as const }}>
                <table
                  style={{ borderCollapse: "collapse", fontSize: "0.82rem", width: "100%" }}
                >
                  <thead>
                    <tr>
                      {["Annotator", "Labeled", "Smile", "Not a Smile", "% Smile"].map((h) => (
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
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.annotators.map((name, i) => {
                      const c = data.per_annotator_counts[name] ?? {
                        smile: 0,
                        not_a_smile: 0,
                        total: 0,
                      };
                      const smilePct = c.total > 0 ? c.smile / c.total : null;
                      return (
                        <tr key={name}>
                          <td
                            style={{
                              padding: "8px 12px",
                              fontWeight: 600,
                              color: ANNOTATOR_PALETTE[i % ANNOTATOR_PALETTE.length],
                            }}
                          >
                            {name}
                          </td>
                          <td style={{ padding: "8px 12px", color: "#e2e8f0" }}>
                            {c.total}
                            <span style={{ color: "#475569", fontSize: "0.7rem" }}>
                              {" "}
                              / {data.total_recall_tasks}
                            </span>
                          </td>
                          <td style={{ padding: "8px 12px", color: "#22c55e", fontWeight: 600 }}>
                            {c.smile}
                          </td>
                          <td style={{ padding: "8px 12px", color: "#64748b" }}>
                            {c.not_a_smile}
                          </td>
                          <td style={{ padding: "8px 12px", color: "#f1f5f9", fontWeight: 600 }}>
                            {smilePct != null ? `${(smilePct * 100).toFixed(1)}%` : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Methodology ── */}
          <div style={{ ...st.card, borderColor: "#1e3a5f" }}>
            <div style={{ ...st.cardTitle, color: "#93c5fd" }}>Methodology</div>
            <p style={st.note}>
              <strong style={{ color: "#e2e8f0" }}>Recall manifest (stratified):</strong> All
              AU12 &gt; 1.0 candidate segments from Stage 1 (194,670 total). Stratified into 10
              equal-count decile bins by 17-AU logistic score; 75 segments per bin (750 total,
              seed 42). Tasks presented blinded and shuffled.
            </p>
            <p style={st.note}>
              <strong style={{ color: "#e2e8f0" }}>PR curve:</strong> Empirical precision
              and recall using all labeled data (recall manifest + main study
              {activeModel === "au12" ? " + pilot study" : ""}). Main/pilot data are
              biased toward high scores; the recall manifest provides stratified coverage
              across the full confidence range. Wilson 95% CIs per threshold.
            </p>
            <p style={{ ...st.note, marginBottom: 0 }}>
              <strong style={{ color: "#e2e8f0" }}>Cache:</strong> Results are computed on first
              load and cached to disk. The cache is automatically invalidated whenever any
              annotation file is updated.
            </p>
          </div>
        </>
      )}

      {!data && !error && loading && (
        <p style={{ color: "#94a3b8" }}>
          Computing results… (first load may take a moment while caches are built)
        </p>
      )}
    </div>
  );
}
