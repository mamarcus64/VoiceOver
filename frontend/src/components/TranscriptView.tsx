import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import type { Utterance } from "../types";

const API = "/api";

// ── types ────────────────────────────────────────────────────────────────────

interface VhaSegment {
  number: number;
  in_time: string;
  out_time: string;
  in_time_sec: number;
  out_time_sec: number;
  in_file: number;
  out_file: number;
  tape_id: string;
  people: string[];
  keywords: string[];
}

interface Metadata {
  int_code: number;
  name: string | null;
  first_name: string | null;
  last_name: string | null;
  gender: string | null;
  label: string | null;
  dob: string | null;
  questionnaire: Record<string, string | string[]>;
  interview_refs: Record<string, string>;
  relations: { modifier: string; name: string; piq_id: string }[];
  segments: VhaSegment[];
}

// ── helpers ───────────────────────────────────────────────────────────────────

function fmtTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtDob(dob: string | null): string {
  if (!dob) return "—";
  const [y, m, d] = dob.split("/");
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const mn = parseInt(m) - 1;
  return `${months[mn] ?? m} ${d}, ${y}`;
}

function toList(val: string | string[] | undefined): string[] {
  if (!val) return [];
  return Array.isArray(val) ? val : [val];
}

// ── SpeechTimeline ────────────────────────────────────────────────────────────

interface TimelineProps {
  utterances: Utterance[];
  segments: VhaSegment[];
  tapeNumber: number;
  durationMs: number;
}

function SpeechTimeline({ utterances, segments, tapeNumber, durationMs }: TimelineProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const W = 900;
  const LANE_H = 28;
  const LABEL_W = 100;
  const MARKER_ZONE_H = 30;
  const PAD_TOP = 8;
  const PAD_BOT = 8;
  const H = PAD_TOP + LANE_H + 6 + LANE_H + 10 + MARKER_ZONE_H + PAD_BOT;
  const DRAW_W = W - LABEL_W;

  const dur = durationMs || 1;
  const toX = (ms: number) => LABEL_W + (ms / dur) * DRAW_W;

  // Filter VHA segments for this tape
  const tapeSegments = segments.filter(
    (s) => s.in_file === tapeNumber || s.out_file === tapeNumber
  );

  const interviewerY = PAD_TOP;
  const intervieweeY = PAD_TOP + LANE_H + 6;
  const markerY = intervieweeY + LANE_H + 10;

  return (
    <div style={{ position: "relative", overflowX: "auto" }}>
      <svg
        ref={svgRef}
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ display: "block", borderRadius: 8, backgroundColor: "#0f172a" }}
        onMouseLeave={() => setTooltip(null)}
      >
        {/* Lane backgrounds */}
        <rect x={LABEL_W} y={interviewerY} width={DRAW_W} height={LANE_H} fill="#1e293b" rx={3} />
        <rect x={LABEL_W} y={intervieweeY} width={DRAW_W} height={LANE_H} fill="#1e293b" rx={3} />

        {/* Lane labels */}
        <text x={LABEL_W - 6} y={interviewerY + LANE_H / 2 + 4} textAnchor="end"
          fill="#94a3b8" fontSize={11} fontFamily="system-ui">Interviewer</text>
        <text x={LABEL_W - 6} y={intervieweeY + LANE_H / 2 + 4} textAnchor="end"
          fill="#94a3b8" fontSize={11} fontFamily="system-ui">Interviewee</text>

        {/* Utterance bars */}
        {utterances.map((u, i) => {
          const x = toX(u.start_ms);
          const w = Math.max(1, toX(u.end_ms) - x);
          const y = u.speaker === "interviewer" ? interviewerY : intervieweeY;
          const color = u.speaker === "interviewer" ? "#3b82f6" : "#10b981";
          const alpha = u.type === "non_verbal" ? "80" : "cc";
          return (
            <rect
              key={i}
              x={x} y={y + 2} width={w} height={LANE_H - 4}
              fill={`${color}${alpha}`}
              rx={1}
              style={{ cursor: "pointer" }}
              onMouseEnter={(e) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  const px = e.clientX - rect.left;
                  const py = e.clientY - rect.top;
                  const preview = u.text.length > 120 ? u.text.slice(0, 120) + "…" : u.text;
                  setTooltip({ x: px, y: py, text: `[${fmtTime(u.start_ms)}] ${preview}` });
                }
              }}
            />
          );
        })}

        {/* VHA segment markers (1-min boundaries) */}
        {tapeSegments.map((seg) => {
          if (!seg.keywords.length && !seg.people.length) return null;
          const mx = toX(seg.in_time_sec * 1000);
          return (
            <g key={seg.number}>
              <line x1={mx} y1={interviewerY} x2={mx} y2={markerY + MARKER_ZONE_H}
                stroke="#f59e0b44" strokeWidth={1} strokeDasharray="3,2" />
              <rect
                x={mx - 10} y={markerY + 2} width={20} height={MARKER_ZONE_H - 4}
                fill="#f59e0b22" rx={3} stroke="#f59e0b66" strokeWidth={1}
                style={{ cursor: "pointer" }}
                onMouseEnter={(e) => {
                  const rect = svgRef.current?.getBoundingClientRect();
                  if (rect) {
                    const px = e.clientX - rect.left;
                    const py = e.clientY - rect.top;
                    const kws = seg.keywords.slice(0, 5).join(", ");
                    const ppl = seg.people.slice(0, 3).join(", ");
                    const lines = [
                      `Min ${seg.number} (${seg.in_time} → ${seg.out_time})`,
                      kws && `Keywords: ${kws}`,
                      ppl && `People: ${ppl}`,
                    ].filter(Boolean).join("\n");
                    setTooltip({ x: px, y: py, text: lines });
                  }
                }}
              />
              <text x={mx} y={markerY + MARKER_ZONE_H / 2 + 4}
                textAnchor="middle" fill="#f59e0b99" fontSize={9} fontFamily="system-ui">
                {seg.number}
              </text>
            </g>
          );
        })}

        {/* Minute tick marks at bottom */}
        {Array.from({ length: Math.ceil(dur / 60000) + 1 }, (_, i) => {
          const x = toX(i * 60000);
          if (x < LABEL_W || x > W) return null;
          return (
            <g key={i}>
              <line x1={x} y1={markerY + MARKER_ZONE_H} x2={x} y2={markerY + MARKER_ZONE_H + 4}
                stroke="#475569" strokeWidth={1} />
              {i % 5 === 0 && (
                <text x={x} y={markerY + MARKER_ZONE_H + 14}
                  textAnchor="middle" fill="#64748b" fontSize={9} fontFamily="system-ui">
                  {i}m
                </text>
              )}
            </g>
          );
        })}

        {/* Label "Keywords" */}
        <text x={LABEL_W - 6} y={markerY + MARKER_ZONE_H / 2 + 4} textAnchor="end"
          fill="#64748b" fontSize={10} fontFamily="system-ui">Segments</text>
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: "absolute",
          left: tooltip.x + 12,
          top: tooltip.y - 8,
          backgroundColor: "#1e293b",
          border: "1px solid #334155",
          borderRadius: 6,
          padding: "6px 10px",
          fontSize: 12,
          color: "#e2e8f0",
          maxWidth: 320,
          pointerEvents: "none",
          zIndex: 10,
          whiteSpace: "pre-line",
          lineHeight: 1.5,
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
        }}>
          {tooltip.text}
        </div>
      )}

      {/* Legend */}
      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: "#64748b" }}>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#3b82fcc0", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />Interviewer speech</span>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#10b981cc", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />Interviewee speech</span>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#f59e0b22", border: "1px solid #f59e0b66", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />VHA 1-min segment (hover for keywords)</span>
      </div>
    </div>
  );
}

// ── BioPanel ──────────────────────────────────────────────────────────────────

function BioPanel({ meta }: { meta: Metadata }) {
  const q = meta.questionnaire;
  const refs = meta.interview_refs;

  const camps = toList(q["Camp(s)"]);
  const ghettos = toList(q["Ghetto(s)"]);

  return (
    <div style={st.bioPanel}>
      <div style={st.bioName}>{meta.name ?? "—"}</div>
      <div style={st.bioSubtitle}>{meta.label ?? ""}</div>

      <div style={st.bioGrid}>
        <BioRow label="Gender" value={meta.gender} />
        <BioRow label="Date of Birth" value={fmtDob(meta.dob)} />
        <BioRow label="City of Birth" value={q["City of Birth"] as string} />
        <BioRow label="Country of Birth" value={q["Country of Birth"] as string} />
        <BioRow label="Interview Date" value={refs["Date of Interview"]} />
        <BioRow label="Interview Length" value={refs["Length of Interview"]} />
        <BioRow label="Language" value={refs["Language(s) of Interview"]} />
        <BioRow label="Location" value={refs["State of Interview"] ? `${refs["State of Interview"]}, ${refs["Country of Interview"] ?? ""}` : refs["Country of Interview"]} />
        <BioRow label="Historic Event" value={refs["Historic Event"]} />
      </div>

      {ghettos.length > 0 && (
        <div style={st.bioSection}>
          <div style={st.bioSectionTitle}>Ghettos</div>
          {ghettos.map((g, i) => <div key={i} style={st.bioTag}>{g}</div>)}
        </div>
      )}

      {camps.length > 0 && (
        <div style={st.bioSection}>
          <div style={st.bioSectionTitle}>Camps</div>
          {camps.map((c, i) => <div key={i} style={st.bioTag}>{c}</div>)}
        </div>
      )}

      {(q["Liberated by"] || q["Location of Liberation"]) && (
        <div style={st.bioSection}>
          <div style={st.bioSectionTitle}>Liberation</div>
          {q["Liberated by"] && <BioRow label="By" value={q["Liberated by"] as string} />}
          {q["Location of Liberation"] && <BioRow label="At" value={q["Location of Liberation"] as string} />}
        </div>
      )}
    </div>
  );
}

function BioRow({ label, value }: { label: string; value: string | undefined | null }) {
  if (!value) return null;
  return (
    <div style={st.bioRow}>
      <span style={st.bioLabel}>{label}</span>
      <span style={st.bioValue}>{value}</span>
    </div>
  );
}

// ── SegmentPanel ──────────────────────────────────────────────────────────────

function SegmentPanel({ segments, tapeNumber }: { segments: VhaSegment[]; tapeNumber: number }) {
  const tapeSegs = segments.filter(s => s.in_file === tapeNumber || s.out_file === tapeNumber);
  if (!tapeSegs.length) {
    return (
      <div style={st.segPanel}>
        <div style={st.segTitle}>Segment Keywords</div>
        <div style={{ color: "#64748b", fontSize: 12 }}>No VHA segments for this tape.</div>
      </div>
    );
  }
  return (
    <div style={st.segPanel}>
      <div style={st.segTitle}>Segment Keywords <span style={{ color: "#64748b", fontWeight: 400 }}>({tapeSegs.length} segments)</span></div>
      <div style={{ maxHeight: 400, overflowY: "auto" }}>
        {tapeSegs.map((seg) => (
          <div key={seg.number} style={st.segItem}>
            <div style={st.segTime}>{seg.in_time} – {seg.out_time}</div>
            {seg.keywords.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                {seg.keywords.map((kw, i) => (
                  <span key={i} style={st.kwTag}>{kw}</span>
                ))}
              </div>
            )}
            {seg.people.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                {seg.people.map((p, i) => (
                  <span key={i} style={st.personTag}>{p}</span>
                ))}
              </div>
            )}
            {!seg.keywords.length && !seg.people.length && (
              <div style={{ color: "#475569", fontSize: 11 }}>no index</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── TranscriptPanel ───────────────────────────────────────────────────────────

function TranscriptPanel({
  utterances,
  segments,
  tapeNumber,
  filter,
}: {
  utterances: Utterance[];
  segments: VhaSegment[];
  tapeNumber: number;
  filter: "all" | "interviewer" | "interviewee";
}) {
  const tapeSegs = segments
    .filter(s => s.in_file === tapeNumber || s.out_file === tapeNumber)
    .sort((a, b) => a.in_time_sec - b.in_time_sec);

  const filtered = filter === "all" ? utterances : utterances.filter(u => u.speaker === filter);

  // Group utterances into VHA segments
  const getSegForMs = (ms: number): VhaSegment | null => {
    const sec = ms / 1000;
    for (const seg of tapeSegs) {
      if (sec >= seg.in_time_sec && sec < seg.out_time_sec) return seg;
    }
    return null;
  };

  // Build list of items: segment headers interspersed with utterances
  const items: (
    | { kind: "utterance"; u: Utterance; idx: number }
    | { kind: "segment"; seg: VhaSegment }
  )[] = [];

  let lastSegNum = -1;
  for (let i = 0; i < filtered.length; i++) {
    const u = filtered[i];
    const seg = getSegForMs(u.start_ms);
    if (seg && seg.number !== lastSegNum) {
      items.push({ kind: "segment", seg });
      lastSegNum = seg.number;
    }
    items.push({ kind: "utterance", u, idx: i });
  }

  return (
    <div style={st.transcriptPanel}>
      {items.map((item, i) => {
        if (item.kind === "segment") {
          const { seg } = item;
          const hasContent = seg.keywords.length > 0 || seg.people.length > 0;
          if (!hasContent) return null;
          return (
            <div key={`seg-${seg.number}-${i}`} style={st.segDivider}>
              <span style={st.segDividerLabel}>
                {seg.in_time} – {seg.out_time}
              </span>
              {seg.keywords.slice(0, 4).map((kw, j) => (
                <span key={j} style={st.kwTagInline}>{kw}</span>
              ))}
              {seg.people.slice(0, 3).map((p, j) => (
                <span key={j} style={st.personTagInline}>{p}</span>
              ))}
            </div>
          );
        }

        const { u } = item;
        const isIr = u.speaker === "interviewer";
        return (
          <div key={`utt-${item.idx}`} style={{ ...st.utterance, ...(isIr ? st.irUtterance : st.ieUtterance) }}>
            <div style={st.uttHeader}>
              <span style={{ ...st.speakerTag, color: isIr ? "#60a5fa" : "#34d399" }}>
                {isIr ? "Interviewer" : "Interviewee"}
                {u.tag && <span style={{ color: "#475569" }}> · {u.tag}</span>}
              </span>
              <span style={st.uttTime}>{fmtTime(u.start_ms)}</span>
              {u.type === "non_verbal" && <span style={st.nvTag}>non-verbal</span>}
            </div>
            <div style={st.uttText}>{u.text}</div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function TranscriptView() {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();

  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [speakerFilter, setSpeakerFilter] = useState<"all" | "interviewer" | "interviewee">("all");

  const intCode = videoId ? parseInt(videoId.split(".")[0], 10) : null;
  const tapeNumber = videoId ? parseInt(videoId.split(".")[1] ?? "1", 10) : 1;

  useEffect(() => {
    if (!videoId || !intCode) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [tRes, mRes] = await Promise.all([
          fetch(`${API}/videos/${videoId}/transcript`),
          fetch(`${API}/metadata/${intCode}`),
        ]);
        if (cancelled) return;
        if (!tRes.ok) throw new Error(`Transcript not found (${tRes.status})`);
        const [transcript, meta] = await Promise.all([tRes.json(), mRes.ok ? mRes.json() : null]);
        if (cancelled) return;
        setUtterances(transcript);
        setMetadata(meta);
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [videoId, intCode]);

  const durationMs = utterances.length ? utterances[utterances.length - 1].end_ms : 0;

  const irCount = utterances.filter(u => u.speaker === "interviewer").length;
  const ieCount = utterances.filter(u => u.speaker === "interviewee").length;
  const irMs = utterances.filter(u => u.speaker === "interviewer").reduce((s, u) => s + (u.end_ms - u.start_ms), 0);
  const ieMs = utterances.filter(u => u.speaker === "interviewee").reduce((s, u) => s + (u.end_ms - u.start_ms), 0);

  return (
    <div style={st.page}>
      {/* Header */}
      <div style={st.header}>
        <button style={st.backBtn} onClick={() => navigate(-1)}>← Back</button>
        <Link to="/transcripts" style={st.groupLink}>Group View</Link>
        <div style={{ flex: 1 }}>
          <h1 style={st.title}>
            Transcript — Interview {intCode}, Tape {tapeNumber}
            {metadata?.name && <span style={st.titleSub}> · {metadata.name}</span>}
          </h1>
        </div>
      </div>

      {loading && <div style={st.msg}>Loading transcript and metadata…</div>}
      {error && <div style={{ ...st.msg, color: "#f87171" }}>{error}</div>}

      {!loading && !error && (
        <>
          {/* Stats bar */}
          <div style={st.statsBar}>
            <StatChip label="Duration" value={fmtTime(durationMs)} />
            <StatChip label="Total utterances" value={String(utterances.length)} />
            <StatChip label="Interviewer" value={`${irCount} turns · ${fmtTime(irMs)}`} color="#60a5fa" />
            <StatChip label="Interviewee" value={`${ieCount} turns · ${fmtTime(ieMs)}`} color="#34d399" />
            {metadata && <StatChip label="VHA segments" value={String(metadata.segments.filter(s => s.in_file === tapeNumber).length)} color="#f59e0b" />}
          </div>

          {/* Timeline */}
          <div style={st.section}>
            <div style={st.sectionTitle}>Speech Timeline</div>
            <SpeechTimeline
              utterances={utterances}
              segments={metadata?.segments ?? []}
              tapeNumber={tapeNumber}
              durationMs={durationMs}
            />
          </div>

          {/* Main content: Bio + Segments left, Transcript right */}
          <div style={st.mainLayout}>
            {/* Left: Bio + Segment Keywords */}
            <div style={st.leftCol}>
              {metadata ? <BioPanel meta={metadata} /> : (
                <div style={st.bioPanel}>
                  <div style={{ color: "#64748b", fontSize: 13 }}>No VHA metadata for interview {intCode}.</div>
                </div>
              )}
              {metadata && (
                <SegmentPanel segments={metadata.segments} tapeNumber={tapeNumber} />
              )}
            </div>

            {/* Right: Transcript */}
            <div style={st.rightCol}>
              <div style={st.sectionTitle}>
                Transcript
                <span style={{ marginLeft: 16, fontSize: 12, fontWeight: 400 }}>
                  <FilterBtn label="All" active={speakerFilter === "all"} onClick={() => setSpeakerFilter("all")} />
                  <FilterBtn label="Interviewer" active={speakerFilter === "interviewer"} onClick={() => setSpeakerFilter("interviewer")} />
                  <FilterBtn label="Interviewee" active={speakerFilter === "interviewee"} onClick={() => setSpeakerFilter("interviewee")} />
                </span>
              </div>
              <TranscriptPanel
                utterances={utterances}
                segments={metadata?.segments ?? []}
                tapeNumber={tapeNumber}
                filter={speakerFilter}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatChip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={st.statChip}>
      <div style={{ ...st.statValue, color: color ?? "#f8fafc" }}>{value}</div>
      <div style={st.statLabel}>{label}</div>
    </div>
  );
}

function FilterBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        marginRight: 6,
        padding: "3px 10px",
        borderRadius: 4,
        border: "1px solid",
        borderColor: active ? "#3b82f6" : "#334155",
        backgroundColor: active ? "#3b82f622" : "transparent",
        color: active ? "#60a5fa" : "#94a3b8",
        cursor: "pointer",
        fontSize: 12,
      }}
    >
      {label}
    </button>
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
    gap: 12,
    marginBottom: 16,
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
  groupLink: {
    padding: "8px 14px",
    backgroundColor: "#1e293b",
    color: "#94a3b8",
    border: "1px solid #334155",
    borderRadius: 6,
    fontSize: "0.9rem",
    textDecoration: "none",
  },
  title: {
    fontSize: "1.3rem",
    fontWeight: 600,
    color: "#f8fafc",
    margin: 0,
  },
  titleSub: {
    color: "#94a3b8",
    fontWeight: 400,
  },
  statsBar: {
    display: "flex",
    gap: 12,
    marginBottom: 20,
    flexWrap: "wrap",
  },
  statChip: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "8px 14px",
    minWidth: 120,
  },
  statValue: {
    fontSize: "1rem",
    fontWeight: 600,
    color: "#f8fafc",
  },
  statLabel: {
    fontSize: "0.75rem",
    color: "#64748b",
    marginTop: 2,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: 10,
  },
  mainLayout: {
    display: "flex",
    gap: 20,
    alignItems: "flex-start",
  },
  leftCol: {
    flex: "0 0 300px",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  rightCol: {
    flex: 1,
    minWidth: 0,
  },
  bioPanel: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 10,
    padding: 16,
  },
  bioName: {
    fontSize: "1.1rem",
    fontWeight: 700,
    color: "#f8fafc",
    marginBottom: 2,
  },
  bioSubtitle: {
    fontSize: "0.8rem",
    color: "#64748b",
    marginBottom: 12,
  },
  bioGrid: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  bioRow: {
    display: "flex",
    gap: 8,
    fontSize: 12,
  },
  bioLabel: {
    color: "#64748b",
    minWidth: 90,
    flexShrink: 0,
  },
  bioValue: {
    color: "#cbd5e1",
  },
  bioSection: {
    marginTop: 12,
  },
  bioSectionTitle: {
    fontSize: "0.75rem",
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    marginBottom: 6,
  },
  bioTag: {
    fontSize: 11,
    color: "#94a3b8",
    backgroundColor: "#0f172a",
    border: "1px solid #334155",
    borderRadius: 4,
    padding: "2px 7px",
    marginBottom: 3,
  },
  segPanel: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 10,
    padding: 16,
  },
  segTitle: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#f8fafc",
    marginBottom: 10,
  },
  segItem: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottom: "1px solid #1e293b80",
  },
  segTime: {
    fontSize: 11,
    color: "#64748b",
    fontVariantNumeric: "tabular-nums",
  },
  kwTag: {
    fontSize: 10,
    color: "#fbbf24",
    backgroundColor: "#f59e0b1a",
    border: "1px solid #f59e0b33",
    borderRadius: 3,
    padding: "1px 6px",
  },
  personTag: {
    fontSize: 10,
    color: "#a78bfa",
    backgroundColor: "#8b5cf61a",
    border: "1px solid #8b5cf633",
    borderRadius: 3,
    padding: "1px 6px",
  },
  transcriptPanel: {
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 10,
    padding: 16,
    maxHeight: "70vh",
    overflowY: "auto",
  },
  utterance: {
    padding: "8px 12px",
    borderRadius: 6,
    marginBottom: 6,
  },
  irUtterance: {
    backgroundColor: "#3b82f608",
    borderLeft: "3px solid #3b82f640",
  },
  ieUtterance: {
    backgroundColor: "#10b98108",
    borderLeft: "3px solid #10b98140",
  },
  uttHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 4,
  },
  speakerTag: {
    fontSize: 11,
    fontWeight: 600,
  },
  uttTime: {
    fontSize: 11,
    color: "#475569",
    fontVariantNumeric: "tabular-nums",
    marginLeft: "auto",
  },
  nvTag: {
    fontSize: 10,
    color: "#64748b",
    backgroundColor: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 3,
    padding: "1px 5px",
  },
  uttText: {
    fontSize: "0.875rem",
    color: "#cbd5e1",
    lineHeight: 1.6,
  },
  segDivider: {
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 5,
    margin: "12px 0 8px",
    paddingLeft: 4,
    borderTop: "1px solid #f59e0b22",
    paddingTop: 8,
  },
  segDividerLabel: {
    fontSize: 10,
    color: "#f59e0b99",
    fontVariantNumeric: "tabular-nums",
    marginRight: 4,
  },
  kwTagInline: {
    fontSize: 10,
    color: "#fbbf24",
    backgroundColor: "#f59e0b1a",
    border: "1px solid #f59e0b33",
    borderRadius: 3,
    padding: "1px 5px",
  },
  personTagInline: {
    fontSize: 10,
    color: "#a78bfa",
    backgroundColor: "#8b5cf61a",
    border: "1px solid #8b5cf633",
    borderRadius: 3,
    padding: "1px 5px",
  },
  msg: {
    textAlign: "center",
    padding: "48px",
    color: "#64748b",
  },
};
