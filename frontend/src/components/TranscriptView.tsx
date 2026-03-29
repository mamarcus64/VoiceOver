import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useNavigate, Link, useSearchParams } from "react-router-dom";
import type { Utterance } from "../types";

const API = "/api";

// ── Types ────────────────────────────────────────────────────────────────────

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

// ── Helpers ───────────────────────────────────────────────────────────────────

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
  return `${months[parseInt(m) - 1] ?? m} ${d}, ${y}`;
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
  const H = PAD_TOP + LANE_H + 6 + LANE_H + 10 + MARKER_ZONE_H + 18;
  const DRAW_W = W - LABEL_W;

  const dur = durationMs || 1;
  const toX = (ms: number) => LABEL_W + (ms / dur) * DRAW_W;

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
            <rect key={i} x={x} y={y + 2} width={w} height={LANE_H - 4}
              fill={`${color}${alpha}`} rx={1} style={{ cursor: "pointer" }}
              onMouseEnter={(e) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  const preview = u.text.length > 120 ? u.text.slice(0, 120) + "…" : u.text;
                  setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, text: `[${fmtTime(u.start_ms)}] ${preview}` });
                }
              }}
            />
          );
        })}

        {/* VHA segment markers */}
        {tapeSegments.map((seg) => {
          if (!seg.keywords.length && !seg.people.length) return null;
          const mx = toX(seg.in_time_sec * 1000);
          if (mx < LABEL_W || mx > W) return null;
          return (
            <g key={seg.number}>
              <line x1={mx} y1={interviewerY} x2={mx} y2={markerY + MARKER_ZONE_H}
                stroke="#f59e0b44" strokeWidth={1} strokeDasharray="3,2" />
              <rect x={mx - 10} y={markerY + 2} width={20} height={MARKER_ZONE_H - 4}
                fill="#f59e0b22" rx={3} stroke="#f59e0b66" strokeWidth={1}
                style={{ cursor: "pointer" }}
                onMouseEnter={(e) => {
                  const rect = svgRef.current?.getBoundingClientRect();
                  if (rect) {
                    const kws = seg.keywords.slice(0, 5).join(", ");
                    const ppl = seg.people.slice(0, 3).join(", ");
                    const txt = [
                      `Min ${seg.number} (${seg.in_time} → ${seg.out_time})`,
                      kws && `Keywords: ${kws}`,
                      ppl && `People: ${ppl}`,
                    ].filter(Boolean).join("\n");
                    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, text: txt });
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

        {/* Minute tick marks */}
        {Array.from({ length: Math.ceil(dur / 60000) + 1 }, (_, i) => {
          const x = toX(i * 60000);
          if (x < LABEL_W || x > W) return null;
          return (
            <g key={i}>
              <line x1={x} y1={markerY + MARKER_ZONE_H} x2={x} y2={markerY + MARKER_ZONE_H + 4}
                stroke="#475569" strokeWidth={1} />
              {i % 5 === 0 && (
                <text x={x} y={markerY + MARKER_ZONE_H + 14}
                  textAnchor="middle" fill="#64748b" fontSize={9} fontFamily="system-ui">{i}m</text>
              )}
            </g>
          );
        })}

        <text x={LABEL_W - 6} y={markerY + MARKER_ZONE_H / 2 + 4} textAnchor="end"
          fill="#64748b" fontSize={10} fontFamily="system-ui">Segments</text>
      </svg>

      {tooltip && (
        <div style={{
          position: "absolute", left: tooltip.x + 12, top: tooltip.y - 8,
          backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 6,
          padding: "6px 10px", fontSize: 12, color: "#e2e8f0", maxWidth: 320,
          pointerEvents: "none", zIndex: 10, whiteSpace: "pre-line", lineHeight: 1.5,
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
        }}>
          {tooltip.text}
        </div>
      )}

      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: "#64748b" }}>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#3b82f6cc", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />Interviewer</span>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#10b981cc", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />Interviewee</span>
        <span><span style={{ display: "inline-block", width: 12, height: 8, backgroundColor: "#f59e0b22", border: "1px solid #f59e0b66", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />VHA 1-min segment (hover)</span>
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
        <BioRow label="Total Length" value={refs["Length of Interview"]} />
        <BioRow label="Language" value={refs["Language(s) of Interview"]} />
        <BioRow label="Location" value={
          refs["State of Interview"]
            ? `${refs["State of Interview"]}, ${refs["Country of Interview"] ?? ""}`
            : refs["Country of Interview"]
        } />
        <BioRow label="Event" value={refs["Historic Event"]} />
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
  const tapeSegs = segments
    .filter(s => s.in_file === tapeNumber || s.out_file === tapeNumber)
    .sort((a, b) => a.in_time_sec - b.in_time_sec);

  return (
    <div style={st.segPanel}>
      <div style={st.segTitle}>
        Segment Keywords
        <span style={{ color: "#64748b", fontWeight: 400 }}> ({tapeSegs.length})</span>
      </div>
      {tapeSegs.length === 0 ? (
        <div style={{ color: "#64748b", fontSize: 12 }}>No VHA segments for this tape.</div>
      ) : (
        <div style={{ maxHeight: 380, overflowY: "auto" }}>
          {tapeSegs.map((seg) => (
            <div key={seg.number} style={st.segItem}>
              <div style={st.segTime}>{seg.in_time} – {seg.out_time}</div>
              {seg.keywords.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {seg.keywords.map((kw, i) => <span key={i} style={st.kwTag}>{kw}</span>)}
                </div>
              )}
              {seg.people.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {seg.people.map((p, i) => <span key={i} style={st.personTag}>{p}</span>)}
                </div>
              )}
              {!seg.keywords.length && !seg.people.length && (
                <div style={{ color: "#475569", fontSize: 11 }}>no index</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── TranscriptPanel ───────────────────────────────────────────────────────────

function TranscriptPanel({
  utterances, segments, tapeNumber, filter,
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

  const getSegForMs = (ms: number): VhaSegment | null => {
    const sec = ms / 1000;
    for (const seg of tapeSegs) {
      if (sec >= seg.in_time_sec && sec < seg.out_time_sec) return seg;
    }
    return null;
  };

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
          if (!seg.keywords.length && !seg.people.length) return null;
          return (
            <div key={`seg-${seg.number}-${i}`} style={st.segDivider}>
              <span style={st.segDividerLabel}>{seg.in_time} – {seg.out_time}</span>
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
  const { intCode: intCodeStr } = useParams<{ intCode: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const intCode = intCodeStr ? parseInt(intCodeStr, 10) : null;
  const currentTape = parseInt(searchParams.get("tape") ?? "1", 10);

  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [availableTapes, setAvailableTapes] = useState<number[]>([]);
  const [allSubjects, setAllSubjects] = useState<number[]>([]);
  const [loadingTranscript, setLoadingTranscript] = useState(true);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [speakerFilter, setSpeakerFilter] = useState<"all" | "interviewer" | "interviewee">("all");

  // Load metadata + tapes + subjects list once per subject
  useEffect(() => {
    if (!intCode) return;
    let cancelled = false;
    setLoadingMeta(true);
    setMetadata(null);
    setAvailableTapes([]);

    Promise.all([
      fetch(`${API}/metadata/${intCode}`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/metadata/${intCode}/tapes`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/metadata/subjects`).then(r => r.ok ? r.json() : []),
    ]).then(([meta, tapesData, subjects]) => {
      if (cancelled) return;
      setMetadata(meta);
      setAvailableTapes(tapesData?.tapes ?? []);
      setAllSubjects(subjects ?? []);
      setLoadingMeta(false);
    }).catch(() => {
      if (!cancelled) setLoadingMeta(false);
    });

    return () => { cancelled = true; };
  }, [intCode]);

  // Load transcript whenever subject or tape changes
  useEffect(() => {
    if (!intCode) return;
    let cancelled = false;
    setLoadingTranscript(true);
    setTranscriptError(null);
    setUtterances([]);

    fetch(`${API}/videos/${intCode}.${currentTape}/transcript`)
      .then(r => {
        if (!r.ok) throw new Error(`No transcript for ${intCode}.${currentTape} (${r.status})`);
        return r.json();
      })
      .then(data => { if (!cancelled) { setUtterances(data); setLoadingTranscript(false); } })
      .catch(e => { if (!cancelled) { setTranscriptError(e.message); setLoadingTranscript(false); } });

    return () => { cancelled = true; };
  }, [intCode, currentTape]);

  const switchTape = useCallback((tape: number) => {
    setSearchParams({ tape: String(tape) }, { replace: true });
    setSpeakerFilter("all");
  }, [setSearchParams]);

  const subjectIdx = allSubjects.indexOf(intCode!);
  const prevSubject = subjectIdx > 0 ? allSubjects[subjectIdx - 1] : null;
  const nextSubject = subjectIdx >= 0 && subjectIdx < allSubjects.length - 1
    ? allSubjects[subjectIdx + 1] : null;

  const goToSubject = (code: number) => {
    navigate(`/transcript/${code}`);
    setSpeakerFilter("all");
  };

  const durationMs = utterances.length ? utterances[utterances.length - 1].end_ms : 0;
  const irMs = utterances.filter(u => u.speaker === "interviewer").reduce((s, u) => s + (u.end_ms - u.start_ms), 0);
  const ieMs = utterances.filter(u => u.speaker === "interviewee").reduce((s, u) => s + (u.end_ms - u.start_ms), 0);
  const irCount = utterances.filter(u => u.speaker === "interviewer").length;
  const ieCount = utterances.filter(u => u.speaker === "interviewee").length;
  const totalLength = metadata?.interview_refs?.["Length of Interview"];

  return (
    <div style={st.page}>
      {/* Header */}
      <div style={st.header}>
        <button style={st.backBtn} onClick={() => navigate("/")}>← Home</button>
        <Link to="/transcripts" style={st.navLink}>Testimony Archive</Link>

        {/* Prev / Next subject */}
        <button
          style={{ ...st.navBtn, opacity: prevSubject ? 1 : 0.35 }}
          disabled={!prevSubject}
          onClick={() => prevSubject && goToSubject(prevSubject)}
          title={prevSubject ? `Subject ${prevSubject}` : "No previous subject"}
        >
          ← Prev
        </button>
        <div style={st.subjectPill}>
          Subject {intCode}
          {allSubjects.length > 0 && subjectIdx >= 0 && (
            <span style={{ color: "#475569", marginLeft: 6 }}>
              {subjectIdx + 1} / {allSubjects.length}
            </span>
          )}
        </div>
        <button
          style={{ ...st.navBtn, opacity: nextSubject ? 1 : 0.35 }}
          disabled={!nextSubject}
          onClick={() => nextSubject && goToSubject(nextSubject)}
          title={nextSubject ? `Subject ${nextSubject}` : "No next subject"}
        >
          Next →
        </button>

        <div style={{ flex: 1, minWidth: 0 }}>
          <h1 style={st.title}>
            {loadingMeta ? `Interview ${intCode}` : (metadata?.name ?? `Interview ${intCode}`)}
            {metadata?.label && <span style={st.titleSub}> · {metadata.label}</span>}
          </h1>
        </div>
      </div>

      {/* Tape tabs */}
      {availableTapes.length > 0 && (
        <div style={st.tapeTabs}>
          <span style={st.tapeLabel}>Tapes:</span>
          {availableTapes.map(tape => (
            <button
              key={tape}
              style={{
                ...st.tapeTab,
                ...(tape === currentTape ? st.tapeTabActive : {}),
              }}
              onClick={() => switchTape(tape)}
            >
              Tape {tape}
            </button>
          ))}
          {availableTapes.length === 0 && (
            <span style={{ color: "#475569", fontSize: 12 }}>No transcript files found for this subject.</span>
          )}
        </div>
      )}

      {/* Stats bar */}
      {!loadingTranscript && !transcriptError && utterances.length > 0 && (
        <div style={st.statsBar}>
          <StatChip label="Tape Duration" value={fmtTime(durationMs)} />
          {totalLength && <StatChip label="Total Interview" value={totalLength} color="#94a3b8" />}
          <StatChip label="Utterances" value={String(utterances.length)} />
          <StatChip label="Interviewer" value={`${irCount} turns · ${fmtTime(irMs)}`} color="#60a5fa" />
          <StatChip label="Interviewee" value={`${ieCount} turns · ${fmtTime(ieMs)}`} color="#34d399" />
          {metadata && (
            <StatChip
              label="VHA Segments (tape)"
              value={String(metadata.segments.filter(s => s.in_file === currentTape).length)}
              color="#f59e0b"
            />
          )}
        </div>
      )}

      {/* Transcript loading/error */}
      {loadingTranscript && (
        <div style={st.msg}>Loading transcript for {intCode}.{currentTape}…</div>
      )}
      {transcriptError && (
        <div style={{ ...st.msg, color: "#f87171" }}>{transcriptError}</div>
      )}

      {/* Timeline */}
      {!loadingTranscript && !transcriptError && utterances.length > 0 && (
        <div style={st.section}>
          <div style={st.sectionTitle}>Speech Timeline — Tape {currentTape}</div>
          <SpeechTimeline
            utterances={utterances}
            segments={metadata?.segments ?? []}
            tapeNumber={currentTape}
            durationMs={durationMs}
          />
        </div>
      )}

      {/* Main content */}
      {!loadingTranscript && (
        <div style={st.mainLayout}>
          {/* Left: Bio + Segments */}
          <div style={st.leftCol}>
            {loadingMeta ? (
              <div style={st.bioPanel}><div style={{ color: "#64748b", fontSize: 13 }}>Loading metadata…</div></div>
            ) : metadata ? (
              <BioPanel meta={metadata} />
            ) : (
              <div style={st.bioPanel}>
                <div style={{ color: "#64748b", fontSize: 13 }}>No VHA metadata for subject {intCode}.</div>
              </div>
            )}
            {metadata && <SegmentPanel segments={metadata.segments} tapeNumber={currentTape} />}
          </div>

          {/* Right: Transcript */}
          <div style={st.rightCol}>
            <div style={st.sectionTitle}>
              Transcript — Tape {currentTape}
              <span style={{ marginLeft: 16, fontSize: 12, fontWeight: 400 }}>
                <FilterBtn label="All" active={speakerFilter === "all"} onClick={() => setSpeakerFilter("all")} />
                <FilterBtn label="Interviewer" active={speakerFilter === "interviewer"} onClick={() => setSpeakerFilter("interviewer")} />
                <FilterBtn label="Interviewee" active={speakerFilter === "interviewee"} onClick={() => setSpeakerFilter("interviewee")} />
              </span>
            </div>
            {utterances.length > 0 ? (
              <TranscriptPanel
                utterances={utterances}
                segments={metadata?.segments ?? []}
                tapeNumber={currentTape}
                filter={speakerFilter}
              />
            ) : (
              !transcriptError && <div style={{ color: "#64748b", fontSize: 13, padding: 16 }}>No transcript available for this tape.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

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
    <button onClick={onClick} style={{
      marginRight: 6, padding: "3px 10px", borderRadius: 4, border: "1px solid",
      borderColor: active ? "#3b82f6" : "#334155",
      backgroundColor: active ? "#3b82f622" : "transparent",
      color: active ? "#60a5fa" : "#94a3b8",
      cursor: "pointer", fontSize: 12,
    }}>
      {label}
    </button>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const st: Record<string, React.CSSProperties> = {
  page: {
    padding: "20px 24px", maxWidth: 1600, margin: "0 auto",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: "#e2e8f0", backgroundColor: "#0f172a", minHeight: "100vh",
  },
  header: {
    display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap",
  },
  backBtn: {
    padding: "7px 12px", backgroundColor: "#1e293b", color: "#e2e8f0",
    border: "1px solid #334155", borderRadius: 6, cursor: "pointer", fontSize: "0.85rem",
  },
  navLink: {
    padding: "7px 12px", backgroundColor: "#1e293b", color: "#94a3b8",
    border: "1px solid #334155", borderRadius: 6, fontSize: "0.85rem", textDecoration: "none",
  },
  navBtn: {
    padding: "7px 14px", backgroundColor: "#1e293b", color: "#e2e8f0",
    border: "1px solid #334155", borderRadius: 6, cursor: "pointer", fontSize: "0.85rem",
    transition: "all 0.15s",
  },
  subjectPill: {
    padding: "6px 12px", backgroundColor: "#0f172a", color: "#94a3b8",
    border: "1px solid #475569", borderRadius: 20, fontSize: "0.8rem",
    fontVariantNumeric: "tabular-nums",
  },
  title: {
    fontSize: "1.2rem", fontWeight: 600, color: "#f8fafc", margin: 0,
  },
  titleSub: { color: "#94a3b8", fontWeight: 400 },
  tapeTabs: {
    display: "flex", alignItems: "center", gap: 6, marginBottom: 16,
    padding: "10px 14px", backgroundColor: "#1e293b", borderRadius: 8,
    border: "1px solid #334155", flexWrap: "wrap",
  },
  tapeLabel: {
    fontSize: 12, color: "#64748b", marginRight: 4, fontWeight: 500,
  },
  tapeTab: {
    padding: "5px 14px", borderRadius: 6, border: "1px solid #334155",
    backgroundColor: "transparent", color: "#94a3b8", cursor: "pointer", fontSize: 13,
    transition: "all 0.12s",
  },
  tapeTabActive: {
    backgroundColor: "#3b82f622", borderColor: "#3b82f6", color: "#60a5fa", fontWeight: 600,
  },
  statsBar: {
    display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap",
  },
  statChip: {
    backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 8,
    padding: "8px 14px", minWidth: 110,
  },
  statValue: {
    fontSize: "0.95rem", fontWeight: 600, color: "#f8fafc",
  },
  statLabel: {
    fontSize: "0.72rem", color: "#64748b", marginTop: 2,
  },
  section: { marginBottom: 20 },
  sectionTitle: {
    fontSize: "0.85rem", fontWeight: 600, color: "#94a3b8",
    textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10,
  },
  mainLayout: { display: "flex", gap: 20, alignItems: "flex-start" },
  leftCol: { flex: "0 0 290px", display: "flex", flexDirection: "column", gap: 14 },
  rightCol: { flex: 1, minWidth: 0 },
  bioPanel: {
    backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 10, padding: 16,
  },
  bioName: { fontSize: "1.05rem", fontWeight: 700, color: "#f8fafc", marginBottom: 2 },
  bioSubtitle: { fontSize: "0.78rem", color: "#64748b", marginBottom: 10 },
  bioGrid: { display: "flex", flexDirection: "column", gap: 4 },
  bioRow: { display: "flex", gap: 8, fontSize: 12 },
  bioLabel: { color: "#64748b", minWidth: 88, flexShrink: 0 },
  bioValue: { color: "#cbd5e1" },
  bioSection: { marginTop: 12 },
  bioSectionTitle: {
    fontSize: "0.72rem", fontWeight: 600, color: "#94a3b8",
    textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 6,
  },
  bioTag: {
    fontSize: 11, color: "#94a3b8", backgroundColor: "#0f172a",
    border: "1px solid #334155", borderRadius: 4, padding: "2px 7px", marginBottom: 3,
  },
  segPanel: {
    backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 10, padding: 16,
  },
  segTitle: { fontSize: "0.85rem", fontWeight: 600, color: "#f8fafc", marginBottom: 10 },
  segItem: { marginBottom: 10, paddingBottom: 10, borderBottom: "1px solid #1e293b80" },
  segTime: { fontSize: 11, color: "#64748b", fontVariantNumeric: "tabular-nums" },
  kwTag: {
    fontSize: 10, color: "#fbbf24", backgroundColor: "#f59e0b1a",
    border: "1px solid #f59e0b33", borderRadius: 3, padding: "1px 6px",
  },
  personTag: {
    fontSize: 10, color: "#a78bfa", backgroundColor: "#8b5cf61a",
    border: "1px solid #8b5cf633", borderRadius: 3, padding: "1px 6px",
  },
  transcriptPanel: {
    backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 10, padding: 16,
    maxHeight: "70vh", overflowY: "auto",
  },
  utterance: { padding: "8px 12px", borderRadius: 6, marginBottom: 6 },
  irUtterance: { backgroundColor: "#3b82f608", borderLeft: "3px solid #3b82f640" },
  ieUtterance: { backgroundColor: "#10b98108", borderLeft: "3px solid #10b98140" },
  uttHeader: { display: "flex", alignItems: "center", gap: 8, marginBottom: 4 },
  speakerTag: { fontSize: 11, fontWeight: 600 },
  uttTime: { fontSize: 11, color: "#475569", fontVariantNumeric: "tabular-nums", marginLeft: "auto" },
  nvTag: {
    fontSize: 10, color: "#64748b", backgroundColor: "#1e293b",
    border: "1px solid #334155", borderRadius: 3, padding: "1px 5px",
  },
  uttText: { fontSize: "0.875rem", color: "#cbd5e1", lineHeight: 1.6 },
  segDivider: {
    display: "flex", flexWrap: "wrap", alignItems: "center", gap: 5,
    margin: "12px 0 8px", paddingLeft: 4, borderTop: "1px solid #f59e0b22", paddingTop: 8,
  },
  segDividerLabel: { fontSize: 10, color: "#f59e0b99", fontVariantNumeric: "tabular-nums", marginRight: 4 },
  kwTagInline: {
    fontSize: 10, color: "#fbbf24", backgroundColor: "#f59e0b1a",
    border: "1px solid #f59e0b33", borderRadius: 3, padding: "1px 5px",
  },
  personTagInline: {
    fontSize: 10, color: "#a78bfa", backgroundColor: "#8b5cf61a",
    border: "1px solid #8b5cf633", borderRadius: 3, padding: "1px 5px",
  },
  msg: { textAlign: "center", padding: "48px", color: "#64748b" },
};
