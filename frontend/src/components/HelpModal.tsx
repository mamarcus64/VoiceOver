import { useEffect } from "react";
import { SMILE_LABELS } from "../types";

interface Props {
  onClose: () => void;
}

export default function HelpModal({ onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        backgroundColor: "rgba(0,0,0,0.65)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "12px",
          padding: "32px 36px",
          maxWidth: "640px",
          width: "90vw",
          maxHeight: "88vh",
          overflowY: "auto",
          boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
          color: "#e2e8f0",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
          <div>
            <div style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f8fafc" }}>How This Works</div>
            <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginTop: "4px" }}>Smile Annotation Guide</div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "#64748b", fontSize: "1.4rem", lineHeight: 1, padding: "0 4px",
            }}
          >
            ✕
          </button>
        </div>

        {/* Section: Your task */}
        <Section title="Your task">
          <p>
            You will be shown short clips from interview videos. In each clip, a smile has been automatically
            detected and is <Hl>highlighted in amber</Hl> — on the border around the video, the seek bar, and
            in the transcript. Your job is to classify that one smile.
          </p>
          <p style={{ marginTop: "8px" }}>
            There may be other smiles in the clip. <strong>Ignore them</strong> — only label the highlighted one.
          </p>
        </Section>

        {/* Section: Labels */}
        <Section title="Label definitions">
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "4px" }}>
            {SMILE_LABELS.map((l) => (
              <div key={l.key} style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
                <span style={{
                  flexShrink: 0, marginTop: "2px",
                  width: "12px", height: "12px", borderRadius: "50%",
                  backgroundColor: l.color, display: "inline-block",
                }} />
                <div>
                  <span style={{ fontWeight: 700, color: "#f8fafc" }}>{l.display}: </span>
                  <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{l.desc}</span>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Section: Not a Smile toggle */}
        <Section title="Not a Smile">
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", lineHeight: 1.7 }}>
            If the detector flagged something that is <strong style={{ color: "#e2e8f0" }}>not actually a smile</strong>,
            click the{" "}
            <span style={{
              backgroundColor: "#475569", color: "#f8fafc",
              padding: "1px 7px", borderRadius: "4px", fontSize: "0.82rem", fontWeight: 600,
            }}>
              Not a Smile
            </span>{" "}
            toggle. This reveals two options:
          </p>
          <ul style={{ margin: "8px 0 0", paddingLeft: "18px", color: "#94a3b8", lineHeight: 1.8, fontSize: "0.9rem" }}>
            <li>
              <strong style={{ color: "#e2e8f0" }}>Assign an emotion label anyway</strong> — pick Genuine, Polite, or Masking
              so we preserve what the expression <em>looks like</em> even when it isn't a true smile.
            </li>
            <li>
              <strong style={{ color: "#e2e8f0" }}>Definitely Not a Smile</strong> — use this only when you are 100% certain
              there is no smile at all. If there is any ambiguity, prefer assigning an emotion label.
            </li>
          </ul>
        </Section>

        {/* Section: Workflow */}
        <Section title="Workflow">
          <ul style={{ margin: 0, paddingLeft: "18px", color: "#94a3b8", lineHeight: 1.8, fontSize: "0.9rem" }}>
            <li>The video plays automatically. Use the speed buttons to slow it down if needed.</li>
            <li>Click a label button to record your answer and advance to the next task.</li>
            <li>Use <strong style={{ color: "#e2e8f0" }}>Prev / Next</strong> or the jump box to revisit tasks — your saved label is shown in the top bar.</li>
            <li>The <strong style={{ color: "#e2e8f0" }}>Before / After</strong> buttons let you adjust how much context is shown around the smile.</li>
          </ul>
        </Section>

        {/* Section: Two-label mode */}
        <Section title="When a smile could be two things">
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", lineHeight: 1.7 }}>
            If a smile feels genuinely ambiguous between two categories, click{" "}
            <span style={{
              backgroundColor: "#664113", color: "#fef3c7",
              padding: "1px 7px", borderRadius: "4px", fontSize: "0.82rem", fontWeight: 600,
            }}>
              This smile could be two labels
            </span>{" "}
            before labelling. Your first click sets the <strong style={{ color: "#e2e8f0" }}>primary</strong> label,
            and your second click sets the <strong style={{ color: "#e2e8f0" }}>runner-up</strong>. Both are recorded.
            Press <strong style={{ color: "#e2e8f0" }}>Escape</strong> or <strong style={{ color: "#e2e8f0" }}>Cancel</strong> to undo the first pick.
          </p>
        </Section>

        {/* Section: Tips */}
        <Section title="Tips">
          <ul style={{ margin: 0, paddingLeft: "18px", color: "#94a3b8", lineHeight: 1.8, fontSize: "0.9rem" }}>
            <li>Read the transcript — the <em>words</em> often clarify the emotional context.</li>
            <li>When unsure, go with your first instinct. Speed matters more than perfection.</li>
            <li>Use <strong style={{ color: "#e2e8f0" }}>Notes</strong> for anything worth flagging to the research team.</li>
            <li>If a smile is simply mislabelled by the detector, toggle <strong style={{ color: "#e2e8f0" }}>Not a Smile</strong> on, then still pick an emotion label.</li>
          </ul>
        </Section>

        <button
          onClick={onClose}
          style={{
            marginTop: "24px", width: "100%",
            padding: "11px", fontSize: "0.95rem", fontWeight: 600,
            border: "none", borderRadius: "8px",
            backgroundColor: "#3b82f6", color: "#fff", cursor: "pointer",
          }}
        >
          Got it
        </button>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: "20px" }}>
      <div style={{
        fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.08em",
        textTransform: "uppercase", color: "#64748b", marginBottom: "8px",
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Hl({ children }: { children: React.ReactNode }) {
  return <strong style={{ color: "#fbbf24" }}>{children}</strong>;
}
