import { useEffect } from "react";

interface Props {
  onClose: () => void;
}

const EKMAN_URL =
  "https://www.paulekman.com/wp-content/uploads/2013/07/Felt-False-And-Miserable-Smiles.pdf";

export default function HelpModal({ onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function handleClose() {
    onClose();
  }

  return (
    <div
      onClick={handleClose}
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        backgroundColor: "rgba(0,0,0,0.65)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <style>{`
        details[open] .label-card-chevron { display: none !important; }
        details summary::-webkit-details-marker { display: none; }
      `}</style>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "12px",
          padding: "32px 36px",
          maxWidth: "660px",
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
            <div style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f8fafc" }}>Annotation Instructions</div>
            <div style={{ fontSize: "0.85rem", color: "#94a3b8", marginTop: "4px" }}>
              Ekman smile taxonomy —{" "}
              <a href={EKMAN_URL} target="_blank" rel="noopener noreferrer"
                style={{ color: "#60a5fa", textDecoration: "underline" }}>
                Felt, False, and Miserable Smiles (Ekman &amp; Friesen, 1982)
              </a>
            </div>
          </div>
          <button
            onClick={handleClose}
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
            in the transcript. Your job is to classify that one smile. If a smile occurs near it but not in the amber, then still label it.
          </p>
        </Section>

        {/* Section: Label definitions */}
        <Section title="Label definitions">
          <LabelCard
            color="#22c55e"
            title="Felt Smile"
            short="A genuine smile of positive emotion, also known as the Duchenne smile. Physical indicators are the lip corners pulled up and the muscles around the eyes tightened, producing raised cheeks and wrinkles."
          >
            The felt smile, also known as the Duchenne smile, is a spontaneous expression of genuine positive
            emotion such as enjoyment, amusement, relief, contentment, or pleasure. It is produced by the
            zygomatic major muscle pulling the lip corners upward, and critically, it is accompanied by
            tightening of the orbicularis oculi muscle around the eyes. This eye involvement produces raised
            cheeks, bagged skin below the eyes, crow's feet wrinkles, and a slight lowering of the eyebrow. No
            other muscles in the lower face are active. The felt smile lasts longer and is more intense when
            the positive feeling is stronger. The key diagnostic feature is the eye muscle activation (the
            "Duchenne marker").
          </LabelCard>

          <LabelCard
            color="#3b82f6"
            title="False Smile"
            short="A deliberate smile intended to convince others that positive emotion is felt when it isn't. It typically lacks eye involvement and may appear asymmetrical, have abrupt timing, or show traces of a negative emotion leaking through."
          >
            The false smile is a deliberate attempt to convince another person that positive emotion is felt
            when it isn't. The person may be feeling nothing in particular, or may be using the smile as a
            mask to conceal a negative emotion. Unlike the miserable smile, the false smile is deceptive.
            Distinguishing features: (1) typically more asymmetrical than a felt smile; (2) lacks orbicularis
            oculi involvement — a slight-to-moderate false smile will not show raised cheeks, bagged skin, or
            crow's feet; (3) timing may be inappropriate — it may drop off too abruptly or show a "stepped"
            offset; (4) when used as a mask, traces of the concealed emotion may leak through, creating the
            appearance of an emotion blend. In this dataset, false smiles are likely to appear when a speaker
            is managing the interviewer's comfort or projecting composure they do not feel.
          </LabelCard>

          <LabelCard
            color="#f59e0b"
            title="Miserable Smile"
            short="A smile that acknowledges negative emotion without trying to hide it. The person is not pretending to be happy. The subject is nonverbally commenting on their own unhappiness, embracing the negative emotion."
          >
            The miserable smile acknowledges the experience of negative emotion. It is not an attempt to
            conceal unhappiness but rather a facial comment on being miserable. This smile is often
            asymmetrical and is frequently superimposed on a clear negative expression (such as sadness, fear,
            or distress) on the rest of the face or before the smile. It may also appear immediately after a
            negative expression. A critical distinction from the false smile is that the miserable smile is
            not deceptive. The person is not trying to convince anyone they are happy. They may also show felt
            negative emotion in the eyebrows and forehead simultaneously. In this dataset, miserable smiles
            are likely to appear when speakers acknowledge the severity of a traumatic experience.
          </LabelCard>
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
              <strong style={{ color: "#e2e8f0" }}>Assign an emotion label anyway</strong> — pick Felt, False, or Miserable
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
          onClick={handleClose}
          style={{
            marginTop: "24px", width: "100%",
            padding: "11px", fontSize: "0.95rem", fontWeight: 600,
            border: "none", borderRadius: "8px",
            backgroundColor: "#3b82f6", color: "#fff", cursor: "pointer",
          }}
        >
          Got it — start annotating
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

function LabelCard({
  color,
  title,
  short,
  children,
}: {
  color: string;
  title: string;
  short: string;
  children: React.ReactNode;
}) {
  return (
    <details
      style={{
        marginBottom: "10px",
        border: "1px solid #334155",
        borderRadius: "8px",
        overflow: "hidden",
      }}
    >
      <summary
        style={{
          cursor: "pointer",
          padding: "10px 14px",
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          backgroundColor: "#0f172a",
          listStyle: "none",
          userSelect: "none",
        }}
      >
        <span
          style={{
            flexShrink: 0,
            marginTop: "4px",
            width: "12px",
            height: "12px",
            borderRadius: "50%",
            backgroundColor: color,
            display: "inline-block",
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{ fontWeight: 700, color: "#f8fafc" }}>{title}: </span>
          <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{short}</span>
        </div>
        <span
          className="label-card-chevron"
          style={{
            flexShrink: 0,
            marginTop: "2px",
            fontSize: "0.7rem",
            color: "#475569",
            display: "flex",
            alignItems: "center",
            gap: "3px",
            whiteSpace: "nowrap",
          }}
        >
          <span style={{ fontSize: "0.65rem", letterSpacing: "0.03em" }}>full definition</span>
          <span>▾</span>
        </span>
      </summary>
      <div
        style={{
          padding: "12px 14px 14px 36px",
          color: "#94a3b8",
          fontSize: "0.875rem",
          lineHeight: 1.75,
          borderTop: "1px solid #1e293b",
        }}
      >
        {children}
      </div>
    </details>
  );
}
