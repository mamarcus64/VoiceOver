import { useEffect, useRef } from "react";
import type { Utterance } from "../types";

const DARK_BG = "#1a1a2e";
const ROW_BG = "#16213e";
const ACTIVE_BG = "#0f3460";
const INTERVIEWER = "#3b82f6";
const INTERVIEWEE = "#22c55e";
const NON_VERBAL = "#94a3b8";

interface TranscriptTrackProps {
  utterances: Utterance[];
  currentTimeMs: number;
  onSeek: (ms: number) => void;
}

export default function TranscriptTrack({
  utterances,
  currentTimeMs,
  onSeek,
}: TranscriptTrackProps) {
  const activeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [currentTimeMs]);

  return (
    <div
      style={{
        maxHeight: "500px",
        overflowY: "auto",
        backgroundColor: DARK_BG,
        borderRadius: "8px",
        padding: "8px",
      }}
    >
      {utterances.map((u, i) => {
        const isActive =
          currentTimeMs >= u.start_ms && currentTimeMs < u.end_ms;
        const isNonVerbal = u.type === "non_verbal";

        return (
          <div
            key={`${u.start_ms}-${i}`}
            ref={isActive ? activeRef : null}
            onClick={() => onSeek(u.start_ms)}
            style={{
              display: "flex",
              gap: "12px",
              padding: "10px 12px",
              marginBottom: "4px",
              backgroundColor: isActive ? ACTIVE_BG : ROW_BG,
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <span
              style={{
                flexShrink: 0,
                width: "100px",
                color: isNonVerbal
                  ? NON_VERBAL
                  : u.speaker === "interviewer"
                    ? INTERVIEWER
                    : INTERVIEWEE,
                fontWeight: 600,
                fontSize: "13px",
              }}
            >
              {u.tag}
            </span>
            <span
              style={{
                flex: 1,
                color: isNonVerbal ? NON_VERBAL : "#e2e8f0",
                fontStyle: isNonVerbal ? "italic" : "normal",
                fontSize: "14px",
                lineHeight: 1.5,
              }}
            >
              {u.text}
            </span>
          </div>
        );
      })}
    </div>
  );
}
