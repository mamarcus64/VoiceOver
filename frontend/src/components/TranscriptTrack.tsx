import { useEffect, useRef } from "react";
import type { Utterance, Word } from "../types";

const DARK_BG = "#1a1a2e";
const ROW_BG = "#16213e";
const ACTIVE_BG = "#0f3460";
const INTERVIEWER = "#3b82f6";
const INTERVIEWEE = "#22c55e";
const NON_VERBAL = "#94a3b8";
const CURRENT_WORD_BG = "#f59e0b";
const CURRENT_WORD_TEXT = "#0f172a";
const FUTURE_WORD = "#64748b";

interface TranscriptTrackProps {
  utterances: Utterance[];
  currentTimeMs: number;
  onSeek: (ms: number) => void;
}

function getCurrentWordIndex(words: Word[], currentTimeMs: number): number {
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    const nextWord = words[i + 1];
    const isCurrent =
      currentTimeMs >= word.ms &&
      (nextWord === undefined || currentTimeMs < nextWord.ms);
    if (isCurrent) return i;
  }
  return -1;
}

function getDisplayTag(u: Utterance): string {
  if (u.tag != null && u.tag !== "") {
    return u.tag;
  }
  return u.speaker === "interviewer" ? "Interviewer" : "Interviewee";
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
        const hasWords = u.words && u.words.length > 0;
        const currentWordIndex = hasWords
          ? getCurrentWordIndex(u.words!, currentTimeMs)
          : -1;
        const baseTextColor = isNonVerbal ? NON_VERBAL : "#e2e8f0";

        const renderTextContent = () => {
          if (isActive && hasWords) {
            return u.words!.map((word, wi) => {
              const isCurrent = wi === currentWordIndex;
              const isFuture = wi > currentWordIndex;

              let color = baseTextColor;
              let backgroundColor = "transparent";
              let padding = "0";
              let borderRadius = "0";

              if (isCurrent) {
                color = CURRENT_WORD_TEXT;
                backgroundColor = CURRENT_WORD_BG;
                padding = "2px 4px";
                borderRadius = "4px";
              } else if (isFuture) {
                color = FUTURE_WORD;
              }

              return (
                <span
                  key={`${word.ms}-${wi}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(word.ms);
                  }}
                  style={{
                    color,
                    backgroundColor,
                    padding,
                    borderRadius,
                    fontStyle: isNonVerbal ? "italic" : "normal",
                    cursor: "pointer",
                  }}
                >
                  {word.text}
                  {wi < u.words!.length - 1 ? " " : ""}
                </span>
              );
            });
          }

          return (
            <span
              style={{
                fontStyle: isNonVerbal ? "italic" : "normal",
              }}
            >
              {u.text}
            </span>
          );
        };

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
                width: "50px",
                color: isNonVerbal
                  ? NON_VERBAL
                  : u.speaker === "interviewer"
                    ? INTERVIEWER
                    : INTERVIEWEE,
                fontWeight: 600,
                fontSize: "13px",
              }}
            >
              {getDisplayTag(u)}
            </span>
            <span
              style={{
                flex: 1,
                color: baseTextColor,
                fontSize: "14px",
                lineHeight: 1.5,
              }}
            >
              {renderTextContent()}
            </span>
          </div>
        );
      })}
    </div>
  );
}
