import { useEffect, useRef, useState, useMemo } from "react";
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

type FontSize = "small" | "medium" | "large";
const FONT_SIZES: Record<FontSize, { text: number; tag: number; lineHeight: number }> = {
  small:  { text: 13, tag: 11, lineHeight: 1.5 },
  medium: { text: 15, tag: 13, lineHeight: 1.6 },
  large:  { text: 18, tag: 15, lineHeight: 1.7 },
};

const SENTENCES_PER_PARAGRAPH = 4;

interface VisualBlock {
  utteranceIndex: number;
  utterance: Utterance;
  words: Word[];
  startMs: number;
  endMs: number;
}

function splitIntoParagraphs(utterances: Utterance[]): VisualBlock[] {
  const blocks: VisualBlock[] = [];
  for (let ui = 0; ui < utterances.length; ui++) {
    const u = utterances[ui];
    if (!u.words || u.words.length === 0 || u.type === "non_verbal") {
      blocks.push({
        utteranceIndex: ui,
        utterance: u,
        words: u.words ?? [],
        startMs: u.start_ms,
        endMs: u.end_ms,
      });
      continue;
    }

    const words = u.words;
    let sentenceCount = 0;
    let blockStart = 0;

    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi].text;
      if (/[.!?]["'\u201D\u2019)]*$/.test(w)) {
        sentenceCount++;
        if (sentenceCount >= SENTENCES_PER_PARAGRAPH && wi < words.length - 1) {
          const slice = words.slice(blockStart, wi + 1);
          const nextWord = words[wi + 1];
          blocks.push({
            utteranceIndex: ui,
            utterance: u,
            words: slice,
            startMs: slice[0].ms,
            endMs: nextWord ? nextWord.ms : u.end_ms,
          });
          blockStart = wi + 1;
          sentenceCount = 0;
        }
      }
    }

    if (blockStart < words.length) {
      const slice = words.slice(blockStart);
      blocks.push({
        utteranceIndex: ui,
        utterance: u,
        words: slice,
        startMs: slice[0].ms,
        endMs: u.end_ms,
      });
    }
  }
  return blocks;
}

function getCurrentWordIndex(words: Word[], currentTimeMs: number): number {
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    const nextWord = words[i + 1];
    if (currentTimeMs >= word.ms && (nextWord === undefined || currentTimeMs < nextWord.ms)) {
      return i;
    }
  }
  return -1;
}

function getDisplayTag(u: Utterance): string {
  if (u.tag != null && u.tag !== "") return u.tag;
  return u.speaker === "interviewer" ? "Interviewer" : "Interviewee";
}

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
  const [autoScroll, setAutoScroll] = useState(false);
  const [fontSize, setFontSize] = useState<FontSize>("medium");

  const blocks = useMemo(() => splitIntoParagraphs(utterances), [utterances]);
  const fs = FONT_SIZES[fontSize];

  useEffect(() => {
    if (autoScroll && activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [currentTimeMs, autoScroll]);

  return (
    <div style={{ display: "flex", flexDirection: "column", maxHeight: "380px" }}>
      {/* Toolbar */}
      <div style={{
        display: "flex", alignItems: "center", gap: "8px", padding: "6px 8px",
        backgroundColor: "#16213e", borderRadius: "8px 8px 0 0", flexShrink: 0,
      }}>
        <button
          onClick={() => setAutoScroll((v) => !v)}
          style={{
            padding: "3px 8px", fontSize: "11px", fontWeight: 600, cursor: "pointer",
            border: "1px solid #475569", borderRadius: "4px",
            backgroundColor: autoScroll ? "#3b82f6" : "#334155",
            borderColor: autoScroll ? "#3b82f6" : "#475569",
            color: autoScroll ? "#fff" : "#94a3b8",
          }}
        >
          Auto-scroll {autoScroll ? "ON" : "OFF"}
        </button>

        <div style={{ marginLeft: "auto", display: "flex", gap: "2px", alignItems: "center" }}>
          {(["small", "medium", "large"] as const).map((size) => {
            const btnFontSize = size === "small" ? 10 : size === "medium" ? 13 : 16;
            return (
              <button
                key={size}
                onClick={() => setFontSize(size)}
                style={{
                  padding: "3px 8px", fontSize: `${btnFontSize}px`, fontWeight: 600,
                  cursor: "pointer", lineHeight: 1,
                  border: "1px solid #475569", borderRadius: "4px",
                  backgroundColor: fontSize === size ? "#3b82f6" : "#334155",
                  borderColor: fontSize === size ? "#3b82f6" : "#475569",
                  color: fontSize === size ? "#fff" : "#94a3b8",
                }}
              >
                A
              </button>
            );
          })}
        </div>
      </div>

      {/* Transcript body */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          backgroundColor: DARK_BG,
          borderRadius: "0 0 8px 8px",
          padding: "8px",
        }}
      >
        {blocks.map((block, bi) => {
          const u = block.utterance;
          const isActive = currentTimeMs >= block.startMs && currentTimeMs < block.endMs;
          const isNonVerbal = u.type === "non_verbal";
          const hasWords = block.words.length > 0;
          const currentWordIndex = isActive && hasWords
            ? getCurrentWordIndex(block.words, currentTimeMs)
            : -1;
          const baseTextColor = isNonVerbal ? NON_VERBAL : "#e2e8f0";

          const renderTextContent = () => {
            if (isActive && hasWords) {
              return block.words.map((word, wi) => {
                const isCurrent = wi === currentWordIndex;
                const isFuture = currentWordIndex >= 0 && wi > currentWordIndex;

                return (
                  <span
                    key={`${word.ms}-${wi}`}
                    onClick={(e) => { e.stopPropagation(); onSeek(word.ms); }}
                    style={{
                      color: isCurrent ? CURRENT_WORD_TEXT : isFuture ? FUTURE_WORD : baseTextColor,
                      backgroundColor: isCurrent ? CURRENT_WORD_BG : "transparent",
                      padding: isCurrent ? "2px 4px" : "0",
                      borderRadius: isCurrent ? "4px" : "0",
                      fontStyle: isNonVerbal ? "italic" : "normal",
                      cursor: "pointer",
                    }}
                  >
                    {word.text}{wi < block.words.length - 1 ? " " : ""}
                  </span>
                );
              });
            }

            if (hasWords) {
              return block.words.map((w) => w.text).join(" ");
            }
            return u.text;
          };

          return (
            <div
              key={`${block.startMs}-${bi}`}
              ref={isActive ? activeRef : null}
              onClick={() => onSeek(block.startMs)}
              style={{
                display: "flex",
                gap: "12px",
                padding: "7px 10px",
                marginBottom: "3px",
                backgroundColor: isActive ? ACTIVE_BG : ROW_BG,
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              <span
                style={{
                  flexShrink: 0,
                  width: "50px",
                  color: isNonVerbal ? NON_VERBAL
                    : u.speaker === "interviewer" ? INTERVIEWER : INTERVIEWEE,
                  fontWeight: 600,
                  fontSize: `${fs.tag}px`,
                }}
              >
                {getDisplayTag(u)}
              </span>
              <span
                style={{
                  flex: 1,
                  color: baseTextColor,
                  fontSize: `${fs.text}px`,
                  lineHeight: fs.lineHeight,
                  fontStyle: isNonVerbal ? "italic" : "normal",
                }}
              >
                {renderTextContent()}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
