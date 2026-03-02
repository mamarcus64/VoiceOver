import { useState, useRef, useCallback, useEffect } from "react";
import type { AnnotationEntry } from "../types";
import { EMOTION_LABELS } from "../types";

export function useAnnotation(getCurrentTime: () => number) {
  const [annotations, setAnnotations] = useState<AnnotationEntry[]>([]);
  const [annotator, setAnnotator] = useState<string>(
    () => localStorage.getItem("voiceover_annotator") || ""
  );
  const activeRef = useRef<{ key: number; startSec: number } | null>(null);

  useEffect(() => {
    if (annotator) localStorage.setItem("voiceover_annotator", annotator);
  }, [annotator]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.repeat) return;
      if (document.activeElement?.tagName === "INPUT") return;
      const key = parseInt(e.key);
      if (key >= 1 && key <= 5 && !activeRef.current) {
        activeRef.current = { key, startSec: getCurrentTime() };
      }
    },
    [getCurrentTime]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent) => {
      const key = parseInt(e.key);
      if (key >= 1 && key <= 5 && activeRef.current?.key === key) {
        const entry: AnnotationEntry = {
          start_sec: activeRef.current.startSec,
          end_sec: getCurrentTime(),
          label: EMOTION_LABELS[key],
          key,
        };
        if (entry.end_sec > entry.start_sec) {
          setAnnotations((prev) => [...prev, entry]);
        }
        activeRef.current = null;
      }
    },
    [getCurrentTime]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp]);

  const clearAnnotations = useCallback(() => setAnnotations([]), []);

  const loadAnnotations = useCallback((data: AnnotationEntry[]) => {
    setAnnotations(data);
  }, []);

  return {
    annotations,
    annotator,
    setAnnotator,
    clearAnnotations,
    loadAnnotations,
  };
}
