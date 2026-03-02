export interface VideoEntry {
  id: string;
  int_code: number;
  tape: number;
  youtube_url: string | null;
  downloaded: boolean;
}

export interface VideoListResponse {
  total: number;
  videos: VideoEntry[];
}

export interface Word {
  text: string;
  ms: number;
}

export interface Utterance {
  speaker: "interviewer" | "interviewee";
  tag: string;
  text: string;
  start_ms: number;
  end_ms: number;
  words?: Word[];
  type?: "non_verbal";
}

export interface AudioVADSegment {
  start: number;
  end: number;
  valence: number;
  arousal: number;
  dominance: number;
}

export interface AudioVADData {
  video_id: string;
  segments: AudioVADSegment[];
}

export interface EyegazeVADSegment {
  timestamp: number;
  valence: number;
  arousal: number;
  dominance: number;
}

export interface EyegazeVADData {
  video_id: string;
  segments: EyegazeVADSegment[];
}

export interface AnnotationEntry {
  start_sec: number;
  end_sec: number;
  label: string;
  key: number;
}

export interface AnnotationData {
  video_id: string;
  annotator: string;
  created_at: string;
  annotations: AnnotationEntry[];
}

export interface DownloadStatus {
  video_id: string;
  status: "none" | "pending" | "downloading" | "complete" | "failed";
  error: string | null;
}

export const EMOTION_LABELS: Record<number, string> = {
  1: "very_happy",
  2: "happy",
  3: "neutral",
  4: "sad",
  5: "very_sad",
};

export const EMOTION_COLORS: Record<string, string> = {
  very_happy: "#22c55e",
  happy: "#86efac",
  neutral: "#94a3b8",
  sad: "#93c5fd",
  very_sad: "#3b82f6",
};
