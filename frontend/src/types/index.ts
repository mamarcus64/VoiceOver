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
  tag: string | null;
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

export interface SmileSegment {
  start_ts: number;
  end_ts: number;
  peak_r: number;
  mean_r: number;
  mass_r: number;
}

export interface SmilingSegmentsData {
  video_id: string;
  fps: number;
  total_frames: number;
  total_duration_sec: number;
  smoothing_sigma_sec: number;
  threshold: number;
  min_duration_sec: number;
  num_segments: number;
  total_smile_sec: number;
  segments: SmileSegment[];
}

export interface SmileParams {
  intensityThreshold: number;
  mergeDistance: number;
  minDuration: number;
  contextBefore: number;
  contextAfter: number;
}

export const DEFAULT_SMILE_PARAMS: SmileParams = {
  intensityThreshold: 1.8,
  mergeDistance: 0.5,
  minDuration: 0.5,
  contextBefore: 3.0,
  contextAfter: 2.0,
};

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

export interface SmileTask {
  task_number: number;
  video_id: string;
  smile_start: number;
  smile_end: number;
  peak_r: number;
  mean_r: number;
  total_tasks: number;
  available_tasks: number;
  video_downloaded: boolean;
}

export interface SmileAnnotationEntry {
  label: string;
  timestamp: string;
}

export interface SmileAnnotations {
  annotator: string;
  annotations: Record<string, SmileAnnotationEntry>;
}

export interface SmileConfigData {
  intensityThreshold: number;
  mergeDistance: number;
  minDuration: number;
  maxPerVideo: number;
  contextBefore: number;
  contextAfter: number;
}

export interface SmilePreviewStats {
  total_tasks: number;
  videos_with_tasks: number;
  tasks_per_video_mean: number;
  tasks_per_video_median: number;
  tasks_per_video_max: number;
  params: Record<string, number>;
}

export type SmileLabel = "genuine" | "polite" | "masking" | "not_a_smile";

export const SMILE_LABELS: { key: SmileLabel; display: string; color: string }[] = [
  { key: "genuine", display: "Genuine Smile", color: "#22c55e" },
  { key: "polite", display: "Polite Smile", color: "#3b82f6" },
  { key: "masking", display: "Masking Smile", color: "#f59e0b" },
  { key: "not_a_smile", display: "Not a Smile", color: "#64748b" },
];
