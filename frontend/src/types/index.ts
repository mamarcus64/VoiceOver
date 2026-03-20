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

/** Per-eye gaze direction in head space (OpenFace-style gaze_0 / gaze_1). */
export interface EyegazeVectorSample {
  t: number;
  g0: [number, number, number];
  g1: [number, number, number];
}

export interface EyegazeVectorsData {
  video_id: string;
  samples: EyegazeVectorSample[];
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
  notes?: string;
  runner_up?: string;
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

export interface SmileAgreementPairwise {
  annotator_a: string;
  annotator_b: string;
  n_tasks: number;
  cohen_kappa: number | null;
  percent_agreement: number | null;
  confusion: number[][];
}

export interface SmileAgreementStats {
  annotators: string[];
  valid_labels: string[];
  per_annotator_counts: Record<string, Record<string, number>>;
  tasks_with_any_label: number;
  tasks_fully_labeled: number;
  percent_full_agreement: number | null;
  fleiss_kappa: number | null;
  pairwise: SmileAgreementPairwise[];
}

export const SMILE_LABELS: { key: SmileLabel; display: string; color: string; desc: string }[] = [
  { key: "genuine", display: "Genuine Smile", color: "#22c55e",
    desc: "A smile that demonstrates true happiness. Look for laughter, \u201Csparkles\u201D in the eyes, closed eyes, and contextually happy words and speech." },
  { key: "polite", display: "Polite Smile", color: "#3b82f6",
    desc: "A smile used as a social function. Look for if the interviewer asked a recent question, if the subject is responding to something, or if the smile indicates conversational signaling. Nodding or head tilting is a potential sign of a polite smile." },
  { key: "masking", display: "Masking Smile", color: "#f59e0b",
    desc: "A smile used for complex emotion processing. Look for signs of trauma, irony, contradictory or negative emotions, or otherwise non-happy behaviors." },
  { key: "not_a_smile", display: "Not a Smile", color: "#64748b",
    desc: "For segments where a smile is misidentified." },
];
