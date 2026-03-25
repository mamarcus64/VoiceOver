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
  not_a_smile?: boolean;
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

export type SmileLabel = "felt" | "false" | "miserable" | "not_a_smile";
export type SmileEmotionLabel = "felt" | "false" | "miserable";

export interface SmileAgreementPairwise {
  annotator_a: string;
  annotator_b: string;
  n_tasks: number;
  cohen_kappa: number | null;
  coarse_cohen_kappa: number | null;
  percent_agreement: number | null;
  coarse_percent_agreement: number | null;
  confusion: number[][];
  coarse_confusion: number[][];
}

export interface SmileModePairwise {
  annotator_a: string;
  annotator_b: string;
  n_tasks: number;
  cohen_kappa: number | null;
  percent_agreement: number | null;
  confusion: number[][];
}

export interface SmileModeStats {
  labels: string[];
  tasks_fully_labeled: number;
  fleiss_kappa: number | null;
  percent_full_agreement: number | null;
  pairwise: SmileModePairwise[];
}

export type SmileModeKey = "fine" | "coarse" | "smile_fine" | "smile_coarse" | "binary";

export interface SmileAgreementStats {
  annotators: string[];
  valid_labels: string[];
  coarse_labels: string[];
  per_annotator_counts: Record<string, Record<string, number>>;
  tasks_with_any_label: number;
  tasks_fully_labeled: number;
  percent_full_agreement: number | null;
  coarse_percent_full_agreement: number | null;
  fleiss_kappa: number | null;
  coarse_fleiss_kappa: number | null;
  pairwise: SmileAgreementPairwise[];
  modes: Record<SmileModeKey, SmileModeStats>;
}

export interface AU12ScatterPoint {
  task_number: number;
  annotator: string;
  mean_r: number;
  peak_r: number;
  is_not_a_smile: boolean;
  label: string;
}

export const SMILE_LABELS: { key: SmileEmotionLabel; display: string; color: string; desc: string }[] = [
  {
    key: "felt",
    display: "Felt Smile",
    color: "#22c55e",
    desc: "A genuine smile of positive emotion (Duchenne smile). Look for lip corners pulled up AND muscles tightened around the eyes — raised cheeks, bagged skin below the eyes, crow's feet wrinkles.",
  },
  {
    key: "false",
    display: "False Smile",
    color: "#3b82f6",
    desc: "A deliberate smile to project positivity that isn't felt. Typically lacks eye involvement, may be asymmetrical, have abrupt timing, or show traces of a negative emotion leaking through.",
  },
  {
    key: "miserable",
    display: "Miserable Smile",
    color: "#f59e0b",
    desc: "A smile that acknowledges negative emotion without hiding it — not deceptive. Often asymmetrical, superimposed on or following a visible negative expression (sadness, distress, fear).",
  },
];

export const ALL_SMILE_LABELS: { key: SmileLabel; display: string; color: string }[] = [
  ...SMILE_LABELS.map(({ key, display, color }) => ({ key: key as SmileLabel, display, color })),
  { key: "not_a_smile", display: "Not a Smile", color: "#64748b" },
];

/** Labels that were used in the pilot study (archived, read-only). */
export const PILOT_SMILE_LABELS: { key: string; display: string; color: string }[] = [
  { key: "genuine", display: "Genuine Smile", color: "#22c55e" },
  { key: "polite", display: "Polite Smile", color: "#3b82f6" },
  { key: "masking", display: "Masking Smile", color: "#f59e0b" },
  { key: "not_a_smile", display: "Not a Smile", color: "#64748b" },
];
