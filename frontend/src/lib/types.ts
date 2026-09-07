export type AttentionState = 'on_task' | 'desk_work' | 'off_task' | 'unknown';

export interface FaceOverlay {
  bbox:          [number, number, number, number]; // normalised [x, y, w, h]
  yaw:           number;   // degrees of deviation from this student's calibrated reference
  pitch:         number;   // degrees of deviation from this student's calibrated reference
  state:         AttentionState;
  on_task_ratio: number | null;   // 0–1, null until the rolling window has data
  measured:      boolean;         // false for `unknown` — excluded from the score
  has_gaze:      boolean;         // eye-gaze model contributed to this face
}

export interface LiveMetrics {
  session_id:        string;
  on_task_ratio:     number;   // 0–1, mean over tracked students
  class_engagement:  number;   // 0–100
  student_count:     number;
  tracked_count:     number;   // students with enough history to score
  detected_count:    number;   // faces in this frame
  expected_count:    number | null;
  engaged_count:     number;
  below_floor_count: number;
  state_counts:      Record<AttentionState, number>;
  is_calibrating:    boolean;
  gaze_model_loaded: boolean;
  faces?:            FaceOverlay[];
  timestamp?:        number;
}

export interface Session {
  id:             string;
  classroom_id:   string;
  teacher_name:   string | null;
  subject:        string | null;
  started_at:     string;
  ended_at:       string | null;
  is_active:      boolean;
  avg_engagement?: number;
}

export interface Classroom {
  id:          string;
  name:        string;
  school_name: string;
  capacity:    number;
}

export interface Alert {
  type:       string;
  session_id: string;
  score:      number;
  message:    string;
}

export interface ClassroomSummary {
  classroom_name: string;
  avg_engagement: number;
  sessions_today: number;
}

// ── Model feedback (offline: teacher rates a recorded lesson) ────────────────
export type SegmentRating = 'focused' | 'mixed' | 'off';
export type RatingConfidence = 'sure' | 'unsure';

export interface SegmentReview {
  rating: SegmentRating | null;
  confidence: RatingConfidence;
  note: string;
}
