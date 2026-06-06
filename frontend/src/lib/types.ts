export interface FaceOverlay {
  bbox:    [number, number, number, number]; // normalised [x, y, w, h]
  yaw:     number;   // radians from yakhyo gaze model
  pitch:   number;   // radians from yakhyo gaze model
  emotion: string;   // FER+ label
  gaze:    string;   // human-readable direction
  engaged: boolean;
  score:   number;   // 0–1
}

export interface LiveMetrics {
  session_id:           string;
  student_count:        number;
  class_engagement:     number;   // 0–100
  engaged_count:        number;
  yawn_rate:            number;   // 0–100
  emotion_distribution: Record<string, number>;
  faces?:               FaceOverlay[];
  timestamp?:           number;
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
