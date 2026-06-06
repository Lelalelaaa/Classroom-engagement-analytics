from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LiveMetricsSchema(BaseModel):
    session_id:           str
    student_count:        int
    class_engagement:     float
    engaged_count:        int
    yawn_rate:            float
    emotion_distribution: dict


class MetricRecordSchema(BaseModel):
    id:               int
    session_id:       str
    recorded_at:      datetime
    student_count:    int
    class_engagement: float
    engaged_count:    int
    yawn_rate:        float

    class Config:
        from_attributes = True
