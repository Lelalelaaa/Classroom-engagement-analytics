from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
import uuid


class Classroom(SQLModel, table=True):
    __tablename__ = "classrooms"

    id:          str      = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name:        str      = Field(max_length=100)
    school_name: str      = Field(max_length=200)
    capacity:    int      = Field(ge=1, le=60)
    created_at:  datetime = Field(default_factory=datetime.utcnow)

    sessions: List["Session"] = Relationship(back_populates="classroom")


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id:            str                = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    classroom_id:  str                = Field(foreign_key="classrooms.id", index=True)
    teacher_name:  Optional[str]      = Field(default=None, max_length=100)
    subject:       Optional[str]      = Field(default=None, max_length=100)
    started_at:    datetime           = Field(default_factory=datetime.utcnow)
    ended_at:      Optional[datetime] = Field(default=None)
    is_active:     bool               = Field(default=True)

    # Computed session summary (set when session ends)
    avg_engagement:    Optional[float] = Field(default=None)
    peak_engagement:   Optional[float] = Field(default=None)
    trough_engagement: Optional[float] = Field(default=None)
    total_yawn_events: Optional[int]   = Field(default=None)
    max_students_seen: Optional[int]   = Field(default=None)

    classroom: Optional[Classroom] = Relationship(back_populates="sessions")
    metrics: List["AggregatedEngagementMetric"] = Relationship(back_populates="session")


class AggregatedEngagementMetric(SQLModel, table=True):
    """
    Stores anonymous, class-level engagement snapshots every N seconds.
    NEVER stores individual student data or raw video frames.
    """
    __tablename__ = "aggregated_engagement_metrics"

    id:               Optional[int] = Field(default=None, primary_key=True)
    session_id:       str           = Field(foreign_key="sessions.id", index=True)
    recorded_at:      datetime      = Field(default_factory=datetime.utcnow, index=True)

    # Anonymous aggregate metrics only
    student_count:    int   = Field(ge=0)
    class_engagement: float = Field(ge=0.0, le=100.0)
    engaged_count:    int   = Field(ge=0)
    yawn_rate:        float = Field(ge=0.0, le=100.0)

    # Emotion distribution (aggregated counts — not per-student)
    emotion_engaged_happy: int = Field(default=0)
    emotion_neutral:       int = Field(default=0)
    emotion_disengaged:    int = Field(default=0)
    emotion_distressed:    int = Field(default=0)

    session: Optional[Session] = Relationship(back_populates="metrics")
