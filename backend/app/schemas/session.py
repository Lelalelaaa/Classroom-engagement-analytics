from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SessionCreate(BaseModel):
    classroom_id: str
    teacher_name: Optional[str] = None
    subject:      Optional[str] = None


class SessionResponse(BaseModel):
    id:           str
    classroom_id: str
    teacher_name: Optional[str]
    subject:      Optional[str]
    started_at:   datetime
    ended_at:     Optional[datetime]
    is_active:    bool

    class Config:
        from_attributes = True
