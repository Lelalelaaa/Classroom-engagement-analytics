from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from ...db.database import get_db
from ...db.models import Session, Classroom, AggregatedEngagementMetric
from ...db.repositories.metrics_repo import MetricsRepository
from ...schemas.metrics import MetricRecordSchema

router = APIRouter()


@router.get("/sessions/{session_id}/metrics", response_model=list[MetricRecordSchema])
async def get_session_metrics(
    session_id: str,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """Fetch historical engagement snapshots for a session."""
    repo    = MetricsRepository(db)
    records = await repo.get_session_metrics(session_id, limit=limit)
    return records


@router.get("/classrooms/summary")
async def classrooms_summary(db: AsyncSession = Depends(get_db)):
    """Aggregate engagement stats per classroom (for Admin dashboard)."""
    result = await db.execute(
        select(
            Classroom.name.label("classroom_name"),
            func.avg(AggregatedEngagementMetric.class_engagement).label("avg_engagement"),
            func.count(Session.id.distinct()).label("sessions_today"),
        )
        .join(Session, Session.classroom_id == Classroom.id)
        .join(AggregatedEngagementMetric, AggregatedEngagementMetric.session_id == Session.id)
        .group_by(Classroom.id, Classroom.name)
    )
    rows = result.all()
    return [
        {
            "classroom_name": r.classroom_name,
            "avg_engagement": round(r.avg_engagement or 0, 1),
            "sessions_today": r.sessions_today,
        }
        for r in rows
    ]
