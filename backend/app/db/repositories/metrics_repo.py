from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import Optional

from ..models import AggregatedEngagementMetric


class MetricsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_session_metrics(
        self, session_id: str, limit: int = 200
    ) -> list[AggregatedEngagementMetric]:
        result = await self.db.execute(
            select(AggregatedEngagementMetric)
            .where(AggregatedEngagementMetric.session_id == session_id)
            .order_by(AggregatedEngagementMetric.recorded_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_classroom_avg(self, classroom_id: str) -> Optional[float]:
        from sqlmodel import func
        from ..models import Session
        result = await self.db.execute(
            select(func.avg(AggregatedEngagementMetric.class_engagement))
            .join(Session, Session.id == AggregatedEngagementMetric.session_id)
            .where(Session.classroom_id == classroom_id)
        )
        return result.scalar()
