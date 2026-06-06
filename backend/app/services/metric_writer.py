import asyncio
import json
from datetime import datetime

from ..db.database import AsyncSessionLocal
from ..db.models import AggregatedEngagementMetric
from ..services.redis_service import get_redis
from ..config import settings


async def periodic_metric_writer(active_sessions: set[str]) -> None:
    """
    Background task: reads latest metrics from Redis and persists to PostgreSQL.
    Runs every METRIC_WRITE_INTERVAL seconds.
    Only anonymous aggregated class-level data is written — no individual student info.
    """
    redis = await get_redis()
    while True:
        await asyncio.sleep(settings.METRIC_WRITE_INTERVAL)
        async with AsyncSessionLocal() as db:
            for session_id in list(active_sessions):
                raw = await redis.get(f"session:{session_id}:live")
                if not raw:
                    continue
                m      = json.loads(raw)
                record = AggregatedEngagementMetric(
                    session_id       =session_id,
                    recorded_at      =datetime.utcnow(),
                    student_count    =m.get("student_count", 0),
                    class_engagement =m.get("class_engagement", 0.0),
                    engaged_count    =m.get("engaged_count", 0),
                    yawn_rate        =m.get("yawn_rate", 0.0),
                    **_unpack_emotions(m.get("emotion_distribution", {})),
                )
                db.add(record)
            await db.commit()


def _unpack_emotions(dist: dict) -> dict:
    return {
        "emotion_engaged_happy": dist.get("engaged_happy", 0),
        "emotion_neutral":       dist.get("neutral", 0),
        "emotion_disengaged":    dist.get("disengaged", 0),
        "emotion_distressed":    dist.get("distressed", 0),
    }
