import json
import redis.asyncio as aioredis

from ..config import settings


async def check_and_fire_alerts(
    redis: aioredis.Redis,
    session_id: str,
    metrics: dict,
) -> None:
    """
    Fire a low_engagement alert when class engagement drops below threshold.
    A cooldown key prevents alert spam: only fires once per ALERT_COOLDOWN seconds.
    """
    score     = metrics.get("class_engagement", 100.0)
    alert_key = f"alert:{session_id}:low_engagement"

    if score < settings.ALERT_THRESHOLD:
        already_alerted = await redis.get(alert_key)
        if not already_alerted:
            alert = {
                "type":       "low_engagement",
                "session_id": session_id,
                "score":      score,
                "message":    f"\u26a0\ufe0f Class engagement dropped to {score:.0f}%",
            }
            await redis.publish(f"alerts:{session_id}", json.dumps(alert))
            await redis.setex(alert_key, settings.ALERT_COOLDOWN, "1")
