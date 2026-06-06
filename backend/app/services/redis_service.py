import json
from typing import AsyncGenerator

from ..config import settings

_redis_client = None
METRICS_TTL_SECONDS = 60 * 60


async def init_redis() -> None:
    global _redis_client

    if settings.REDIS_URL == "memory://":
        # Local dev: use in-memory implementation (no Redis server needed)
        from .memory_redis import InMemoryRedis
        _redis_client = InMemoryRedis()
        print("[redis_service] Using in-memory Redis (dev mode)")
        return

    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await _redis_client.ping()
        print("[redis_service] Connected to Redis server")
    except Exception as e:
        print(f"[redis_service] Redis unavailable ({e}), falling back to in-memory")
        from .memory_redis import InMemoryRedis
        _redis_client = InMemoryRedis()


async def close_redis() -> None:
    if _redis_client:
        await _redis_client.aclose()


async def get_redis():
    return _redis_client


async def publish_metrics(redis, session_id: str, metrics: dict) -> None:
    key     = f"session:{session_id}:live"
    channel = f"channel:session:{session_id}"
    payload = json.dumps(metrics)
    await redis.setex(key, METRICS_TTL_SECONDS, payload)
    await redis.publish(channel, payload)


async def get_latest_metrics(redis, session_id: str) -> dict | None:
    raw = await redis.get(f"session:{session_id}:live")
    return json.loads(raw) if raw else None


async def subscribe_metrics(session_id: str) -> AsyncGenerator[str, None]:
    subscriber = _redis_client.pubsub()
    channel    = f"channel:session:{session_id}"
    await subscriber.subscribe(channel)
    async for message in subscriber.listen():
        if message["type"] == "message":
            yield message["data"]
