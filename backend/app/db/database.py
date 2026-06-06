from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from ..config import settings


def _build_async_url(url: str) -> str:
    """Convert sync DB URL to async driver URL."""
    if url.startswith("sqlite:///"):
        # SQLite: use aiosqlite driver
        return url.replace("sqlite:///", "sqlite+aiosqlite:///")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


_async_url = _build_async_url(settings.DATABASE_URL)

# SQLite needs check_same_thread=False passed via connect_args
_connect_args = {"check_same_thread": False} if "sqlite" in _async_url else {}

engine = create_async_engine(
    _async_url,
    echo=False,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
