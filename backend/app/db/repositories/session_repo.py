from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from datetime import datetime
from typing import Optional

from ..models import Session
from ...schemas.session import SessionCreate


class SessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, payload: SessionCreate) -> Session:
        session = Session(**payload.model_dump())
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        result = await self.db.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def end_session(self, session_id: str) -> Optional[Session]:
        session = await self.get(session_id)
        if not session:
            return None
        session.ended_at  = datetime.utcnow()
        session.is_active = False
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def list_active(self) -> list[Session]:
        result = await self.db.execute(
            select(Session).where(Session.is_active == True)
        )
        return result.scalars().all()
