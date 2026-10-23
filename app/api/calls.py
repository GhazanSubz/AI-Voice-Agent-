"""Read-only REST endpoint for call records (outcome + transcript), used by the dashboard."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from app.api.envelope import Envelope

router = APIRouter(prefix="/calls", tags=["calls"])


class TranscriptLine(BaseModel):
    speaker: Literal["caller", "agent"]
    text: str


class CallOut(BaseModel):
    call_sid: str
    from_number: str | None
    status: Literal["in_progress", "completed", "abandoned", "resumed"]
    patient_id: uuid.UUID | None
    started_at: datetime
    ended_at: datetime | None
    transcript: list[TranscriptLine]


@router.get("", response_model=Envelope[list[CallOut]])
def list_calls(request: Request, limit: Annotated[int, Query(ge=1, le=200)] = 50):
    """Most recent calls first. A call is `completed` when it registered or updated a patient."""
    return Envelope(data=request.app.state.call_service.list_recent(limit=limit))
