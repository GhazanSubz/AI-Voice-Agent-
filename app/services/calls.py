"""Call records: transcripts, in-progress drafts and resume-after-drop support."""

import logging
import uuid
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import CallRecord, utcnow
from app.services.patients import PersistenceError

log = logging.getLogger(__name__)

RESUME_WINDOW = timedelta(hours=24)


class CallService:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            log.exception("call record operation failed")
            raise PersistenceError(str(exc)) from exc
        finally:
            session.close()

    def start(self, call_sid: str, from_number: str | None) -> None:
        with self._session() as session:
            session.merge(CallRecord(call_sid=call_sid, from_number=from_number))

    def find_resumable_draft(self, from_number: str | None, exclude_sid: str) -> CallRecord | None:
        """Most recent dropped call from this number that captured something but never saved."""
        if not from_number:
            return None
        query = (
            select(CallRecord)
            .where(
                CallRecord.from_number == from_number,
                CallRecord.status == "abandoned",
                CallRecord.patient_id.is_(None),
                CallRecord.call_sid != exclude_sid,
                CallRecord.started_at >= utcnow() - RESUME_WINDOW,
            )
            .order_by(CallRecord.started_at.desc())
            .limit(1)
        )
        with self._session() as session:
            record = session.scalars(query).first()
            return record if record and record.draft_fields else None

    def mark_resumed(self, call_sid: str) -> None:
        with self._session() as session:
            record = session.get(CallRecord, call_sid)
            if record:
                record.status = "resumed"

    def save_progress(self, call_sid: str, draft_fields: dict[str, Any], transcript: list | None = None) -> None:
        with self._session() as session:
            record = session.get(CallRecord, call_sid)
            if record:
                record.draft_fields = draft_fields
                if transcript is not None:
                    record.transcript = transcript

    def get_draft(self, call_sid: str) -> dict[str, Any] | None:
        with self._session() as session:
            record = session.get(CallRecord, call_sid)
            return dict(record.draft_fields) if record and record.draft_fields else None

    def finish(
        self,
        call_sid: str,
        *,
        patient_id: uuid.UUID | None,
        draft_fields: dict[str, Any],
        transcript: list,
    ) -> None:
        with self._session() as session:
            record = session.get(CallRecord, call_sid)
            if record is None:
                return
            record.status = "completed" if patient_id else "abandoned"
            record.patient_id = patient_id
            record.draft_fields = draft_fields
            record.transcript = transcript
            record.ended_at = utcnow()

    def list_for_patient(self, patient_id: uuid.UUID) -> list[dict[str, Any]]:
        query = (
            select(CallRecord)
            .where(CallRecord.patient_id == patient_id)
            .order_by(CallRecord.started_at.desc())
        )
        with self._session() as session:
            return [_call_dict(r) for r in session.scalars(query)]

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        query = select(CallRecord).order_by(CallRecord.started_at.desc()).limit(limit)
        with self._session() as session:
            return [_call_dict(r) for r in session.scalars(query)]


def _call_dict(record: CallRecord) -> dict[str, Any]:
    return {
        "call_sid": record.call_sid,
        "from_number": record.from_number,
        "status": record.status,
        "patient_id": str(record.patient_id) if record.patient_id else None,
        "started_at": record.started_at.isoformat().replace("+00:00", "Z"),
        "ended_at": record.ended_at.isoformat().replace("+00:00", "Z") if record.ended_at else None,
        "transcript": record.transcript,
    }
