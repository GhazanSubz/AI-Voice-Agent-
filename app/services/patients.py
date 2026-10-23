"""Patient service layer: the only code that reads or writes the patients table.

Both the REST API and the voice agent call this, so persistence rules (soft delete, update
timestamps, duplicate lookup) live in exactly one place. Methods return PatientOut DTOs, never
ORM objects, so no session state leaks to callers.
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import date
from typing import Iterator

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Patient, utcnow
from app.domain.schemas import PatientCreate, PatientOut, PatientUpdate

log = logging.getLogger(__name__)


class PatientNotFound(Exception):
    pass


class PersistenceError(Exception):
    """The database could not complete the operation. Callers should surface a retryable error."""


class PatientService:
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
            log.exception("database operation failed")
            raise PersistenceError(str(exc)) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _get_active(session: Session, patient_id: uuid.UUID) -> Patient:
        patient = session.get(Patient, patient_id)
        if patient is None or patient.deleted_at is not None:
            raise PatientNotFound(str(patient_id))
        return patient

    def create(self, data: PatientCreate) -> PatientOut:
        with self._session() as session:
            patient = Patient(**data.model_dump())
            session.add(patient)
            session.flush()
            log.info("patient created", extra={"patient_id": str(patient.patient_id)})
            return PatientOut.model_validate(patient)

    def get(self, patient_id: uuid.UUID) -> PatientOut:
        with self._session() as session:
            return PatientOut.model_validate(self._get_active(session, patient_id))

    def search(
        self,
        *,
        last_name: str | None = None,
        date_of_birth: date | None = None,
        phone_number: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PatientOut]:
        query = select(Patient).where(Patient.deleted_at.is_(None))
        if last_name:
            query = query.where(func.lower(Patient.last_name) == last_name.lower())
        if date_of_birth:
            query = query.where(Patient.date_of_birth == date_of_birth)
        if phone_number:
            query = query.where(Patient.phone_number == phone_number)
        query = query.order_by(Patient.created_at.desc()).limit(limit).offset(offset)

        with self._session() as session:
            return [PatientOut.model_validate(p) for p in session.scalars(query)]

    def find_by_phone(self, phone_number: str) -> list[PatientOut]:
        """Active patients registered with this phone (a household can share one number)."""
        return self.search(phone_number=phone_number)

    def update(self, patient_id: uuid.UUID, data: PatientUpdate) -> PatientOut:
        with self._session() as session:
            patient = self._get_active(session, patient_id)
            for field, value in data.changes().items():
                setattr(patient, field, value)
            if patient.preferred_language is None:
                patient.preferred_language = "English"
            session.flush()
            log.info("patient updated", extra={"patient_id": str(patient_id)})
            return PatientOut.model_validate(patient)

    def soft_delete(self, patient_id: uuid.UUID) -> PatientOut:
        with self._session() as session:
            patient = self._get_active(session, patient_id)
            patient.deleted_at = utcnow()
            session.flush()
            log.info("patient soft-deleted", extra={"patient_id": str(patient_id)})
            return PatientOut.model_validate(patient)

    def count_active(self) -> int:
        with self._session() as session:
            return session.scalar(
                select(func.count()).select_from(Patient).where(Patient.deleted_at.is_(None))
            )
