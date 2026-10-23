"""ORM models.

Validation lives in app.domain.schemas; the CHECK constraints here are a second line of defence so
that bad data cannot reach the table even if some future code path skips the schemas.
"""

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    String,
    TypeDecorator,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base

SEX_VALUES = ("Male", "Female", "Other", "Decline to Answer")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Stores UTC and always hands back timezone-aware datetimes (SQLite drops the offset)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime passed to UTCDateTime")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=timezone.utc) if value is not None else None


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))
    date_of_birth: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(20))
    phone_number: Mapped[str] = mapped_column(String(10))
    email: Mapped[str | None] = mapped_column(String(254))
    address_line_1: Mapped[str] = mapped_column(String(200))
    address_line_2: Mapped[str | None] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(2))
    zip_code: Mapped[str] = mapped_column(String(10))
    insurance_provider: Mapped[str | None] = mapped_column(String(100))
    insurance_member_id: Mapped[str | None] = mapped_column(String(30))
    preferred_language: Mapped[str] = mapped_column(String(50), default="English")
    emergency_contact_name: Mapped[str | None] = mapped_column(String(100))
    emergency_contact_phone: Mapped[str | None] = mapped_column(String(10))

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    # On insert, reuse created_at so a fresh record has identical timestamps.
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda ctx: ctx.get_current_parameters()["created_at"],
        onupdate=utcnow,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    __table_args__ = (
        CheckConstraint("length(first_name) BETWEEN 1 AND 50", name="ck_first_name_len"),
        CheckConstraint("length(last_name) BETWEEN 1 AND 50", name="ck_last_name_len"),
        CheckConstraint(f"sex IN {SEX_VALUES!r}", name="ck_sex_enum"),
        CheckConstraint("length(phone_number) = 10", name="ck_phone_len"),
        CheckConstraint(
            "emergency_contact_phone IS NULL OR length(emergency_contact_phone) = 10",
            name="ck_emergency_phone_len",
        ),
        CheckConstraint("length(city) BETWEEN 1 AND 100", name="ck_city_len"),
        CheckConstraint("length(state) = 2", name="ck_state_len"),
        CheckConstraint("length(zip_code) IN (5, 10)", name="ck_zip_len"),
        # Lookups the API filters on, and the voice agent's returning-caller check (phone).
        Index("ix_patients_last_name", "last_name"),
        Index("ix_patients_date_of_birth", "date_of_birth"),
        Index("ix_patients_phone_number", "phone_number"),
    )


class CallRecord(Base):
    """One phone call: its transcript, the fields captured so far, and the patient it produced.

    A call that drops before saving keeps its draft_fields, so the next call from the same
    number can offer to resume instead of starting from scratch.
    """

    __tablename__ = "call_records"

    call_sid: Mapped[str] = mapped_column(String(64), primary_key=True)
    from_number: Mapped[str | None] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    patient_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("patients.patient_id"))
    draft_fields: Mapped[dict] = mapped_column(JSON, default=dict)
    transcript: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    __table_args__ = (
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'abandoned', 'resumed')", name="ck_call_status"
        ),
        Index("ix_call_records_from_number", "from_number"),
        Index("ix_call_records_patient_id", "patient_id"),
    )
