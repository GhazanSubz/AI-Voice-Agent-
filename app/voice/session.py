"""Per-call registration state.

Design rule: the LLM runs the conversation, this class owns the facts. Every value the caller
gives goes through update_fields(), which validates it with the same schemas the REST API uses.
The LLM gets back what was accepted, what was rejected (with a speakable reason) and what is
still missing, so it never has to remember or re-validate anything itself. Saving is refused
unless all required fields are present, the readback happened, and the caller confirmed.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable

from pydantic import ValidationError

from app.domain.schemas import REQUIRED_FIELDS, PatientCreate, PatientFields, PatientOut, PatientUpdate
from app.domain.validators import US_STATES, normalize_us_phone
from app.services.patients import PatientNotFound, PatientService, PersistenceError

log = logging.getLogger(__name__)

ALL_FIELDS = tuple(PatientFields.model_fields)
OPTIONAL_FIELDS = tuple(f for f in ALL_FIELDS if f not in REQUIRED_FIELDS)

FIELD_LABELS = {
    "first_name": "first name",
    "last_name": "last name",
    "date_of_birth": "date of birth",
    "sex": "sex",
    "phone_number": "phone number",
    "email": "email",
    "address_line_1": "street address",
    "address_line_2": "apartment or unit",
    "city": "city",
    "state": "state",
    "zip_code": "ZIP code",
    "insurance_provider": "insurance provider",
    "insurance_member_id": "insurance member ID",
    "preferred_language": "preferred language",
    "emergency_contact_name": "emergency contact name",
    "emergency_contact_phone": "emergency contact phone",
}

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}


class Phase(str, Enum):
    COLLECTING = "collecting_required"
    READY_FOR_READBACK = "ready_for_readback"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SAVED = "saved"


@dataclass
class RegistrationSession:
    call_sid: str
    patients: PatientService
    caller_number: str | None = None
    language: str = "en"
    fields: dict[str, Any] = field(default_factory=dict)
    phase: Phase = Phase.COLLECTING
    # When the caller is a returning patient and chose to update their record.
    update_target: uuid.UUID | None = None
    saved_patient: PatientOut | None = None
    on_change: Callable[["RegistrationSession"], None] | None = None
    # A save that arrives sooner than this after the agent asked "is that correct?" cannot be the
    # caller's answer (the readback alone takes longer to say), so it is the LLM confirming on the
    # caller's behalf, which live testing showed gpt-oss doing. 0 disables the check.
    min_confirm_secs: float = 3.0
    clock: Callable[[], float] = time.monotonic
    _checked_phones: set[str] = field(default_factory=set)
    _asked_to_confirm_at: float | None = None

    def __post_init__(self):
        if self.caller_number:
            try:
                self.caller_number = normalize_us_phone(self.caller_number)
            except ValueError:
                self.caller_number = None

    # ---- tool: update_fields -------------------------------------------------------------

    def update_fields(self, raw: dict[str, Any]) -> dict[str, Any]:
        accepted: dict[str, str] = {}
        rejected: dict[str, str] = {}
        cleared: list[str] = []

        for name, value in raw.items():
            if name not in ALL_FIELDS:
                rejected[name] = "not a field we collect"
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                if name in REQUIRED_FIELDS:
                    rejected[name] = f"{FIELD_LABELS[name]} is required"
                else:
                    self.fields.pop(name, None)
                    cleared.append(name)
                continue
            try:
                parsed = PatientFields.model_validate({name: value})
            except ValidationError as exc:
                rejected[name] = exc.errors()[0]["msg"].removeprefix("Value error, ")
                continue
            self.fields[name] = getattr(parsed, name)
            accepted[name] = spoken_value(name, self.fields[name])

        result: dict[str, Any] = {"accepted": accepted, "rejected": rejected}
        if cleared:
            result["cleared"] = cleared

        if self.phase is Phase.SAVED:
            result["note"] = "The record is already saved. Ask before saving these changes as an update."
        elif accepted or cleared:
            self._after_change(result)

        result["still_missing_required"] = self.missing_required()
        result["phase"] = self.phase.value
        self._notify()
        return result

    def _after_change(self, result: dict[str, Any]) -> None:
        if self.phase is Phase.AWAITING_CONFIRMATION:
            # A correction during readback: confirm the changed fields, not the whole record again.
            result["note"] = "Read back only the corrected fields, then ask if everything is now correct."
            self._asked_to_confirm_at = self.clock()
        elif not self.missing_required():
            self.phase = Phase.READY_FOR_READBACK
        else:
            self.phase = Phase.COLLECTING

        phone = self.fields.get("phone_number")
        if phone and phone not in self._checked_phones and self.update_target is None:
            self._checked_phones.add(phone)
            matches = self._safe_find_by_phone(phone)
            if matches:
                result["existing_records"] = [
                    {"patient_id": str(p.patient_id), "first_name": p.first_name, "last_name": p.last_name}
                    for p in matches
                ]
                result["note"] = (
                    "This phone number matches an existing patient. Ask whether they want to update "
                    "that record instead of creating a new one."
                )

    def _safe_find_by_phone(self, phone: str) -> list[PatientOut]:
        try:
            return self.patients.find_by_phone(phone)
        except PersistenceError:
            return []  # duplicate detection is a nicety; never block registration on it

    # ---- tool: use_existing_record -------------------------------------------------------

    def use_existing_record(self, patient_id: str) -> dict[str, Any]:
        try:
            existing = self.patients.get(uuid.UUID(patient_id))
        except (ValueError, PatientNotFound):
            return {"ok": False, "error": "no such record; continue registering as new"}
        except PersistenceError:
            return {"ok": False, "error": "records are unavailable right now; continue as new"}

        self.update_target = existing.patient_id
        # Round-trip through the input schema so stored values get their canonical Python types.
        on_file = existing.model_dump(include=set(ALL_FIELDS), exclude_none=True)
        stored = PatientFields.model_validate(on_file).model_dump(exclude_none=True)
        # Values given on this call win over what is on file.
        self.fields = {**stored, **self.fields}
        self.phase = Phase.READY_FOR_READBACK if not self.missing_required() else Phase.COLLECTING
        self._notify()
        return {
            "ok": True,
            "mode": "updating existing record",
            "on_file": {k: spoken_value(k, v) for k, v in self.fields.items()},
            "note": "Ask what they would like to change, then do the readback before saving.",
        }

    # ---- tool: get_readback --------------------------------------------------------------

    def get_readback(self) -> dict[str, Any]:
        missing = self.missing_required()
        if missing:
            return {"ready": False, "still_missing_required": missing}
        self.phase = Phase.AWAITING_CONFIRMATION
        self._asked_to_confirm_at = self.clock()
        self._notify()
        return {
            "ready": True,
            "mode": "update" if self.update_target else "new registration",
            "readback": [
                {"field": FIELD_LABELS[f], "say": spoken_value(f, self.fields[f])}
                for f in ALL_FIELDS
                if f in self.fields
            ],
            "note": "Read these back naturally in a few short groups, then ask if everything is correct.",
        }

    # ---- tool: save_patient --------------------------------------------------------------

    def save_patient(self, caller_confirmed: bool) -> dict[str, Any]:
        if self.phase is Phase.SAVED and self.saved_patient:
            return {"saved": True, "already_saved": True, "first_name": self.saved_patient.first_name}
        if missing := self.missing_required():
            return {"saved": False, "error": "missing required fields", "still_missing_required": missing}
        if self.phase is not Phase.AWAITING_CONFIRMATION:
            return {"saved": False, "error": "do the readback with get_readback before saving"}
        if not caller_confirmed:
            return {"saved": False, "error": "the caller has not confirmed the readback yet"}
        if self._asked_to_confirm_at is not None and self.clock() - self._asked_to_confirm_at < self.min_confirm_secs:
            log.warning("save refused: too soon after the readback", extra={"call_sid": self.call_sid})
            return {
                "saved": False,
                "error": "refused: the caller has not answered the readback yet",
                "note": "Finish the readback, ask if everything is correct, and wait for their answer.",
            }

        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                if self.update_target:
                    patient = self.patients.update(self.update_target, PatientUpdate(**self.fields))
                else:
                    patient = self.patients.create(PatientCreate(**self.fields))
                break
            except PersistenceError as exc:
                last_error = exc
                log.warning("save attempt %d failed", attempt, extra={"call_sid": self.call_sid})
            except (ValidationError, PatientNotFound) as exc:
                last_error = exc
                break
        else:
            patient = None

        if patient is None:
            log.error(
                "registration not saved",
                extra={"call_sid": self.call_sid, "payload": self.draft(), "error": str(last_error)},
            )
            return {
                "saved": False,
                "error": "our records system is not responding",
                "note": "Apologize, say nothing was lost on their side, and offer to try once more.",
            }

        self.saved_patient = patient
        self.phase = Phase.SAVED
        log.info(
            "registration saved",
            extra={
                "call_sid": self.call_sid,
                "patient_id": str(patient.patient_id),
                "mode": "update" if self.update_target else "create",
                "payload": patient.model_dump(mode="json"),
            },
        )
        self._notify()
        return {"saved": True, "patient_id": str(patient.patient_id), "first_name": patient.first_name}

    # ---- tool: start_over / set_language -------------------------------------------------

    def start_over(self) -> dict[str, Any]:
        self.fields.clear()
        self.update_target = None
        self.saved_patient = None
        self.phase = Phase.COLLECTING
        self._checked_phones.clear()
        self._notify()
        return {"ok": True, "note": "Everything was cleared. Start again from their name."}

    def set_language(self, language: str) -> dict[str, Any]:
        if language not in LANGUAGE_NAMES:
            return {"ok": False, "error": "supported languages are en and es"}
        self.language = language
        # Speaking a language on the call is a good default, but never override an explicit answer
        # such as "I prefer Portuguese".
        if self.fields.get("preferred_language") in (None, *LANGUAGE_NAMES.values()):
            self.fields["preferred_language"] = LANGUAGE_NAMES[language]
        self._notify()
        return {"ok": True, "language": language}

    def resume_from(self, draft: dict[str, Any]) -> dict[str, Any]:
        result = self.update_fields(draft)
        return {"resumed": True, "have": result["accepted"], "still_missing_required": result["still_missing_required"]}

    # ---- helpers -------------------------------------------------------------------------

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if f not in self.fields]

    def draft(self) -> dict[str, Any]:
        """JSON-safe copy of the captured fields (for drafts and logs)."""
        return {k: (v.strftime("%m/%d/%Y") if isinstance(v, date) else v) for k, v in self.fields.items()}

    def state_summary(self) -> str:
        """Compact state line injected into the LLM context each turn."""
        have = ", ".join(self.fields) or "nothing yet"
        missing = ", ".join(self.missing_required()) or "none"
        mode = "updating existing record" if self.update_target else "new registration"
        return f"phase={self.phase.value}; mode={mode}; captured: {have}; missing required: {missing}"

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self)


_MONTHS = (
    "January February March April May June July August September October November December"
).split()


def _digits(value: str) -> str:
    return " ".join(value)


def spoken_value(field_name: str, value: Any) -> str:
    """How the agent should say a stored value out loud (TTS-friendly, unambiguous)."""
    if field_name in ("first_name", "last_name"):
        # Commas, not hyphens: Kokoro reads "J-A-N-E" as one word that sounds like "Jane".
        letters = ", ".join(ch.upper() for ch in value if ch.isalpha())
        return f"{value}, spelled {letters}"
    if field_name == "date_of_birth":
        return f"{_MONTHS[value.month - 1]} {value.day}, {value.year}"
    if field_name in ("phone_number", "emergency_contact_phone"):
        return f"{_digits(value[:3])}, {_digits(value[3:6])}, {_digits(value[6:])}"
    if field_name == "zip_code":
        return _digits(value.replace("-", ""))
    if field_name == "state":
        return US_STATES.get(value, value)
    if field_name == "insurance_member_id":
        return _digits(value)
    if field_name == "email":
        return value.replace(".", " dot ").replace("@", " at ")
    return str(value)
