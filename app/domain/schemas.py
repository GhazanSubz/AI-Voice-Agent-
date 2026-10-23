"""Pydantic schemas: the single source of truth for patient validation.

The REST API and the voice agent both build these models, so a value the agent accepts on the
phone is exactly a value the API would accept, and vice versa.
"""

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    field_serializer,
    model_validator,
)

from app.domain import validators as v


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"
    DECLINE = "Decline to Answer"


def _before(*funcs):
    """Chain plain-text cleanup with a field-specific normalizer, skipping non-strings."""

    def run(value: Any) -> Any:
        value = v.clean_text(value)
        if isinstance(value, str):
            for func in funcs:
                value = func(value)
        return value

    return BeforeValidator(run)


def _blank_to_none(value: Any) -> Any:
    return None if isinstance(value, str) and not value.strip() else value


Name = Annotated[str, _before(), AfterValidator(v.validate_name)]
FullName = Annotated[str, _before(), AfterValidator(lambda s: v.validate_name(s, max_len=100))]
DateOfBirth = Annotated[date, BeforeValidator(v.parse_date_of_birth)]
SexField = Annotated[Sex, _before(v.normalize_sex)]
UsPhone = Annotated[str, _before(v.normalize_us_phone)]
Email = Annotated[EmailStr, _before(str.lower)]
AddressLine = Annotated[str, _before(), AfterValidator(lambda s: v.require_text(s, "address", 200))]
City = Annotated[str, _before(), AfterValidator(lambda s: v.require_text(s, "city", 100))]
State = Annotated[str, _before(v.normalize_state)]
ZipCode = Annotated[str, _before(v.normalize_zip)]
InsuranceProvider = Annotated[
    str, _before(), AfterValidator(lambda s: v.require_text(s, "insurance provider", 100))
]
MemberId = Annotated[str, _before(v.normalize_member_id)]
Language = Annotated[str, _before(), AfterValidator(lambda s: v.require_text(s, "language", 50))]


def Optional(field_type):  # noqa: N802 - reads like a type in the models below
    """Optional field where an empty string means "not provided"."""
    return Annotated[field_type | None, BeforeValidator(_blank_to_none)]


REQUIRED_FIELDS = (
    "first_name", "last_name", "date_of_birth", "sex", "phone_number",
    "address_line_1", "city", "state", "zip_code",
)


class PatientFields(BaseModel):
    """Every client-writable field, all optional. Used for partial updates."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    first_name: Name | None = None
    last_name: Name | None = None
    date_of_birth: DateOfBirth | None = None
    sex: SexField | None = None
    phone_number: UsPhone | None = None
    email: Optional(Email) = None
    address_line_1: AddressLine | None = None
    address_line_2: Optional(AddressLine) = None
    city: City | None = None
    state: State | None = None
    zip_code: ZipCode | None = None
    insurance_provider: Optional(InsuranceProvider) = None
    insurance_member_id: Optional(MemberId) = None
    preferred_language: Optional(Language) = None
    emergency_contact_name: Optional(FullName) = None
    emergency_contact_phone: Optional(UsPhone) = None


class PatientUpdate(PatientFields):
    @model_validator(mode="after")
    def _required_fields_not_nulled(self):
        nulled = [f for f in REQUIRED_FIELDS if f in self.model_fields_set and getattr(self, f) is None]
        if nulled:
            raise ValueError(f"required fields cannot be cleared: {', '.join(nulled)}")
        return self

    def changes(self) -> dict[str, Any]:
        """Only the fields the client actually sent, so omitted fields are left alone."""
        return self.model_dump(exclude_unset=True)


class PatientCreate(PatientFields):
    first_name: Name
    last_name: Name
    date_of_birth: DateOfBirth
    sex: SexField
    phone_number: UsPhone
    address_line_1: AddressLine
    city: City
    state: State
    zip_code: ZipCode
    preferred_language: Optional(Language) = Field(default="English")

    @model_validator(mode="after")
    def _default_language(self):
        if self.preferred_language is None:
            self.preferred_language = "English"
        return self


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: uuid.UUID
    first_name: str
    last_name: str
    date_of_birth: date
    sex: str
    phone_number: str
    email: str | None
    address_line_1: str
    address_line_2: str | None
    city: str
    state: str
    zip_code: str
    insurance_provider: str | None
    insurance_member_id: str | None
    preferred_language: str
    emergency_contact_name: str | None
    emergency_contact_phone: str | None
    created_at: datetime
    updated_at: datetime

    @field_serializer("date_of_birth")
    def _dob_us_format(self, value: date) -> str:
        return value.strftime("%m/%d/%Y")

    @field_serializer("created_at", "updated_at")
    def _utc_iso(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
