"""REST endpoints for patients. Thin layer: parse, delegate to PatientService, wrap in the envelope."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.envelope import Envelope, FieldError, error_response
from app.domain.schemas import DateOfBirth, PatientCreate, PatientOut, PatientUpdate, UsPhone
from app.services.patients import PatientService

router = APIRouter(prefix="/patients", tags=["patients"])


def get_service(request: Request) -> PatientService:
    return request.app.state.patient_service


Service = Annotated[PatientService, Depends(get_service)]


@router.get("", response_model=Envelope[list[PatientOut]])
def list_patients(
    service: Service,
    last_name: Annotated[str | None, Query(max_length=50)] = None,
    date_of_birth: Annotated[DateOfBirth | None, Query(description="MM/DD/YYYY or YYYY-MM-DD")] = None,
    phone_number: Annotated[UsPhone | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    patients = service.search(
        last_name=last_name.strip() if last_name else None,
        date_of_birth=date_of_birth,
        phone_number=phone_number,
        limit=limit,
        offset=offset,
    )
    return Envelope(data=patients)


@router.get("/{patient_id}", response_model=Envelope[PatientOut])
def get_patient(patient_id: uuid.UUID, service: Service):
    return Envelope(data=service.get(patient_id))


@router.post("", status_code=201, response_model=Envelope[PatientOut])
def create_patient(body: PatientCreate, service: Service):
    return Envelope(data=service.create(body))


@router.put("/{patient_id}", response_model=Envelope[PatientOut])
def update_patient(patient_id: uuid.UUID, body: PatientUpdate, service: Service):
    if not body.changes():
        return error_response(
            422,
            "validation_error",
            "Provide at least one field to update.",
            [FieldError(field=None, message="request body has no fields")],
        )
    return Envelope(data=service.update(patient_id, body))


@router.delete("/{patient_id}", response_model=Envelope[PatientOut])
def delete_patient(patient_id: uuid.UUID, service: Service):
    """Soft delete: sets deleted_at; the record is hidden from all reads but kept in the table."""
    return Envelope(data=service.soft_delete(patient_id))
