"""Fictional demo patients (555-01xx numbers are reserved for fiction)."""

import logging

from app.domain.schemas import PatientCreate
from app.services.patients import PatientService

log = logging.getLogger(__name__)

DEMO_PATIENTS = [
    {
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "04/12/1988",
        "sex": "Female",
        "phone_number": "4155550132",
        "email": "jane.doe@example.com",
        "address_line_1": "100 Market Street",
        "address_line_2": "Apt 4B",
        "city": "San Francisco",
        "state": "CA",
        "zip_code": "94105",
        "insurance_provider": "Blue Shield of California",
        "insurance_member_id": "XKB123456789",
        "emergency_contact_name": "John Doe",
        "emergency_contact_phone": "4155550199",
    },
    {
        "first_name": "Carlos",
        "last_name": "Ramirez",
        "date_of_birth": "11/30/1975",
        "sex": "Male",
        "phone_number": "5125550147",
        "address_line_1": "2200 South Congress Avenue",
        "city": "Austin",
        "state": "TX",
        "zip_code": "78704-1203",
        "preferred_language": "Spanish",
    },
]


def seed_demo_patients(service: PatientService) -> None:
    if service.count_active() > 0:
        return
    for record in DEMO_PATIENTS:
        service.create(PatientCreate(**record))
    log.info("seeded %d demo patients", len(DEMO_PATIENTS))
