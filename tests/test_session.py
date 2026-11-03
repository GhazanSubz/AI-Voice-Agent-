"""RegistrationSession: the state machine the voice agent's tools drive."""

import pytest

from app.db.session import init_db, make_engine, make_session_factory
from app.domain.schemas import PatientCreate
from app.services.patients import PatientService, PersistenceError
from app.voice.session import Phase, RegistrationSession

REQUIRED = {
    "first_name": "Maria",
    "last_name": "Davies",
    "date_of_birth": "07/04/1990",
    "sex": "female",
    "phone_number": "312 555 0110",
    "address_line_1": "42 Lake Shore Drive",
    "city": "Chicago",
    "state": "Illinois",
    "zip_code": "60611",
}


@pytest.fixture
def patients(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 's.db'}")
    init_db(engine)
    return PatientService(make_session_factory(engine))


@pytest.fixture
def session(patients):
    return RegistrationSession(call_sid="CA1", patients=patients, caller_number="+13125550110", min_confirm_secs=0)


def confirm_and_save(session):
    session.get_readback()
    return session.save_patient(caller_confirmed=True)


def test_out_of_order_fields_are_accepted_in_any_batch(session):
    first = session.update_fields({"zip_code": "60611", "first_name": "Maria"})
    assert set(first["accepted"]) == {"zip_code", "first_name"}
    assert "last_name" in first["still_missing_required"]
    assert first["phase"] == Phase.COLLECTING.value


def test_invalid_field_is_rejected_with_reason_and_others_kept(session):
    result = session.update_fields({"date_of_birth": "01/01/2999", "phone_number": "555", "city": "Chicago"})

    assert "future" in result["rejected"]["date_of_birth"]
    assert "10 digits" in result["rejected"]["phone_number"]
    assert "city" in result["accepted"]


def test_correction_overwrites_value(session):
    session.update_fields({"last_name": "Davies"})
    result = session.update_fields({"last_name": "Davis"})
    assert result["accepted"]["last_name"] == "Davis, spelled D, A, V, I, S"
    assert session.fields["last_name"] == "Davis"


def test_save_requires_readback_and_confirmation(session, patients):
    session.update_fields(REQUIRED)
    assert session.phase is Phase.READY_FOR_READBACK

    assert session.save_patient(caller_confirmed=True)["saved"] is False  # no readback yet
    readback = session.get_readback()
    assert readback["ready"] and session.phase is Phase.AWAITING_CONFIRMATION
    assert session.save_patient(caller_confirmed=False)["saved"] is False

    result = session.save_patient(caller_confirmed=True)

    assert result["saved"] is True
    assert result["first_name"] == "Maria"
    assert patients.count_active() == 1


def test_readback_is_speakable(session):
    session.update_fields(REQUIRED)
    said = {item["field"]: item["say"] for item in session.get_readback()["readback"]}
    assert said["phone number"] == "3 1 2, 5 5 5, 0 1 1 0"
    assert said["date of birth"] == "July 4, 1990"
    assert said["state"] == "Illinois"


def test_readback_refused_while_fields_missing(session):
    session.update_fields({"first_name": "Maria"})
    assert session.get_readback()["ready"] is False


def test_correction_during_readback_keeps_confirmation_phase(session):
    session.update_fields(REQUIRED)
    session.get_readback()
    result = session.update_fields({"last_name": "Davis"})
    assert session.phase is Phase.AWAITING_CONFIRMATION
    assert "corrected" in result["note"]


def test_required_field_cannot_be_blanked_but_optional_can(session):
    session.update_fields({"email": "maria@example.com"})
    result = session.update_fields({"email": "", "first_name": ""})
    assert result["cleared"] == ["email"]
    assert "first_name" in result["rejected"]


def test_start_over_clears_everything(session):
    session.update_fields(REQUIRED)
    session.start_over()
    assert session.fields == {}
    assert session.phase is Phase.COLLECTING


def test_duplicate_phone_is_flagged_and_update_path_works(session, patients):
    existing = patients.create(PatientCreate(**{**REQUIRED, "last_name": "Davis"}))

    result = session.update_fields({"phone_number": "(312) 555-0110"})

    assert result["existing_records"][0]["patient_id"] == str(existing.patient_id)
    use = session.use_existing_record(str(existing.patient_id))
    assert use["ok"] and session.phase is Phase.READY_FOR_READBACK

    session.update_fields({"city": "Evanston"})
    saved = confirm_and_save(session)

    assert saved["saved"] and saved["patient_id"] == str(existing.patient_id)
    assert patients.get(existing.patient_id).city == "Evanston"
    assert patients.count_active() == 1


def test_database_failure_is_reported_not_raised(session, monkeypatch):
    session.update_fields(REQUIRED)
    calls = []

    def boom(_data):
        calls.append(1)
        raise PersistenceError("disk I/O error")

    monkeypatch.setattr(session.patients, "create", boom)

    result = confirm_and_save(session)

    assert result["saved"] is False
    assert len(calls) == 2  # retried once
    assert session.phase is Phase.AWAITING_CONFIRMATION  # caller can say "try again"


def test_save_is_idempotent(session, patients):
    session.update_fields(REQUIRED)
    confirm_and_save(session)
    again = session.save_patient(caller_confirmed=True)
    assert again["already_saved"] is True
    assert patients.count_active() == 1


def test_spanish_sets_preferred_language_unless_caller_chose_another(session):
    session.set_language("es")
    assert session.fields["preferred_language"] == "Spanish"

    session.update_fields({"preferred_language": "Portuguese"})
    session.set_language("en")
    assert session.fields["preferred_language"] == "Portuguese"


def test_unknown_field_rejected(session):
    assert "ssn" in session.update_fields({"ssn": "123"})["rejected"]


def test_draft_round_trip(session, patients):
    session.update_fields({"first_name": "Maria", "date_of_birth": "07/04/1990"})
    draft = session.draft()

    resumed = RegistrationSession(call_sid="CA2", patients=patients)
    result = resumed.resume_from(draft)

    assert set(result["have"]) == {"first_name", "date_of_birth"}
    assert resumed.fields == session.fields


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_save_seconds_after_readback_is_refused_as_not_the_callers_answer(patients):
    clock = FakeClock()
    session = RegistrationSession(call_sid="CA9", patients=patients, clock=clock)  # default 3 s
    session.update_fields(REQUIRED)
    session.get_readback()

    clock.now += 1.0  # the LLM "confirms" one second after starting the readback
    refused = session.save_patient(caller_confirmed=True)
    clock.now += 8.0  # a real caller: readback spoken, then "yes"
    saved = session.save_patient(caller_confirmed=True)

    assert refused["saved"] is False and "not answered" in refused["error"]
    assert saved["saved"] is True


def test_correction_during_readback_restarts_the_confirmation_wait(patients):
    clock = FakeClock()
    session = RegistrationSession(call_sid="CA9", patients=patients, clock=clock)
    session.update_fields(REQUIRED)
    session.get_readback()
    clock.now += 10
    session.update_fields({"last_name": "Davis"})  # agent reads back the fix and asks again

    assert session.save_patient(caller_confirmed=True)["saved"] is False
    clock.now += 4
    assert session.save_patient(caller_confirmed=True)["saved"] is True


# ---- sex decided from the caller's words (live call: "male" was transcribed "Mary", saved Female)

@pytest.mark.parametrize(
    "heard, expected",
    [
        ("male", ("Male", False)), ("I'm a man", ("Male", False)), ("Female.", ("Female", False)),
        ("mail", ("Male", True)), ("Meal. Mail. M a l e.", ("Male", True)), ("fe mail", ("Female", True)),
        ("other", ("Other", False)), ("I'd prefer not to say", ("Decline to Answer", False)),
        ("Mary", (None, False)), ("It is me.", (None, False)), ("male, no, female", (None, False)), ("", (None, False)),
    ],
)
def test_sex_from_speech(heard, expected):
    from app.voice.session import sex_from_speech

    assert sex_from_speech(heard) == expected


def test_unclear_sex_is_rejected_not_guessed(session):
    result = session.record_answer({"sex": "Female"}, sex_heard="Mary")

    assert "sex" in result["rejected"]
    assert "sex" not in session.fields
    assert "sex" in result["still_missing_required"]
    assert "male or female" in result["note"]


def test_the_callers_words_win_over_the_llms_guess(session):
    result = session.record_answer({"sex": "Female", "first_name": "Ghazan"}, sex_heard="male")

    assert session.fields["sex"] == "Male"
    assert result["accepted"]["sex"] == "Male"
    assert "first_name" in result["accepted"]


def test_misheard_sex_is_accepted_with_a_confirmation_note(session):
    result = session.record_answer({"sex": "Male"}, sex_heard="mail")
    assert session.fields["sex"] == "Male"
    assert "is that right" in result["note"]


def test_sex_without_the_callers_words_is_rejected(session):
    assert "sex" in session.record_answer({"sex": "Male"}, sex_heard=None)["rejected"]


def test_other_fields_need_no_heard_text(session):
    result = session.record_answer({"first_name": "Maria"}, sex_heard=None)
    assert result["accepted"] and not result["rejected"]


# ---- state: ZIP first, no abbreviations (live call: "Delaware" heard as "dialogue", "B E")

def test_zip_proposes_the_missing_state(session):
    result = session.update_fields({"address_line_1": "57 Seventh Avenue", "city": "Dover", "zip_code": "19901"})

    assert result["state_from_zip"] == "Delaware"
    assert "That's in Delaware" in result["note"]
    assert "state" not in session.fields  # only proposed; the caller has to confirm it


def test_a_given_state_is_saved_as_is_even_if_it_differs_from_the_zip(session):
    result = session.update_fields({"state": "New Jersey", "zip_code": "19901"})

    assert session.fields["state"] == "NJ"
    assert "state_from_zip" not in result and "note" not in result


def test_misheard_state_is_rejected_with_a_suggestion(session):
    result = session.update_fields({"state": "Delivery"})

    assert "state" in result["rejected"]
    assert result["did_you_mean"] == {"state": ["Delaware"]}


def test_notes_add_up_instead_of_replacing_each_other(session, patients):
    patients.create(PatientCreate(**REQUIRED))
    result = session.update_fields({"phone_number": REQUIRED["phone_number"], "zip_code": "19901"})

    assert "existing patient" in result["note"] and "Delaware" in result["note"]
