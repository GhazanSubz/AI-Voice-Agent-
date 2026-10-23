"""CallToolbox and CallService: tool dispatch, resume-after-drop, transcripts."""

import asyncio

import pytest

from app.db.session import init_db, make_engine, make_session_factory
from app.services.calls import CallService
from app.services.patients import PatientService
from app.voice.session import RegistrationSession
from app.voice.tools import TOOL_SCHEMAS, CallToolbox

REQUIRED = {
    "first_name": "Maria", "last_name": "Davis", "date_of_birth": "07/04/1990", "sex": "Female",
    "phone_number": "3125550110", "address_line_1": "42 Lake Shore Drive", "city": "Chicago",
    "state": "IL", "zip_code": "60611",
}


@pytest.fixture
def services(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 't.db'}")
    init_db(engine)
    factory = make_session_factory(engine)
    return PatientService(factory), CallService(factory)


def run(coro):
    return asyncio.run(coro)


def test_every_schema_has_a_handler(services):
    patients, _ = services
    box = CallToolbox(session=RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0))
    assert {t["name"] for t in TOOL_SCHEMAS} == set(box.handlers)


def test_full_registration_through_tools(services):
    patients, calls = services
    ended = []

    async def on_end():
        ended.append(True)

    box = CallToolbox(session=RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0), on_end_call=on_end)

    async def flow():
        await box.call("update_fields", {"fields": REQUIRED})
        await box.call("get_readback", {})
        saved = await box.call("save_patient", {"caller_confirmed": True})
        await box.call("end_call", {"reason": "completed"})
        return saved

    saved = run(flow())
    assert saved["saved"] and ended == [True]
    assert patients.count_active() == 1


def test_tool_exception_becomes_recoverable_result(services, monkeypatch):
    patients, _ = services
    box = CallToolbox(session=RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0))
    monkeypatch.setattr(box.session, "get_readback", lambda: 1 / 0)
    result = run(box.call("get_readback", {}))
    assert "error" in result


def test_bad_arguments_do_not_crash(services):
    patients, _ = services
    box = CallToolbox(session=RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0))
    assert "error" in run(box.call("update_fields", {"fields": "Maria"}))
    assert "error" in run(box.call("no_such_tool", {}))


def test_language_switch_triggers_pipeline_callback(services):
    patients, _ = services
    switched = []

    async def on_lang(lang):
        switched.append(lang)

    box = CallToolbox(session=RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0), on_language_change=on_lang)
    run(box.call("set_language", {"language": "es"}))
    assert switched == ["es"]


def test_dropped_call_can_be_resumed_from_same_number(services):
    patients, calls = services
    caller = "3125550110"

    first = RegistrationSession(call_sid="CA1", patients=patients, caller_number=caller)
    calls.start("CA1", caller)
    first.update_fields({"first_name": "Maria", "last_name": "Davis"})
    calls.finish("CA1", patient_id=None, draft_fields=first.draft(), transcript=[{"speaker": "caller", "text": "hi"}])

    calls.start("CA2", caller)
    draft = calls.find_resumable_draft(caller, exclude_sid="CA2")
    assert draft is not None and draft.call_sid == "CA1"

    second = RegistrationSession(call_sid="CA2", patients=patients, caller_number=caller)
    box = CallToolbox(session=second, calls=calls, resume_draft=dict(draft.draft_fields), resume_from_sid="CA1")
    result = run(box.call("resume_previous_call", {"accept": True}))

    assert set(result["have"]) == {"first_name", "last_name"}
    assert calls.find_resumable_draft(caller, exclude_sid="CA3") is None  # marked resumed


def test_completed_call_is_linked_to_patient_with_transcript(services):
    patients, calls = services
    session = RegistrationSession(call_sid="CA1", patients=patients, min_confirm_secs=0)
    calls.start("CA1", None)
    session.update_fields(REQUIRED)
    session.get_readback()
    session.save_patient(caller_confirmed=True)

    calls.finish(
        "CA1",
        patient_id=session.saved_patient.patient_id,
        draft_fields=session.draft(),
        transcript=[{"speaker": "agent", "text": "You're all set, Maria."}],
    )

    history = calls.list_for_patient(session.saved_patient.patient_id)
    assert history[0]["status"] == "completed"
    assert history[0]["transcript"][0]["text"] == "You're all set, Maria."
    assert calls.find_resumable_draft(None, exclude_sid="x") is None
