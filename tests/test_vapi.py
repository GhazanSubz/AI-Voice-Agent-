"""Vapi integration: webhook auth, per-call assistant, tool calls, end-of-call report, custom TTS."""

import json
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SECRET = "test-vapi-secret"
BASE = "https://agent.example.ngrok-free.app"
AUTH = {"X-Vapi-Secret": SECRET}

REQUIRED = {
    "first_name": "Maria", "last_name": "Davis", "date_of_birth": "07/04/1990", "sex": "Female",
    "phone_number": "3125550110", "address_line_1": "42 Lake Shore Drive", "city": "Chicago",
    "state": "IL", "zip_code": "60611",
}


@pytest.fixture
def client(db_url):
    settings = Settings(
        database_url=db_url, seed_demo_data=False, log_level="DEBUG",
        public_base_url=BASE, vapi_secret=SECRET, confirm_min_secs=0,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def event(kind, call_id="call-1", caller="+13125550110", **extra):
    return {"message": {"type": kind, "call": {"id": call_id, "customer": {"number": caller}}, **extra}}


def tool_call(name, args, call_id="call-1", tc_id="tc-1"):
    body = event("tool-calls", call_id, toolCallList=[
        {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}
    ])
    return body


def call_tool(client, name, args, **kw):
    response = client.post("/vapi/webhook", json=tool_call(name, args, **kw), headers=AUTH)
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    return json.loads(result["result"])


class TestAuth:
    def test_missing_secret_is_401(self, client):
        assert client.post("/vapi/webhook", json=event("status-update")).status_code == 401

    def test_wrong_secret_is_401(self, client):
        response = client.post("/vapi/webhook", json=event("status-update"), headers={"X-Vapi-Secret": "nope"})
        assert response.status_code == 401

    def test_bearer_credential_is_accepted(self, client):
        headers = {"Authorization": f"Bearer {SECRET}"}
        assert client.post("/vapi/webhook", json=event("status-update"), headers=headers).status_code == 200

    def test_tts_requires_secret(self, client):
        assert client.post("/vapi/tts", json={"message": {"type": "voice-request"}}).status_code == 401


class TestAssistantRequest:
    def test_returns_groq_assistant_with_our_tools_and_kokoro_voice(self, client):
        response = client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH)

        assistant = response.json()["assistant"]
        assert assistant["model"]["provider"] == "groq"
        names = [t.get("function", {}).get("name", t["type"]) for t in assistant["model"]["tools"]]
        assert "update_fields" in names and "save_patient" in names
        assert "end_call" not in names and "endCall" in names
        assert assistant["voice"]["provider"] == "custom-voice"
        assert assistant["voice"]["server"]["url"] == f"{BASE}/vapi/tts"
        # Inline assistants don't get dashboard credentials, so each URL carries the secret.
        assert assistant["voice"]["server"]["headers"] == AUTH
        assert assistant["server"]["headers"] == AUTH
        system = assistant["model"]["messages"][0]["content"]
        assert "endCall" in system and "ending in 0 1 1 0" in system

    def test_offers_resume_after_a_dropped_call(self, client):
        client.post("/vapi/webhook", json=event("assistant-request", "call-1"), headers=AUTH)
        call_tool(client, "update_fields", {"fields": {"first_name": "Maria"}}, call_id="call-1")
        client.post("/vapi/webhook", json=event("end-of-call-report", "call-1", endedReason="customer-ended-call"), headers=AUTH)

        second = client.post("/vapi/webhook", json=event("assistant-request", "call-2"), headers=AUTH).json()

        assert "pick up where we left off" in second["assistant"]["firstMessage"]
        result = call_tool(client, "resume_previous_call", {"accept": True}, call_id="call-2")
        assert result["have"] == {"first_name": "Maria, spelled M, A, R, I, A"}


def test_full_registration_over_webhook_is_saved_with_transcript(client):
    client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH)

    assert call_tool(client, "update_fields", {"fields": {"date_of_birth": "01/01/2999"}})["rejected"]
    assert call_tool(client, "update_fields", {"fields": REQUIRED, "sex_heard": "female"})["still_missing_required"] == []
    assert call_tool(client, "get_readback", {})["ready"] is True
    saved = call_tool(client, "save_patient", {"caller_confirmed": True})
    assert saved["saved"] is True

    report = event(
        "end-of-call-report",
        endedReason="assistant-ended-call",
        artifact={"messages": [
            {"role": "system", "message": "prompt"},
            {"role": "bot", "message": "You're all set, Maria."},
            {"role": "user", "message": "Thanks!"},
        ]},
    )
    assert client.post("/vapi/webhook", json=report, headers=AUTH).status_code == 200

    patients = client.get("/patients", params={"last_name": "Davis"}).json()["data"]
    assert [p["patient_id"] for p in patients] == [saved["patient_id"]]
    history = client.app.state.call_service.list_for_patient(uuid.UUID(patients[0]["patient_id"]))
    assert history[0]["status"] == "completed"
    assert [t["speaker"] for t in history[0]["transcript"]] == ["agent", "caller"]


def test_tool_call_for_unknown_call_rebuilds_session_from_draft(client):
    client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH)
    call_tool(client, "update_fields", {"fields": {"first_name": "Maria"}})
    client.app.state.call_registry.pop("call-1")  # simulate a server restart mid-call

    result = call_tool(client, "update_fields", {"fields": {"last_name": "Davis"}})

    assert "first_name" not in result["still_missing_required"]


def test_unknown_event_types_are_acknowledged(client):
    assert client.post("/vapi/webhook", json=event("speech-update"), headers=AUTH).json() == {}


class FakeKokoro:
    """Yields two 0.1 s chunks of 24 kHz audio, like Kokoro does per sentence."""

    def __init__(self):
        self.calls = []

    async def create_stream(self, text, voice, lang):
        self.calls.append((text, voice, lang))
        for _ in range(2):
            yield np.zeros(2400, dtype=np.float32), 24000


class FakeEngine:
    def __init__(self):
        self.kokoro = FakeKokoro()

    def get(self):
        return self.kokoro

    def voice_for(self, language):
        return "ef_dora" if language == "es" else "af_heart"


def test_tts_streams_pcm_at_requested_rate_in_call_language(client):
    engine = FakeEngine()
    client.app.state.tts_engine = engine
    client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH)
    call_tool(client, "set_language", {"language": "es"})

    body = {"message": {"type": "voice-request", "text": "Hola", "sampleRate": 8000, "call": {"id": "call-1"}}}
    response = client.post("/vapi/tts", json=body, headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    # 0.2 s of audio at 8 kHz, 2 bytes per sample (resampler edge effects allow a little slack).
    assert abs(len(response.content) - 3200) <= 64
    assert engine.kokoro.calls == [("Hola", "ef_dora", "es")]


def test_tts_rejects_unsupported_sample_rate(client):
    body = {"message": {"type": "voice-request", "text": "Hi", "sampleRate": 12345}}
    assert client.post("/vapi/tts", json=body, headers=AUTH).status_code == 400


def test_long_replies_are_split_into_sentences_for_faster_first_audio():
    from app.vapi.tts import split_sentences

    assert split_sentences("Hi, I'm Sam. How are you?  Great!") == ["Hi, I'm Sam.", "How are you?", "Great!"]
    assert split_sentences("Your ZIP is 6 0 6 1 1") == ["Your ZIP is 6 0 6 1 1"]


def test_stage_directions_are_not_spoken():
    from app.vapi.tts import clean_for_speech

    assert clean_for_speech("Is everything now correct?(Waiting for your response.)") == "Is everything now correct?"
    assert clean_for_speech("Could you tell me your date of birth?We have to wait for user input.") == (
        "Could you tell me your date of birth?"
    )
    assert clean_for_speech("Your sex\u2014Male or Female? [Awaiting user response]") == "Your sex, Male or Female?"
    assert clean_for_speech("Spelled D\u2011A\u2011V\u2011I\u2011S.") == "Spelled D-A-V-I-S."


def test_repeated_sentences_are_spoken_once():
    from app.vapi.tts import clean_for_speech

    assert clean_for_speech("What is your sex? What is your sex? Male or female?") == "What is your sex? Male or female?"


def test_assistant_turn_taking_fillers_and_goodbye(client):
    assistant = client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH).json()["assistant"]

    # Pauses mid-sentence ("Ladan ... and my last name is") shouldn't end the caller's turn,
    # using pause-based endpointing only (LiveKit swallowed the caller's turn on a live call).
    plan = assistant["startSpeakingPlan"]
    assert "smartEndpointingPlan" not in plan
    assert plan["transcriptionEndpointingPlan"]["onPunctuationSeconds"] >= 0.8
    # A one-word answer given while Sam is still asking must interrupt Sam.
    assert assistant["stopSpeakingPlan"]["numWords"] == 0
    # Our tools are silent: no "Just a sec" filler.
    functions = [t for t in assistant["model"]["tools"] if t["type"] == "function"]
    assert functions and all(t["messages"] == [{"type": "request-start", "content": ""}] for t in functions)
    # The goodbye is said once, by the hang-up tool.
    end_call = next(t for t in assistant["model"]["tools"] if t["type"] == "endCall")
    assert "Goodbye" in end_call["messages"][0]["content"]
    assert "endCallMessage" not in assistant
    # The LLM is told to pass the caller's words for sex.
    update = next(t for t in functions if t["function"]["name"] == "update_fields")
    assert "sex_heard" in update["function"]["parameters"]["properties"]


def test_missing_model_file_is_downloaded_once(tmp_path):
    # A fresh deploy has no models/ folder (git-ignored); the engine must fetch the files itself.
    from app.vapi.tts import ensure_file

    source = tmp_path / "release.onnx"
    source.write_bytes(b"model-bytes")
    target = tmp_path / "volume" / "models" / "kokoro.onnx"

    ensure_file(str(target), source.as_uri())
    assert target.read_bytes() == b"model-bytes"

    source.write_bytes(b"changed")
    ensure_file(str(target), source.as_uri())  # already there: not downloaded again
    assert target.read_bytes() == b"model-bytes"


def test_failed_download_leaves_no_partial_model(tmp_path):
    from app.vapi.tts import ensure_file

    target = tmp_path / "models" / "kokoro.onnx"
    with pytest.raises(OSError):
        ensure_file(str(target), (tmp_path / "missing.onnx").as_uri())
    assert not target.exists()


def test_note_asides_are_not_spoken():
    from app.vapi.tts import clean_for_speech

    assert clean_for_speech("What is your date of birth? Note:") == "What is your date of birth?"
    assert clean_for_speech("Could you tell me your street address? Note, the caller") == (
        "Could you tell me your street address?"
    )
    assert clean_for_speech("Please note that we need a mailing address.") == "Please note that we need a mailing address."


# ---- one question per turn (live: "Did I get that right? What is your last name? What is ...")

class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_gate_cuts_text_after_the_first_question():
    from app.vapi.speech_gate import SpeechGate

    gate = SpeechGate(clock=Clock())
    assert gate.filter("Got it, Ghazan. Did I get that right? What is your last name?") == (
        "Got it, Ghazan. Did I get that right?"
    )


def test_gate_holds_follow_up_speech_then_lets_it_through():
    from app.vapi.speech_gate import SpeechGate

    clock = Clock()
    gate = SpeechGate(hold_secs=3.0, clock=clock)
    assert gate.filter("What is your date of birth?") == "What is your date of birth?"
    clock.now += 1.0
    assert gate.filter("Could you tell me your date of birth?") == ""  # stacked question
    clock.now += 3.0
    assert gate.filter("Got it, January first. What sex should I put down?")  # real next turn


def test_gate_never_holds_statements():
    from app.vapi.speech_gate import SpeechGate

    gate = SpeechGate(clock=Clock())
    assert gate.filter("Maria Davis, born July fourth.") == "Maria Davis, born July fourth."
    assert gate.filter("Phone three one two, five five five.") == "Phone three one two, five five five."


def test_caller_interrupting_releases_the_hold():
    from app.vapi.speech_gate import SpeechGate

    gate = SpeechGate(clock=Clock())
    gate.filter("What sex should I put down?")
    gate.caller_spoke()
    assert gate.filter("Got it, male. What is your phone number?") == "Got it, male. What is your phone number?"


def tts(client, text, call_id="call-1"):
    body = {"message": {"type": "voice-request", "text": text, "sampleRate": 8000, "call": {"id": call_id}}}
    return client.post("/vapi/tts", json=body, headers=AUTH)


def test_stacked_question_gets_empty_audio_and_barge_in_resets(client):
    client.app.state.tts_engine = FakeEngine()

    first = tts(client, "What is your date of birth?")
    stacked = tts(client, "Could you tell me your date of birth?")
    assert first.status_code == 200 and len(first.content) > 0
    assert stacked.status_code == 200 and stacked.content == b""

    interrupted = event("user-interrupted")
    assert client.post("/vapi/webhook", json=interrupted, headers=AUTH).status_code == 200
    assert len(tts(client, "Got it, January first. What sex should I put down?").content) > 0


def test_speech_update_from_the_caller_also_resets(client):
    client.app.state.tts_engine = FakeEngine()
    tts(client, "What is your date of birth?")
    started = event("speech-update", status="started", role="user")
    client.post("/vapi/webhook", json=started, headers=AUTH)
    assert len(tts(client, "Got it. What sex should I put down?").content) > 0


def test_assistant_subscribes_to_interruption_events(client):
    assistant = client.post("/vapi/webhook", json=event("assistant-request"), headers=AUTH).json()["assistant"]
    assert {"user-interrupted", "speech-update"} <= set(assistant["serverMessages"])


def test_leaked_reasoning_is_never_spoken():
    # Spoken aloud on a live call, word for word.
    from app.vapi.tts import clean_for_speech

    leaked = (
        "The assistant responded with garbled text. Must correct, we should acknowledge the phone "
        "number. Then ask for address line 1. The last response is invalid."
    )
    assert clean_for_speech(leaked) == ""
    assert clean_for_speech("Need to correct, we need to respond with proper text. No extra characters, "
                            "The phone number is accepted.") == ""
    assert clean_for_speech("Got. We need to continue.") == ""


def test_real_speech_survives_the_reasoning_filter():
    from app.vapi.tts import clean_for_speech

    for spoken in (
        "Thanks, Ghazan. What is your date of birth?",
        "We need a U.S. mailing address to register you. Could you give me one?",
        "Got it, Kevin, spelled K, E, V, I, N. Did I get that right?",
        "Could you say the last four digits?",
        "Five nine six, got it. And the next three digits?",
    ):
        assert clean_for_speech(spoken) == spoken


def test_us_abbreviation_is_not_split_into_two_sentences():
    from app.vapi.tts import split_sentences

    assert split_sentences("We need a U.S. address. Do you have one?") == [
        "We need a U.S. address.", "Do you have one?"
    ]
