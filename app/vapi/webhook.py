"""Vapi server webhook: POST /vapi/webhook.

Vapi sends every server event for a call here. We handle three:
  assistant-request   a call is ringing: create the per-call session, return the assistant
  tool-calls          the LLM called our tools: run them against the session
  end-of-call-report  the call is over: store transcript + outcome, drop the session

Authentication: Vapi sends our shared secret either as X-Vapi-Secret (inline assistant URLs, see
assistant.auth_headers) or as "Authorization: Bearer <secret>" (the dashboard credential on the
phone number). Anything else gets a 401.
"""

import hmac
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.config import Settings
from app.services.calls import CallService
from app.services.patients import PatientService, PersistenceError
from app.vapi.assistant import build_assistant
from app.vapi.registry import CallRegistry
from app.voice.session import FIELD_LABELS, RegistrationSession
from app.voice.tools import CallToolbox

log = logging.getLogger(__name__)

router = APIRouter(prefix="/vapi", tags=["telephony"])


def require_vapi_secret(request: Request) -> None:
    secret = request.app.state.settings.vapi_secret
    if not secret:
        log.error("VAPI_SECRET is not set; refusing Vapi requests")
        raise HTTPException(status_code=503, detail="server not configured")
    given = request.headers.get("X-Vapi-Secret") or ""
    bearer = request.headers.get("Authorization", "")
    if bearer.startswith("Bearer "):
        given = given or bearer.removeprefix("Bearer ")
    if not hmac.compare_digest(given.encode(), secret.encode()):
        raise HTTPException(status_code=401, detail="invalid Vapi secret")


@router.post("/webhook", include_in_schema=False)
async def vapi_webhook(request: Request):
    require_vapi_secret(request)
    payload = await request.json()
    message = payload.get("message") or {}
    kind = message.get("type")
    state = request.app.state

    if kind == "assistant-request":
        return _assistant_request(message, state.settings, state.patient_service, state.call_service, state.call_registry)
    if kind == "tool-calls":
        return await _tool_calls(message, state.patient_service, state.call_service, state.call_registry)
    if kind == "end-of-call-report":
        _end_of_call(message, state.call_service, state.call_registry)
    elif kind == "status-update":
        log.info("call status", extra={"call_id": _call_id(message), "status": message.get("status")})
    return {}


def _call_id(message: dict[str, Any]) -> str:
    return str((message.get("call") or {}).get("id") or "")


def _caller_number(message: dict[str, Any]) -> str | None:
    call = message.get("call") or {}
    customer = call.get("customer") or message.get("customer") or {}
    return customer.get("number")


def _new_toolbox(call_id, caller, patients, calls) -> CallToolbox:
    session = RegistrationSession(call_sid=call_id, patients=patients, caller_number=caller)
    _safe(calls.start, call_id, session.caller_number)
    resume = _safe(calls.find_resumable_draft, session.caller_number, call_id)
    return CallToolbox(
        session=session,
        calls=calls,
        resume_draft=dict(resume.draft_fields) if resume else None,
        resume_from_sid=resume.call_sid if resume else None,
    )


def _assistant_request(message, settings: Settings, patients, calls, registry: CallRegistry):
    call_id = _call_id(message)
    toolbox = _new_toolbox(call_id, _caller_number(message), patients, calls)
    registry.put(call_id, toolbox)

    resume_fields = [FIELD_LABELS.get(f, f) for f in toolbox.resume_draft] if toolbox.resume_draft else None
    log.info(
        "incoming call",
        extra={"call_id": call_id, "caller": toolbox.session.caller_number, "resume_offered": bool(resume_fields)},
    )
    return {
        "assistant": build_assistant(
            settings, caller_number=toolbox.session.caller_number, resume_fields=resume_fields
        )
    }


async def _tool_calls(message, patients: PatientService, calls: CallService, registry: CallRegistry):
    call_id = _call_id(message)
    toolbox = registry.get(call_id)
    if toolbox is None:
        # e.g. the server restarted mid-call: rebuild from the caller ID and the saved draft.
        toolbox = _new_toolbox(call_id, _caller_number(message), patients, calls)
        record_draft = _safe(calls.get_draft, call_id)
        if record_draft:
            toolbox.session.resume_from(record_draft)
        registry.put(call_id, toolbox)

    results = []
    for tool_call in message.get("toolCallList") or []:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or ""
        args = function.get("arguments") or {}
        if isinstance(args, str):  # tolerate stringified arguments
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        result = await toolbox.call(name, args)
        results.append({"name": name, "toolCallId": tool_call.get("id"), "result": json.dumps(result)})

    _safe(calls.save_progress, call_id, toolbox.session.draft(), None)
    return {"results": results}


def _end_of_call(message, calls: CallService, registry: CallRegistry) -> None:
    call_id = _call_id(message)
    toolbox = registry.pop(call_id)
    session = toolbox.session if toolbox else None
    artifact = message.get("artifact") or {}
    transcript = [
        {"speaker": "caller" if m.get("role") == "user" else "agent", "text": m.get("message", "")}
        for m in artifact.get("messages") or []
        if m.get("role") in ("user", "bot", "assistant") and m.get("message")
    ]
    patient_id = session.saved_patient.patient_id if session and session.saved_patient else None
    draft = session.draft() if session else {}

    _safe(calls.finish, call_id, patient_id=patient_id, draft_fields=draft, transcript=transcript)
    log.info(
        "call finished",
        extra={
            "call_id": call_id,
            "ended_reason": message.get("endedReason"),
            "outcome": "registered" if patient_id else "not_registered",
            "patient_id": str(patient_id) if patient_id else None,
            "final_payload": draft,
            "turns": len(transcript),
        },
    )


def _safe(func, *args, **kwargs):
    """Call-record bookkeeping must never break a live call."""
    try:
        return func(*args, **kwargs)
    except PersistenceError:
        log.warning("call record write failed", extra={"op": func.__name__})
        return None
