"""LLM tool definitions and handlers.

Kept independent of Pipecat: each tool is a JSON schema plus a plain function returning a dict.
The voice pipeline (app/voice/pipeline.py) and the text harness (scripts/chat_cli.py) both
adapt these, so a conversation tested in text exercises exactly the code a phone call does.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.services.calls import CallService
from app.voice.session import FIELD_LABELS, RegistrationSession

log = logging.getLogger(__name__)

_FIELD_PROPERTIES = {
    name: {"type": ["string", "null"], "description": f"The caller's {label}."}
    for name, label in FIELD_LABELS.items()
}
_FIELD_PROPERTIES["date_of_birth"]["description"] = "Date of birth as MM/DD/YYYY."
_FIELD_PROPERTIES["sex"]["description"] = "One of: Male, Female, Other, Decline to Answer."
_FIELD_PROPERTIES["phone_number"]["description"] = "10-digit U.S. phone number, digits only."
_FIELD_PROPERTIES["state"]["description"] = "U.S. state name or 2-letter abbreviation."
_FIELD_PROPERTIES["emergency_contact_phone"]["description"] = "10-digit U.S. phone number, digits only."
for _optional in ("email", "address_line_2", "insurance_provider", "insurance_member_id",
                  "preferred_language", "emergency_contact_name", "emergency_contact_phone"):
    _FIELD_PROPERTIES[_optional]["description"] += " Use null to remove it if the caller takes it back."

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "update_fields",
        "description": (
            "Record or correct any patient details the caller just gave, in any order. Pass only "
            "the fields mentioned. Returns accepted values, rejected fields with reasons, and what "
            "is still missing."
        ),
        "properties": {
            "fields": {"type": "object", "properties": _FIELD_PROPERTIES, "additionalProperties": False}
        },
        "required": ["fields"],
    },
    {
        "name": "use_existing_record",
        "description": "The caller confirmed they want to update their existing record instead of creating a new one.",
        "properties": {"patient_id": {"type": "string", "description": "patient_id from existing_records."}},
        "required": ["patient_id"],
    },
    {
        "name": "get_readback",
        "description": "Get every collected field in speakable form to read back before saving.",
        "properties": {},
        "required": [],
    },
    {
        "name": "save_patient",
        "description": "Save the registration. Only call after the readback and an explicit yes from the caller.",
        "properties": {
            "caller_confirmed": {"type": "boolean", "description": "True only if the caller said the readback is correct."}
        },
        "required": ["caller_confirmed"],
    },
    {
        "name": "start_over",
        "description": "Discard everything collected on this call because the caller wants to start over.",
        "properties": {},
        "required": [],
    },
    {
        "name": "set_language",
        "description": "Switch the conversation language. Use 'es' for Spanish, 'en' for English.",
        "properties": {"language": {"type": "string", "enum": ["en", "es"]}},
        "required": ["language"],
    },
    {
        "name": "resume_previous_call",
        "description": "Answer to whether the caller wants to continue the registration from their dropped call.",
        "properties": {"accept": {"type": "boolean"}},
        "required": ["accept"],
    },
    {
        "name": "end_call",
        "description": "Hang up. Call this right after your goodbye sentence.",
        "properties": {"reason": {"type": "string", "enum": ["completed", "caller_request", "other"]}},
        "required": ["reason"],
    },
]


@dataclass
class CallToolbox:
    """Binds the tools to one call's session and to pipeline side effects."""

    session: RegistrationSession
    calls: CallService | None = None
    resume_draft: dict[str, Any] | None = None
    resume_from_sid: str | None = None
    # Side effects only the transport layer can perform.
    on_language_change: Callable[[str], Awaitable[None]] | None = None
    on_end_call: Callable[[], Awaitable[None]] | None = None
    handlers: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = field(init=False)

    def __post_init__(self):
        self.handlers = {
            "update_fields": self._update_fields,
            "use_existing_record": self._use_existing_record,
            "get_readback": self._get_readback,
            "save_patient": self._save_patient,
            "start_over": self._start_over,
            "set_language": self._set_language,
            "resume_previous_call": self._resume_previous_call,
            "end_call": self._end_call,
        }

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = self.handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name}"}
        try:
            result = await handler(args or {})
        except Exception:
            # A tool bug must never become dead air: give the LLM something it can recover from.
            log.exception("tool failed", extra={"call_sid": self.session.call_sid, "tool": name})
            result = {"error": "internal error, apologize and ask the caller to repeat that"}
        log.info(
            "tool call",
            extra={"call_sid": self.session.call_sid, "tool": name, "args": args, "result": result},
        )
        return result

    async def _update_fields(self, args):
        fields = args.get("fields") or {}
        if not isinstance(fields, dict):
            return {"error": "fields must be an object of field name to value"}
        return self.session.update_fields(fields)

    async def _use_existing_record(self, args):
        return self.session.use_existing_record(str(args.get("patient_id", "")))

    async def _get_readback(self, _args):
        return self.session.get_readback()

    async def _save_patient(self, args):
        return self.session.save_patient(bool(args.get("caller_confirmed")))

    async def _start_over(self, _args):
        return self.session.start_over()

    async def _set_language(self, args):
        result = self.session.set_language(str(args.get("language", "")))
        if result.get("ok") and self.on_language_change:
            await self.on_language_change(self.session.language)
        return result

    async def _resume_previous_call(self, args):
        if not self.resume_draft:
            return {"resumed": False, "note": "There is no earlier registration to resume."}
        if not args.get("accept"):
            self.resume_draft = None
            return {"resumed": False, "note": "Start a fresh registration from their name."}
        result = self.session.resume_from(self.resume_draft)
        if self.calls and self.resume_from_sid:
            self.calls.mark_resumed(self.resume_from_sid)
        self.resume_draft = None
        return result

    async def _end_call(self, _args):
        if self.on_end_call:
            await self.on_end_call()
        return {"ok": True}
