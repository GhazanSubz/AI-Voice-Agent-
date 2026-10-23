"""The Vapi assistant definition, built per call in response to Vapi's assistant-request.

Building it in code (instead of clicking it together in the Vapi dashboard) keeps the whole agent
configuration in version control, and lets each call get a prompt that knows the caller ID and
whether there is a dropped registration to resume.
"""

from typing import Any

from app.config import Settings
from app.voice.prompts import build_system_prompt, first_message
from app.voice.tools import TOOL_SCHEMAS

# Our end_call is replaced by Vapi's built-in hang-up tool, which actually ends the phone call.
_VAPI_BUILTIN = {"end_call"}
VAPI_END_CALL_TOOL = "endCall"


def auth_headers(settings: Settings) -> dict[str, str]:
    # Vapi does not attach dashboard credentials to inline (per-call) assistants, so every server
    # URL inside it carries the shared secret explicitly. See app/vapi/webhook.py for the check.
    return {"X-Vapi-Secret": settings.vapi_secret}


def build_assistant(
    settings: Settings,
    *,
    caller_number: str | None,
    resume_fields: list[str] | None,
) -> dict[str, Any]:
    base = settings.public_base_url
    headers = auth_headers(settings)
    webhook = {"url": f"{base}/vapi/webhook", "headers": headers, "timeoutSeconds": 20}

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object", "properties": t["properties"], "required": t["required"]},
            },
            "server": webhook,
        }
        for t in TOOL_SCHEMAS
        if t["name"] not in _VAPI_BUILTIN
    ]
    tools.append({"type": "endCall"})

    return {
        "name": "Sam - patient intake",
        "firstMessage": first_message(resume_available=bool(resume_fields)),
        "model": {
            "provider": "groq",
            "model": settings.vapi_llm_model,
            "temperature": 0.3,  # warm but consistent; tool arguments must be exact
            "maxTokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        caller_number=caller_number,
                        resume_fields=resume_fields,
                        end_call_tool=VAPI_END_CALL_TOOL,
                    ),
                }
            ],
            "tools": tools,
        },
        # Deepgram's "multi" mode transcribes English and Spanish without a manual switch.
        "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "multi"},
        # Kokoro-82M on our own server (app/vapi/tts.py).
        "voice": {
            "provider": "custom-voice",
            "server": {"url": f"{base}/vapi/tts", "headers": headers, "timeoutSeconds": 30},
        },
        "server": webhook,
        "serverMessages": ["tool-calls", "end-of-call-report", "status-update"],
        "endCallMessage": "Thanks for calling. Goodbye!",
        "backgroundSound": "off",
        "maxDurationSeconds": 900,
    }
