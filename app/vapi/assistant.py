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
GOODBYE = "Thanks for calling, and take care. Goodbye!"


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
            # An empty request-start message silences Vapi's default filler ("Just a sec", "Give me
            # a moment"), which live calls showed after nearly every answer.
            "messages": [{"type": "request-start", "content": ""}],
        }
        for t in TOOL_SCHEMAS
        if t["name"] not in _VAPI_BUILTIN
    ]
    # The goodbye belongs to the hang-up tool; the prompt tells the model not to say it as well.
    tools.append({"type": "endCall", "messages": [{"type": "request-start", "content": GOODBYE}]})

    return {
        "name": "Sam - patient intake",
        "firstMessage": first_message(resume_available=bool(resume_fields)),
        "model": {
            "provider": "groq",
            "model": settings.vapi_llm_model,
            "temperature": settings.llm_temperature,
            # gpt-oss reasons before answering and those tokens count here; at 300 a confusing
            # utterance used the whole budget on reasoning and the caller heard silence.
            "maxTokens": 1000,
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
        # See Settings.vapi_transcriber_language for the English-accuracy vs Spanish trade-off.
        "transcriber": {"provider": "deepgram", "model": "nova-3", "language": settings.vapi_transcriber_language},
        # Kokoro-82M on our own server (app/vapi/tts.py).
        "voice": {
            "provider": "custom-voice",
            "server": {"url": f"{base}/vapi/tts", "headers": headers, "timeoutSeconds": 30},
        },
        # Turn-taking, tuned from live calls:
        # - Callers read numbers in chunks and pause between first and last name ("Ladan ... and
        #   my last name is"). Ending the turn on punctuation cut them off, so a LiveKit model
        #   judges whether the sentence sounds finished (English only, matching the transcriber)
        #   and waits up to 2 s when it doesn't. The transcription plan is the fallback.
        # - Callers answer while Sam is still asking ("male" mid-question). With numWords 2 a
        #   one-word answer never interrupted, so Sam talked over it. 0 stops on any voice.
        "startSpeakingPlan": {
            "waitSeconds": 0.6,
            "smartEndpointingPlan": {"provider": "livekit", "waitFunction": "2000 / (1 + exp(-10 * (x - 0.5)))"},
            "transcriptionEndpointingPlan": {
                "onPunctuationSeconds": 0.8,
                "onNoPunctuationSeconds": 1.5,
                "onNumberSeconds": 1.8,
            },
        },
        "stopSpeakingPlan": {"numWords": 0, "voiceSeconds": 0.3, "backoffSeconds": 1},
        "server": webhook,
        # user-interrupted / speech-update let the speech gate (speech_gate.py) know the caller
        # started talking, so a barge-in answer is never held back.
        "serverMessages": ["tool-calls", "end-of-call-report", "status-update", "user-interrupted", "speech-update"],
        "backgroundSound": "off",
        "maxDurationSeconds": 900,
    }
