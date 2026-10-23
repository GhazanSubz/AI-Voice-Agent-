"""Talk to the intake agent in text, no phone needed.

Uses the same system prompt, tool schemas, CallToolbox and database as a real call, against the
same Groq model Vapi uses, so prompt changes can be iterated on in seconds.

    python -m scripts.chat_cli                       # interactive
    python -m scripts.chat_cli --script lines.txt    # replay caller lines (one per line)
    python -m scripts.chat_cli --caller +13125550110 # pretend caller ID
"""

import argparse
import asyncio
import json
import sys
import uuid

from openai import AsyncOpenAI

from app.config import get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.services.calls import CallService
from app.services.patients import PatientService
from app.voice.prompts import build_system_prompt, first_message
from app.voice.session import FIELD_LABELS, RegistrationSession
from app.voice.tools import TOOL_SCHEMAS, CallToolbox

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": {"type": "object", "properties": t["properties"], "required": t["required"]},
        },
    }
    for t in TOOL_SCHEMAS
]


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # model output can contain non-ASCII punctuation
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", help="file of caller lines to replay")
    parser.add_argument("--caller", help="simulated caller ID, e.g. +13125550110")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.groq_api_key:
        sys.exit("Set GROQ_API_KEY in .env first.")

    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    patients, calls = PatientService(factory), CallService(factory)

    call_sid = f"CLI{uuid.uuid4().hex[:12]}"
    session = RegistrationSession(call_sid=call_sid, patients=patients, caller_number=args.caller)
    calls.start(call_sid, session.caller_number)
    resume = calls.find_resumable_draft(session.caller_number, call_sid)
    ended = False

    async def end_call():
        nonlocal ended
        ended = True

    toolbox = CallToolbox(
        session=session,
        calls=calls,
        resume_draft=dict(resume.draft_fields) if resume else None,
        resume_from_sid=resume.call_sid if resume else None,
        on_end_call=end_call,
    )

    client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    system = build_system_prompt(
        caller_number=session.caller_number,
        resume_fields=[FIELD_LABELS.get(f, f) for f in resume.draft_fields] if resume else None,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": first_message(resume_available=bool(resume))},
    ]
    script = open(args.script, encoding="utf-8").read().splitlines() if args.script else None
    print(f"\nSAM: {messages[-1]['content']}")

    while not ended:
        if script is not None:
            if not script:
                break
            line = script.pop(0)
            print(f"\nCALLER: {line}")
        else:
            try:
                line = input("\nCALLER: ")
            except EOFError:
                break
        messages.append({"role": "user", "content": line})

        # Let the model speak, running any tool calls it makes until it produces plain text.
        while True:
            response = await client.chat.completions.create(
                model=settings.vapi_llm_model, messages=messages, tools=OPENAI_TOOLS, temperature=0.3
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if msg.content:
                print(f"\nSAM: {msg.content}")
            if not msg.tool_calls:
                break
            for tc in msg.tool_calls:
                tool_args = json.loads(tc.function.arguments or "{}")
                result = await toolbox.call(tc.function.name, tool_args)
                print(f"   [tool] {tc.function.name}({json.dumps(tool_args)}) -> {json.dumps(result)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            if ended:
                break

    patient_id = session.saved_patient.patient_id if session.saved_patient else None
    calls.finish(call_sid, patient_id=patient_id, draft_fields=session.draft(), transcript=[])
    print(f"\n--- call ended; saved patient_id={patient_id}")


if __name__ == "__main__":
    asyncio.run(main())
