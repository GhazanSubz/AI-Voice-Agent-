# Voice AI Patient Registration Agent

A phone agent ("Sam") that registers new U.S. patients through natural conversation, validates
every field server-side, reads everything back for confirmation, saves the record to a
persistent database, and exposes the data through a REST API and a web dashboard.

Built for the CareCloud Voice AI take-home. Total running cost: **$0** (free tiers only, open
models wherever possible).

## What a call sounds like

1. Sam greets the caller and asks for their first and last name.
2. Sam collects the required fields in a natural order. The caller can answer out of order,
   give several details at once, or correct earlier answers at any time.
3. Invalid answers are re-asked for that field only, with the reason ("that date is in the
   future", "I got 9 digits").
4. If the phone number matches an existing patient, Sam offers to update that record instead.
5. Sam offers the optional details once (insurance, emergency contact, preferred language).
6. Sam reads everything back, spelling names letter by letter, and asks for confirmation.
   Corrections are re-read on their own.
7. Only after a clear yes is the record saved. Sam says "You're all set, <first name>" and
   hangs up. If the save fails, the caller hears an apology and an offer to retry, never silence.

The caller can say "start over" at any point. If a call drops mid-registration, the next call
from the same number offers to pick up where it left off.

## Architecture

```
 Caller ──PSTN──> Vapi (number, telephony, Deepgram STT, turn-taking)
                    │  1. assistant-request ─────────────┐
                    │  2. LLM turns ──> Groq: gpt-oss-120b│ (open weights, Apache-2.0)
                    │  3. tool-calls ────────────────────┤
                    │  4. voice-request (text) ──────────┤
                    │  5. end-of-call-report ────────────┤
                    ▼                                    ▼
             ngrok static domain ──> FastAPI app on the laptop (one process)
                                       ├─ /vapi/webhook   assistant config, tools, call end
                                       ├─ /vapi/tts       Kokoro-82M TTS (local CPU, Apache-2.0)
                                       ├─ /patients       REST API
                                       ├─ /calls          REST API (call records + transcripts)
                                       └─ /dashboard      web UI, reads the REST API
                                             │
                                    SQLite (WAL) file: data/patients.db
```

**Request flow for one call**

1. A call comes in. Vapi POSTs `assistant-request` to `/vapi/webhook`. The server creates a
   per-call `RegistrationSession`, checks for a dropped registration from the same number,
   and returns the full assistant definition built in code ([app/vapi/assistant.py](app/vapi/assistant.py)):
   model, system prompt, tools, transcriber, custom voice and turn-taking settings.
2. Vapi transcribes the caller and asks the LLM for each reply.
3. When the LLM calls a tool, Vapi POSTs `tool-calls`; the server runs it against the session
   and returns the result for the LLM to speak from.
4. Each reply is sent to `/vapi/tts`. Kokoro synthesizes it sentence by sentence and streams PCM
   back, so the caller hears the first words in under a second.
5. On hangup, `end-of-call-report` stores the transcript and outcome in `call_records`.

### Separation of concerns

```
app/
  domain/        Validation rules (Pydantic). Single source of truth for API and agent.
  db/            SQLAlchemy models with CHECK constraints, engine, seed data.
  services/      All database access. Patient CRUD, duplicate lookup, call records.
  api/           REST endpoints, {data, error} envelope, dashboard page.
  voice/         Provider-agnostic conversation: session state, tools, system prompt.
  vapi/          Vapi adapter only: webhook, per-call assistant, custom TTS, call registry.
scripts/chat_cli.py   Talk to the agent by text, same prompt and tools, no phone needed.
tests/                90 offline tests.
```

The telephony provider is isolated in `app/vapi/`. Everything that decides how the
conversation behaves lives in `app/voice/` and has no Vapi imports, which is why it could be
reused unchanged when the project switched from Twilio to Vapi.

### The central design rule: the LLM talks, the server owns the facts

The LLM never stores or validates patient data itself. Whenever the caller says anything
relevant, it calls `update_fields` with only what was just said. The server validates each field
with the same schemas the REST API uses and returns:

```json
{"accepted": {"last_name": "Davis, spelled D, A, V, I, S"},
 "rejected": {"date_of_birth": "date of birth cannot be in the future"},
 "still_missing_required": ["phone_number", "zip_code"],
 "phase": "collecting_required"}
```

So a value the agent accepts on the phone is exactly a value `POST /patients` would accept,
and the LLM always knows what to ask next. Saving is enforced server-side, not by the prompt:
`save_patient` is refused unless every required field is present, the readback happened, the
caller confirmed, and enough time has passed since "is that correct?" for a real answer.

| Tool | Purpose |
|---|---|
| `update_fields` | Record or correct any fields, in any order; returns accepted / rejected / missing |
| `use_existing_record` | Switch to updating a returning patient's record |
| `get_readback` | Speakable version of every field (digits grouped, names spelled) |
| `save_patient` | Create or update the patient; one automatic retry on DB failure |
| `start_over` | Discard everything collected on this call |
| `set_language` | Switch to Spanish (voice and prompt) or back |
| `resume_previous_call` | Accept or decline continuing a dropped registration |
| `endCall` | Vapi's built-in hang-up |

### Prompt engineering

The system prompt is in [app/voice/prompts.py](app/voice/prompts.py), with comments explaining
each section. Key decisions:

- **Grounding rules first.** In testing, gpt-oss would ask a question and then invent the
  caller's answer in the same turn. The prompt opens with blunt rules against that.
- **Written for the ear.** One question at a time, one or two short sentences, no lists or
  symbols, numbers read digit by digit, names spelled.
- **Mirrors the server's phases** (collect, offer optional, readback, confirm, save), so the
  prompt describes the flow and the server enforces it.
- **Every rule came from a failure seen in testing**, and the comments say which (for example
  "a single Spanish-looking word is usually the transcriber mishearing English").

The same prompt and tools can be exercised by text with `python -m scripts.chat_cli`.

## Tech stack and why

| Layer | Choice | Why |
|---|---|---|
| Telephony + STT | **Vapi** (free U.S. number, Deepgram nova-3) | Twilio offers no free trial from Pakistan, where this was built. Vapi provides a free number callable by anyone, handles turn-taking and interruptions, and lets us plug in our own LLM tools and voice. |
| LLM | **gpt-oss-120b on Groq** | Open weights (Apache-2.0), strong tool calling, fast. Llama 3.3 70B was the plan but Groq has retired it. |
| TTS | **Kokoro-82M** (int8 ONNX, local CPU) | Open source (Apache-2.0), good quality, English and Spanish voices, no per-character cost. About 0.7-1.0 s to first audio on a laptop CPU. |
| Backend | **Python 3.13, FastAPI** | One language and one process for API, webhook, TTS and dashboard. Pydantic gives one validation layer for both the API and the agent. |
| Database | **SQLite (WAL mode) via SQLAlchemy 2** | Survives restarts, needs no server, and the brief suggests it. SQLAlchemy keeps a move to Postgres to a change of `DATABASE_URL`. |
| Hosting | **Laptop + ngrok static domain** | Free, and the domain never changes, so the Vapi configuration stays valid. |

**What was tried and dropped:** the first build used Twilio Media Streams with a Pipecat
pipeline (Silero VAD, Groq Whisper, Kokoro). It was replaced when Twilio's free trial turned out
to be unavailable. The conversation layer (`app/voice/`) carried over unchanged.

## Setup

Prerequisites: **Python 3.13** (`kokoro-onnx` does not yet support 3.14), a free
[Groq](https://console.groq.com) key, a free [Vapi](https://dashboard.vapi.ai) account, and a
free [ngrok](https://ngrok.com) account with a static domain.

**1. Install**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

(On macOS or Linux use `.venv/bin/python`.)

**2. Download the Kokoro model files** (about 115 MB) into `models/`:

```bash
curl -L -o models/kokoro-v1.0.int8.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
```

```bash
curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

**3. Configure.** Copy `.env.example` to `.env` and fill it in; each setting is explained in
that file. Generate the shared secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**4. Run the server and the tunnel** (two terminals):

```bash
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
ngrok http --url=<your-static-domain> 8000
```

The database file, tables and two fictional demo patients are created on first start.

**5. Configure Vapi.** In the Vapi dashboard, open your phone number and:

- leave **Assistant** empty (the server provides it per call),
- set **Server URL** to `https://<your-static-domain>/vapi/webhook`,
- under Authorization, add the header `X-Vapi-Secret` with your `VAPI_SECRET` value, and save.

Call the number.

**6. Test without a phone** (optional): `python -m scripts.chat_cli` runs the same agent by
text against Groq. `--script file.txt` replays caller lines; `--caller +13125550110` simulates
caller ID.

**Run the tests**

```bash
.venv/Scripts/python -m pytest
```

90 tests, all offline: API status codes and envelope, validation, soft delete, persistence
across restarts, database failure handling, the session state machine, every tool, webhook
authentication, the assistant config, call resume, TTS streaming and speech cleanup, and the
dashboard.

## Observability

Every log line is one JSON object, written to stdout and `logs/app.log`. Per call you get
`incoming call`, each `tool call` with its arguments and result, `tts first audio` with
latency, `registration saved` with the **final payload**, and `call finished` with the outcome,
end reason and collected fields. Transcripts are stored in `call_records`, linked to the
patient, and shown on the dashboard.

## Known limitations and trade-offs

- **Testing is not yet complete.** Due to time constraints, end-to-end testing over live phone
  calls is still in progress.
- **The server runs on a laptop.** The phone line works only while the laptop is on, awake and
  online, with the tunnel running. A cloud VM would remove this; it was left out for cost.
- **Speech-to-text is Deepgram via Vapi, not open source.** It's the one non-open model.
  The first version used open Whisper on Groq through Pipecat and Twilio, which was dropped
  with Twilio.
- **The LLM sometimes invents answers or repeats itself.** gpt-oss occasionally fills in an
  answer the caller didn't give, or says a question twice. Mitigations: grounding rules in the
  prompt, server-side validation, the timing guard on saves, a readback the caller must
  confirm, and speech cleanup. The text harness also blocks any data tool called after an
  unanswered question. That stronger guard is **not** enforced on live calls, because Vapi's
  conversation snapshot can lag behind the caller's latest sentence and it refused real answers.
- **Spanish is partial.** The agent can switch its voice and language to Spanish, but the
  transcriber defaults to English for accuracy, so Spanish answers are recognized poorly unless
  `VAPI_TRANSCRIBER_LANGUAGE=multi`. With `multi`, a live test misheard "male" as "bien".
- **Free-tier limits.** Vapi's free credits run out (about $0.05 per minute). The personal Groq
  key used by the text harness allows 200,000 tokens per day, which heavy prompt testing hit.
- **Call sessions live in memory.** One process holds the live calls. A restart mid-call
  recovers from the saved draft, but scaling to several processes would need a shared store
  such as Redis.
- **No authentication on the REST API or dashboard.** They are public for review. Only the Vapi
  endpoints check a secret. Fine for fictional demo data, not for real patients.
- **Not HIPAA compliant**, as the brief says is expected. Do not enter real patient data.
- **Calls from outside the U.S.** may not connect, or may have poor audio. A test from Pakistan
  first failed to connect, then worked with recognition confidence around 54%.

## Next steps

- Host on a small always-on VM (or Fly.io) instead of the laptop, and drop the ngrok banner.
- Make the "agent asked, caller hasn't answered" guard reliable on Vapi by tracking turns from
  Vapi's `conversation-update` events instead of the snapshot attached to tool calls.
- Self-host STT (faster-whisper) to make the voice stack fully open source.
- Appointment scheduling after registration (bonus item, not built).
- API keys for the REST API and a login for the dashboard.
- Alembic migrations, and Postgres for anything beyond a demo.
- A recorded set of scripted conversations run in CI against the text harness.
