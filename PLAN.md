# Voice AI Patient Registration Agent - Summary & Build Plan

## Context

CareCloud take-home (3 hours max): build a voice agent on a **real, dialable U.S. phone number**. It registers a new patient through natural conversation, saves the data in a persistent DB, and exposes the data through a REST API. Reviewers will call the number, register someone, call back, and check that the data is still there via the API.

Five criteria, 20% each: **Working system, Conversational quality, Technical architecture, Code quality & docs, Edge cases & resilience.** In their words: "a simple system that works flawlessly beats an over-engineered one that crashes on the first call."

User constraints (decided):
- **$0 cost and open-source models**: open-weights Llama 3.3 70B (LLM) and Whisper-large-v3-turbo (STT) on Groq's free tier, plus **Kokoro-82M** TTS running locally (Apache-2.0).
- **Hosting**: the user's Windows laptop behind a tunnel (ngrok free static domain, so the Twilio webhook URL never changes).
- **Telephony**: Twilio trial (free number + ~$15 credit) using raw Media Streams, so the STT/LLM/TTS pipeline is our own.

### Assignment requirements checklist
- Natural conversation (not an IVR) that handles varied phrasing, clarifications, corrections, interruptions and out-of-order answers.
- Required fields: first/last name, DOB (not future, MM/DD/YYYY), sex (Male/Female/Other/Decline to Answer), 10-digit US phone, address line 1, city, 2-letter state, ZIP (5 or ZIP+4).
- Optional fields, offered in one opt-in question: email, address line 2, insurance provider + member ID, preferred language (default English), emergency contact name + phone.
- Auto fields: patient_id (UUID), created_at, updated_at (UTC), deleted_at (soft delete).
- Reads every field back before saving, re-prompts for exactly the invalid field, closes with "You're all set, [First Name]" and hangs up.
- REST: `GET /patients?last_name=&date_of_birth=&phone_number=`, `GET/PUT/DELETE /patients/:id`, `POST /patients`. Envelope `{data, error}`; status codes 200/201/400/404/422/500; server-side validation; soft delete.
- Agent saves through the same service layer and tells the caller whether it succeeded, including a graceful message on failure.
- README covering setup, architecture, stack justification, env vars, limitations/trade-offs and Next Steps. No hardcoded keys. Conversations logged.
- Bonus: duplicate detection by phone, appointment scheduling (mock), Spanish switch, transcripts linked to the patient, dashboard, automated tests.

## Architecture

```
Caller -> Twilio number -> POST /voice (TwiML <Connect><Stream>) -> wss://<ngrok>/media
   Pipecat pipeline (per call):
   Twilio serializer -> Silero VAD (barge-in) -> Groq Whisper STT -> LLM context
     -> Groq Llama 3.3 70B (tool calls) -> Kokoro TTS (local CPU, sentence streaming) -> Twilio
   Tools -> RegistrationSession (slot state + validation) -> PatientService -> SQLite (WAL)
FastAPI also serves: /patients REST, /dashboard (HTML), /health
```

**Stack: Python 3.11, FastAPI, Pipecat (BSD, open source), SQLAlchemy 2 + SQLite file DB, Pydantic v2.** One process, one language, one shared service layer. Pipecat gives us VAD, interruptions and Twilio framing so we don't build them ourselves. SQLite keeps data across restarts and needs no ops (the brief itself suggests it). If Pipecat has no built-in Kokoro service, wrap `kokoro-onnx` in a small custom `TTSService` (check this first, about 10 min).

### Separation of concerns (repo layout)
```
app/
  main.py               FastAPI app, routers, startup (DB init, seed)
  config.py             pydantic-settings, env vars only
  db/models.py          SQLAlchemy Patient, CallTranscript, Appointment (CHECK constraints)
  db/session.py
  domain/schemas.py     Pydantic PatientCreate/Update/Out; the single source of validation rules
  domain/validators.py  normalize phone ("five five five..." -> digits), state names -> USPS code, DOB parsing
  services/patients.py  create/update/get/list/soft_delete/find_by_phone (used by API AND agent)
  api/patients.py       REST routes + envelope + error handlers (422 validation, 404, 500)
  api/dashboard.py      minimal HTML table of patients + transcripts
  voice/twilio_routes.py  /voice TwiML, /media websocket
  voice/pipeline.py     Pipecat pipeline assembly
  voice/session.py      RegistrationSession: slots, phase, validation via domain schemas
  voice/tools.py        LLM tool schemas + handlers
  voice/prompts.py      system prompt, heavily commented (graded!)
tests/                  pytest: API, validators, service, tool handlers, scripted text conversations
scripts/seed.py, scripts/chat_cli.py (talk to the agent in text, no phone)
README.md, .env.example, docs/AI_BRIEF.md
```

## Conversation design (where most of the score is)

- **The LLM drives the conversation, the code holds the truth.** The LLM never "remembers" fields. It calls `update_fields({...partial})` whenever the caller says anything relevant, in any order. The server validates each field with the same Pydantic rules the API uses and returns `{accepted, rejected: {field: reason}, still_missing}`. The prompt tells the LLM to re-ask only the rejected fields, with the reason ("that date is in the future").
- **Tools:** `update_fields`, `lookup_patient_by_phone` (dup detection, fired as soon as the phone is captured), `get_summary_for_readback`, `save_patient` (only allowed once phase=`confirmed`; the server enforces this), `start_over`, `schedule_appointment` (mock slots), `set_language`, `end_call`.
- **Phases** (tracked server-side, injected into context each turn): collecting_required -> offer_optional -> readback -> confirmed -> saved -> closing.
- **Voice-specific prompt rules:** 1-2 short sentences per turn, ask at most two related things together, no lists/markdown, read phones/ZIPs digit by digit, spell names back letter by letter in the readback, warm intake-coordinator persona, never invent data.
- **Corrections:** "Actually it's D-A-V-I-S" means call `update_fields` again. Spelled letters and NATO words get normalized in the validator.
- **Filler before slow tools** ("One moment while I save that") so the caller never hears silence.
- **Start over:** `start_over` clears the slots and the agent confirms.
- **Spanish:** "Hablo español" -> `set_language("es")` switches the prompt language, Kokoro Spanish voice and Whisper language hint, and sets preferred_language.

## Resilience
- **DB write failure:** one retry. If it still fails, the agent says it couldn't save and offers to try again or take a callback. The draft is kept in `call_drafts` and logged.
- **Call dropped mid-call:** partial slots are persisted per CallSid as a draft. On the next call from the same caller ID, the agent offers to pick up where they left off. A transcript is saved on disconnect.
- **LLM/STT errors or Groq rate limits (429):** fall back to `llama-3.1-8b-instant`, with a spoken apology rather than dead air.
- **Silence:** after ~8s, prompt "Are you still there?". After two tries, hang up politely.
- **Structured JSON logs** (call_sid, turn, tool calls, final payload) to stdout and `logs/`.

## Standout extras (priority order, only after the core works)
1. Duplicate detection -> offer to update (bonus + edge-case points).
2. Transcript stored per call and linked to patient_id, shown in the dashboard.
3. pytest suite + `chat_cli.py` scripted-conversation tests (invalid DOB, correction, start over, out-of-order answers).
4. Dashboard at `/dashboard`.
5. Spanish switch.
6. Mock appointment scheduling.
7. README with a sequence diagram, prompt-design rationale, latency numbers and an honest limitations section (trial preamble, laptop uptime, Groq free-tier limits).

## Timeline (3h)
| Time | Work |
|---|---|
| 0:00-0:20 | Accounts: Twilio trial number, Groq key, ngrok static domain. Scaffold the repo, `.env.example` |
| 0:20-0:55 | DB models, schemas/validators, service, REST API, seed, API tests |
| 0:55-1:50 | Pipecat pipeline, Twilio routes, tools, session state, prompt. **First real phone call by 1:30** |
| 1:50-2:25 | Iterate on the prompt with real calls; resilience paths; dup detection; transcripts |
| 2:25-2:45 | Dashboard, Spanish, scheduling (cut whatever doesn't fit) |
| 2:45-3:00 | README, final end-to-end call test, submit |

## Verification
- `pytest` passes (API status codes, envelope, validation 422s, soft delete hides records, the service layer).
- `scripts/chat_cli.py` runs the scripted flows: happy path, invalid DOB, 3-digit phone, spelling correction, start over, duplicate phone.
- **Real phone calls:** (1) register "Jane Doe" end to end; (2) restart the server, call again from the same phone, and the duplicate is recognized; (3) `curl $BASE/patients?last_name=Doe` returns her; (4) hang up mid-call, call back, and you're offered a resume; (5) stop the DB (rename the file / make it read-only) and the caller hears a graceful error.
- `/dashboard` shows the patients and transcripts.

## Known risks to state in the README
- The Twilio trial plays a "trial account" message and may require a key press before the call connects. Upgrading (~$20) removes it. If inbound calls from unverified numbers are blocked, document it and ask reviewers for numbers to verify.
- The laptop must stay awake, online and tunnelled during the review window (disable sleep).
- Groq free-tier rate/token limits. Kokoro CPU latency, so measure it and report it.

---

## AI handoff brief (to save as `docs/AI_BRIEF.md` + `CLAUDE.md` in the repo)

> **Task:** Build a phone-callable voice agent that registers U.S. patients (CareCloud take-home, 3h). Python 3.11 + FastAPI + Pipecat + SQLAlchemy/SQLite. Twilio trial number -> TwiML `<Connect><Stream>` -> `/media` websocket -> Pipecat: Silero VAD -> Groq Whisper-large-v3-turbo -> Groq `llama-3.3-70b-versatile` with tools -> Kokoro-82M (kokoro-onnx, local CPU) -> Twilio. Exposed via an ngrok static domain from a Windows laptop. Cost must stay $0, so use open-weights models only.
> **Non-negotiables:** real number works end to end; data persists across restarts; read-back + confirm before save; field-specific re-prompts; graceful spoken errors (never silence); REST `/patients` CRUD with `{data, error}` envelope, 200/201/400/404/422/500, soft delete, filters last_name/date_of_birth/phone_number; server-side validation shared by the API and the agent (`domain/schemas.py` is the single source of truth); env vars only (no keys in code); conversation + final payload logged.
> **Design rule:** the LLM converses, the server owns state. All captured data goes through `update_fields` -> validation -> `{accepted, rejected, still_missing}`. `save_patient` is refused server-side unless phase=confirmed.
> **Fields:** required first_name, last_name (1-50, letters/hyphen/apostrophe), date_of_birth (past, MM/DD/YYYY), sex (Male|Female|Other|Decline to Answer), phone_number (10-digit US), address_line_1, city (1-100), state (USPS 2-letter), zip_code (5 or ZIP+4). Optional: email, address_line_2, insurance_provider, insurance_member_id (alphanumeric), preferred_language (default English), emergency_contact_name, emergency_contact_phone. Auto: patient_id UUID, created_at/updated_at UTC, deleted_at.
> **Bonus order:** dup detection by phone -> transcripts linked to patient -> tests -> dashboard -> Spanish -> mock appointments.
> **Style:** org rules apply: no em dashes, never edit auto-generated files, no AI co-author on commits.
