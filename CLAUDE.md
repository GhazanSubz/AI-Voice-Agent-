# AI Brief - Voice AI Patient Registration Agent

**Task:** Phone-callable voice agent that registers U.S. patients (CareCloud take-home, 3h). Python 3.13 + FastAPI + SQLAlchemy/SQLite. Cost must stay $0 and models open where possible.

**Architecture:** Vapi (free US number, telephony, Deepgram STT) -> `assistant-request` webhook returns a per-call assistant built in code (`app/vapi/assistant.py`): Groq-hosted open-weights LLM (`openai/gpt-oss-120b` by default), our tools served at `/vapi/webhook`, and a custom voice at `/vapi/tts` served by local Kokoro-82M. Exposed from a Windows laptop through an ngrok static domain. Twilio/Pipecat were dropped: Twilio offers no free trial from Pakistan.

**Layers:** `app/domain` (validation, single source of truth) -> `app/services` (DB access, used by API and agent) -> `app/api` (REST) and `app/voice` (provider-agnostic conversation: session state, tools, prompt) -> `app/vapi` (Vapi adapter only).

**Non-negotiables:** real number works end to end; data persists across restarts; read-back + confirm before save; field-specific re-prompts; graceful spoken errors (never silence); REST `/patients` CRUD with `{data, error}` envelope, 200/201/400/404/422/500, soft delete, filters last_name/date_of_birth/phone_number; env vars only (no keys in code); conversation + final payload logged.

**Design rule:** the LLM converses, the server owns state. All captured data goes through `update_fields` -> validation -> `{accepted, rejected, still_missing_required}`. `save_patient` is refused server-side unless readback happened and the caller confirmed.

**Testing:** `pytest` (offline). `python -m scripts.chat_cli` talks to the agent in text through the same tools and prompt.

**Style:** org rules apply: no em dashes, never edit auto-generated files, no AI co-author on commits.

Full original plan: see PLAN.md (its telephony section predates the Vapi switch).
