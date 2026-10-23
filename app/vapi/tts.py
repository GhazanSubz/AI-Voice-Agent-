"""Custom TTS for Vapi: POST /vapi/tts, served by Kokoro-82M (Apache-2.0) on our own CPU.

Vapi posts {"message": {"type": "voice-request", "text", "sampleRate", "call": {...}}} and expects
raw 16-bit little-endian mono PCM at exactly that sample rate. Kokoro synthesizes 24 kHz float
audio sentence by sentence; each chunk is resampled and streamed back immediately so the caller
hears the first words before the whole reply is synthesized.
"""

import asyncio
import logging
import re
import threading
import time

import numpy as np
import soxr
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.vapi.webhook import require_vapi_secret

log = logging.getLogger(__name__)

router = APIRouter(prefix="/vapi", tags=["telephony"])

_SUPPORTED_RATES = {8000, 16000, 22050, 24000, 44100}
_KOKORO_LANG = {"en": "en-us", "es": "es"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s*(?=[A-Z¿¡\"'(\[]|$)|(?<=[.!?])\s+")
_ASIDE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_META_SENTENCE = re.compile(
    r"\b(wait(ing)? for (the )?(user|caller|your|their)|awaiting (user|caller|your)|user input|"
    r"user (says|said|response))\b",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")
# Typographic punctuation the model likes -> what a TTS phonemizer reads cleanly.
_PUNCTUATION = str.maketrans({"‑": "-", "‐": "-", "–": ", ", "—": ", ", "’": "'", "‘": "'", "“": '"', "”": '"'})


class KokoroEngine:
    """Loads the ONNX model once per process; loading takes seconds, synthesis is per request."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._kokoro = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._kokoro is None:
                from kokoro_onnx import Kokoro  # heavy import, keep it off the API's startup path

                started = time.perf_counter()
                self._kokoro = Kokoro(self._settings.kokoro_model_path, self._settings.kokoro_voices_path)
                log.info("kokoro loaded", extra={"secs": round(time.perf_counter() - started, 2)})
            return self._kokoro

    async def warm_up(self) -> None:
        try:
            kokoro = await asyncio.to_thread(self.get)
            # The first inference after loading is several times slower; pay that cost now.
            async for _ in kokoro.create_stream("Hello.", voice=self.voice_for("en"), lang="en-us"):
                pass
        except Exception:
            log.exception("kokoro failed to load; calls will have no voice until this is fixed")

    def voice_for(self, language: str) -> str:
        return self._settings.kokoro_voice_es if language == "es" else self._settings.kokoro_voice_en


def clean_for_speech(text: str) -> str:
    """Drop text the LLM meant as stage directions rather than speech.

    gpt-oss sometimes appends asides such as "(Waiting for your answer.)" or "[Awaiting user
    response]", or a meta sentence like "We have to wait for user input." None of that should be
    spoken to a caller.
    """
    text = _ASIDE.sub(" ", text)
    kept: list[str] = []
    for sentence in split_sentences(text):
        # gpt-oss sometimes says the same question twice in a row; say it once.
        if _META_SENTENCE.search(sentence) or (kept and _norm(sentence) == _norm(kept[-1])):
            continue
        kept.append(sentence)
    text = " ".join(kept)
    return _SPACES.sub(" ", text.translate(_PUNCTUATION)).strip()


def _norm(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]", "", sentence.lower())


def split_sentences(text: str) -> list[str]:
    """Kokoro synthesizes a whole request before yielding, so long replies are fed sentence by
    sentence: the caller hears the first sentence while the rest is still being generated."""
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _to_pcm16(samples: np.ndarray, source_rate: int, target_rate: int) -> bytes:
    if source_rate != target_rate:
        samples = soxr.resample(samples, source_rate, target_rate)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@router.post("/tts", include_in_schema=False)
async def synthesize(request: Request):
    require_vapi_secret(request)
    message = (await request.json()).get("message") or {}
    text = clean_for_speech(message.get("text") or "")
    sample_rate = int(message.get("sampleRate") or 0)
    if message.get("type") != "voice-request" or not text or sample_rate not in _SUPPORTED_RATES:
        raise HTTPException(status_code=400, detail="expected a voice-request with text and a supported sampleRate")

    # The language follows the call: set_language("es") on the session switches the voice too.
    call_id = str((message.get("call") or {}).get("id") or "")
    toolbox = request.app.state.call_registry.get(call_id)
    language = toolbox.session.language if toolbox else "en"

    engine: KokoroEngine = request.app.state.tts_engine
    kokoro = await asyncio.to_thread(engine.get)
    voice = engine.voice_for(language)

    async def audio():
        started = time.perf_counter()
        first_chunk = True
        try:
            for sentence in split_sentences(text):
                async for samples, rate in kokoro.create_stream(sentence, voice=voice, lang=_KOKORO_LANG[language]):
                    if first_chunk:
                        log.info(
                            "tts first audio",
                            extra={"call_id": call_id, "ms": round((time.perf_counter() - started) * 1000), "chars": len(text)},
                        )
                        first_chunk = False
                    yield _to_pcm16(samples, rate, sample_rate)
        except Exception:
            # Headers are already sent, so all we can do is end the stream and leave a trace.
            log.exception("tts synthesis failed", extra={"call_id": call_id, "text": text})

    return StreamingResponse(audio(), media_type="application/octet-stream")
