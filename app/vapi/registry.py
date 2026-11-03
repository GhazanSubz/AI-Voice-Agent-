"""In-memory map of live Vapi calls to their CallToolbox.

Vapi's webhooks are stateless HTTP requests, so the per-call RegistrationSession has to live
somewhere between them. One process serves every call, so a dict is enough; the durable copy
(draft fields + transcript) is written to call_records after every tool call, so a restart
mid-call loses nothing the resume flow cannot recover.
"""

import threading
import time

from app.vapi.speech_gate import SpeechGate
from app.voice.tools import CallToolbox

# A call can't outlive Vapi's max duration; anything older is a call whose report never arrived.
_STALE_AFTER_SECS = 2 * 60 * 60


class CallRegistry:
    def __init__(self, speech_hold_secs: float = 3.0):
        self._calls: dict[str, tuple[float, CallToolbox]] = {}
        self._gates: dict[str, SpeechGate] = {}
        self._speech_hold_secs = speech_hold_secs
        self._lock = threading.Lock()

    def put(self, call_id: str, toolbox: CallToolbox) -> None:
        with self._lock:
            self._evict_stale()
            self._calls[call_id] = (time.monotonic(), toolbox)

    def get(self, call_id: str) -> CallToolbox | None:
        with self._lock:
            entry = self._calls.get(call_id)
            return entry[1] if entry else None

    def pop(self, call_id: str) -> CallToolbox | None:
        with self._lock:
            self._gates.pop(call_id, None)
            entry = self._calls.pop(call_id, None)
            return entry[1] if entry else None

    def speech_gate(self, call_id: str) -> SpeechGate:
        """The call's speech gate, created on first use (TTS can arrive before any tool call)."""
        with self._lock:
            gate = self._gates.get(call_id)
            if gate is None:
                gate = self._gates[call_id] = SpeechGate(self._speech_hold_secs)
            return gate

    def _evict_stale(self) -> None:
        cutoff = time.monotonic() - _STALE_AFTER_SECS
        for call_id in [cid for cid, (started, _) in self._calls.items() if started < cutoff]:
            del self._calls[call_id]
            self._gates.pop(call_id, None)
