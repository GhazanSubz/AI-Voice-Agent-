"""In-memory map of live Vapi calls to their CallToolbox.

Vapi's webhooks are stateless HTTP requests, so the per-call RegistrationSession has to live
somewhere between them. One process serves every call, so a dict is enough; the durable copy
(draft fields + transcript) is written to call_records after every tool call, so a restart
mid-call loses nothing the resume flow cannot recover.
"""

import threading
import time

from app.voice.tools import CallToolbox

# A call can't outlive Vapi's max duration; anything older is a call whose report never arrived.
_STALE_AFTER_SECS = 2 * 60 * 60


class CallRegistry:
    def __init__(self):
        self._calls: dict[str, tuple[float, CallToolbox]] = {}
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
            entry = self._calls.pop(call_id, None)
            return entry[1] if entry else None

    def _evict_stale(self) -> None:
        cutoff = time.monotonic() - _STALE_AFTER_SECS
        for call_id in [cid for cid, (started, _) in self._calls.items() if started < cutoff]:
            del self._calls[call_id]
