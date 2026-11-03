"""One question per turn: stops Sam from stacking questions.

Live and text tests showed gpt-oss producing replies like "Did I get that right? What is your
last name? What is your last name? Could you spell...". Vapi sends each sentence to /vapi/tts as
a separate request, so a filter on one request cannot see the rest of the reply. This gate keeps
a little state per call instead:

- within one request, text after the first question is dropped;
- after a question has been sent to speech, anything else Sam says within `hold_secs` is dropped.
  Stacked questions arrive within a second or two of each other, while a genuine next reply
  cannot: the caller first has to hear the question, answer it, and the LLM has to respond;
- the hold ends early when the caller interrupts (Vapi's user-interrupted / speech-update
  events), and it always expires on its own, so a missed event can never leave Sam silent.
"""

import logging
import re
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

_QUESTION_END = re.compile(r"\?")


def cut_after_first_question(text: str) -> tuple[str, bool]:
    """Return the text up to and including its first question, and whether it had one."""
    match = _QUESTION_END.search(text)
    if not match:
        return text, False
    return text[: match.end()].strip(), True


class SpeechGate:
    def __init__(self, hold_secs: float = 3.0, clock: Callable[[], float] = time.monotonic):
        self.hold_secs = hold_secs
        self._clock = clock
        self._asked_at: float | None = None
        self._lock = threading.Lock()

    def filter(self, text: str, call_id: str = "") -> str:
        """The part of `text` that should be spoken now ("" to say nothing)."""
        with self._lock:
            now = self._clock()
            if self._asked_at is not None and now - self._asked_at < self.hold_secs:
                log.info("suppressed speech after a question", extra={"call_id": call_id, "text": text})
                return ""
            spoken, asked = cut_after_first_question(text)
            if spoken != text:
                log.info("dropped text after a question", extra={"call_id": call_id, "dropped": text[len(spoken):]})
            if asked:
                self._asked_at = now
            return spoken

    def caller_spoke(self) -> None:
        """The caller started talking: Sam may answer them right away."""
        with self._lock:
            self._asked_at = None
