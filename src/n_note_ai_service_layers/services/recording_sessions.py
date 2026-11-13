"""녹음 중 청크 단위로 받은 음성의 전사 결과를 모아 두는 세션 저장소다.

녹음하는 동안 청크마다 STT·비식별화를 해 화면에 전사문을 바로 보여 주고, 녹음이 끝나면 모아 둔 원본 전사문으로
초안 생성 작업을 시작한다. 청크마다 한 번만 전사하므로 녹음을 한 파일로 올릴 때와 STT 처리량이 같다.

원본 전사문(비식별화 전)은 프로세스 메모리에만 두고 디스크에 쓰지 않는다. 그래서 세션은 그 세션을 만든
worker 프로세스에서만 이어 쓸 수 있다(``N_NOTE_WORKERS`` > 1이면 세션 단위 sticky 라우팅이 필요하다).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from uuid import uuid4

from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.contracts.privacy import RedactedText

# 다음 청크 STT에 문맥으로 넘길 앞 전사문 길이(글자)다. Whisper 프롬프트 상한(약 224토큰) 안에서 기본 용어 힌트를 밀어내지 않게 짧게 둔다.
PRECEDING_TEXT_CHARS = 60


class RecordingSessionError(RuntimeError):
    """세션 상태 때문에 요청을 처리할 수 없다. ``code``로 원인을 구분한다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RecordingChunk:
    sequence: int
    # 비식별화 전 원본이다. 응답·로그에 넣지 않고 초안 생성 작업에만 넘긴다.
    raw_text: str
    redacted_text: str
    redaction_count: int
    provider: str
    model: str
    language: str | None


@dataclass
class RecordingSession:
    session_id: str
    trace_id: str
    patient_id: str
    encounter_id: str
    language_hint: str | None
    created_at: float
    last_activity: float
    chunks: list[RecordingChunk] = field(default_factory=list)
    # 녹음을 마치고 시작한 초안 생성 작업이다. 정해지면 더는 청크를 받지 않는다.
    job_id: str | None = None
    # 같은 세션의 청크는 순서대로 하나씩 전사한다(앞 청크 문맥을 다음 STT에 넘기고, 재전송을 중복 전사하지 않기 위해).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def next_sequence(self) -> int:
        return len(self.chunks)

    @property
    def redacted_transcript(self) -> str:
        return " ".join(chunk.redacted_text for chunk in self.chunks if chunk.redacted_text)

    def raw_transcript(self) -> Transcript:
        """청크 원본 전사문을 순서대로 이어 붙인 전체 전사 결과다."""
        text = " ".join(chunk.raw_text for chunk in self.chunks if chunk.raw_text)
        last = self.chunks[-1] if self.chunks else None
        language = next((chunk.language for chunk in self.chunks if chunk.language), None)
        return Transcript(text, last.provider if last else "unknown", last.model if last else "unknown", language)


class RecordingSessionStore:
    """프로세스 메모리의 녹음 세션 목록이다. 마지막 활동 후 ``idle_seconds``가 지나면 지운다."""

    def __init__(self, idle_seconds: float, max_sessions: int, max_chunks: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.idle_seconds = idle_seconds
        self.max_sessions = max_sessions
        self.max_chunks = max_chunks
        self._clock = clock
        self._sessions: dict[str, RecordingSession] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        cutoff = self._clock() - self.idle_seconds
        for session_id in [key for key, value in self._sessions.items() if value.last_activity < cutoff]:
            del self._sessions[session_id]

    def create(self, patient_id: str, encounter_id: str, language_hint: str | None, trace_id: str | None = None) -> RecordingSession:
        with self._lock:
            self._purge()
            if len(self._sessions) >= self.max_sessions:
                raise RecordingSessionError("too_many_sessions", "too many recording sessions are open")
            now = self._clock()
            session = RecordingSession(str(uuid4()), trace_id or str(uuid4()), patient_id, encounter_id, language_hint, now, now)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> RecordingSession | None:
        with self._lock:
            self._purge()
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def touch(self, session: RecordingSession) -> None:
        session.last_activity = self._clock()

    def add_chunk(
        self,
        session: RecordingSession,
        sequence: int,
        transcribe: Callable[[str | None], Transcript],
        redact: Callable[[str], RedactedText],
    ) -> tuple[RecordingChunk, bool]:
        """청크를 전사·비식별화해 세션에 붙인다. 두 번째 값은 새로 전사했는지 여부다.

        이미 받은 ``sequence``를 다시 보내면(응답 유실 후 재시도) 전사하지 않고 저장한 결과를 돌려준다.
        ``transcribe``는 앞 청크 전사문 끝부분(문맥)을 받아 STT를 호출한다. 실패하면 청크를 붙이지 않으므로 같은
        ``sequence``로 다시 보낼 수 있다.
        """
        with session.lock:
            if sequence < session.next_sequence:
                self.touch(session)
                return session.chunks[sequence], False
            if session.job_id is not None:
                raise RecordingSessionError("session_finished", "recording session already started a workflow")
            if sequence > session.next_sequence:
                raise RecordingSessionError("sequence_gap", f"expected chunk sequence {session.next_sequence}")
            if session.next_sequence >= self.max_chunks:
                raise RecordingSessionError("too_many_chunks", f"recording session accepts at most {self.max_chunks} chunks")
            previous = next((chunk.raw_text for chunk in reversed(session.chunks) if chunk.raw_text), "")
            transcript = transcribe(previous[-PRECEDING_TEXT_CHARS:].strip() or None)
            text = transcript.text.strip()
            redacted = redact(text) if text else RedactedText("", 0)
            chunk = RecordingChunk(sequence, text, redacted.text, redacted.replacement_count, transcript.provider, transcript.model, transcript.language)
            session.chunks.append(chunk)
            self.touch(session)
            return chunk, True

    def start_workflow(self, session: RecordingSession, submit: Callable[[Transcript], str]) -> tuple[str, bool]:
        """녹음을 마치고 초안 생성 작업을 시작한다. 이미 시작했으면 그 작업 ID를 돌려준다(두 번째 값 ``False``).

        ``submit``은 전체 원본 전사문으로 작업을 접수하고 작업 ID를 돌려준다. 실패하면 세션을 그대로 두어 다시 시도할 수 있다.
        """
        with session.lock:
            if session.job_id is not None:
                return session.job_id, False
            transcript = session.raw_transcript()
            if not transcript.text:
                raise RecordingSessionError("empty_transcript", "recording session has no transcribed speech")
            session.job_id = submit(transcript)
            # 작업이 전사문을 넘겨받았으므로 세션에는 화면용 비식별 전사문만 남긴다.
            session.chunks = [replace(chunk, raw_text="") for chunk in session.chunks]
            self.touch(session)
            return session.job_id, True
