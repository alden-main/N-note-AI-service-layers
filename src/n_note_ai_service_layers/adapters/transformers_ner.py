"""Hugging Face 토큰 분류 모델로 전사문의 사람 이름을 찾는 로컬 NER 어댑터이다.

``uv sync --extra ner``로 ``transformers``·``torch``(CPU)를 설치해야 하며, 모델은 로컬
Hugging Face 캐시에서 읽는다(폐쇄망은 ``HF_HUB_OFFLINE=1``). 원문을 외부로 보내지 않는다.
"""

from __future__ import annotations

import threading
from typing import Any

from n_note_ai_service_layers.contracts.privacy import EntitySpan

# 모델별 사람 라벨 표기다(모두의 말뭉치 PS, 네이버 NER PER 등).
DEFAULT_PERSON_LABELS = ("PS", "PER", "PERSON")
# 모델 최대 입력(512 토큰)보다 충분히 짧게 자른다. 한국어는 대략 1~2자당 1토큰이다.
_CHUNK_CHARS = 300


class TransformersNerRecognizer:
    """토큰 분류 pipeline을 처음 사용할 때 한 번 올리고, 요청 스레드 간에 공유한다.

    입력: 모델 이름 또는 로컬 경로, 최소 신뢰도, 사람으로 볼 라벨, torch 스레드 수이다.
    출력: ``detect`` 호출 시 원문 offset 기준 ``PERSON`` 범위 목록이다.
    """

    def __init__(
        self, model: str, min_score: float = 0.8, person_labels: tuple[str, ...] = DEFAULT_PERSON_LABELS,
        threads: int | None = None, pipeline: Any | None = None,
    ) -> None:
        self.model = model
        self.min_score = min_score
        self.person_labels = frozenset(label.upper() for label in person_labels)
        self.threads = threads
        self._pipeline = pipeline
        self._lock = threading.Lock()

    def load(self) -> None:
        """모델을 미리 올린다. 기동 시 호출해 설정 오류를 첫 요청 전에 드러낸다."""
        with self._lock:
            if self._pipeline is None:
                import torch
                from transformers import pipeline

                if self.threads:
                    torch.set_num_threads(self.threads)
                self._pipeline = pipeline("token-classification", model=self.model, aggregation_strategy="simple", device=-1)

    def detect(self, text: str) -> list[EntitySpan]:
        if not text.strip():
            return []
        self.load()
        spans: list[EntitySpan] = []
        for offset, chunk in self._chunks(text):
            with self._lock:
                entities = self._pipeline(chunk)
            for entity in entities:
                label = str(entity.get("entity_group") or entity.get("entity") or "").upper().removeprefix("B-").removeprefix("I-")
                score = float(entity.get("score", 0.0))
                start, end = entity.get("start"), entity.get("end")
                if label not in self.person_labels or score < self.min_score or start is None or end is None:
                    continue
                spans.append(EntitySpan(int(start) + offset, int(end) + offset, "PERSON", score))
        return spans

    @staticmethod
    def _chunks(text: str) -> list[tuple[int, str]]:
        """공백 경계에서 잘라 이름이 두 조각으로 나뉘지 않게 한다."""
        chunks: list[tuple[int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + _CHUNK_CHARS, len(text))
            if end < len(text):
                space = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
                if space > start:
                    end = space + 1
            chunks.append((start, text[start:end]))
            start = end
        return chunks
