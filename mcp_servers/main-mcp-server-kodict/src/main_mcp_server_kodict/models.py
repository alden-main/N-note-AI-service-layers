"""사전 조회와 텍스트 검토에 사용하는 불변 데이터 모델이다."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Source:
    """사전 원본 행의 추적 정보다."""

    file_name: str
    row_number: int
    source_hash: str


@dataclass(frozen=True)
class Evidence:
    """후보를 뒷받침하는 사전 원본 근거다."""

    headword: str
    pronunciation: str
    part_of_speech: str
    definition: str
    source: Source

    def as_dict(self) -> dict[str, object]:
        """MCP JSON 응답용 사전으로 변환한다."""
        return asdict(self)


@dataclass(frozen=True)
class Entry:
    """검색 색인으로 찾은 사전 항목과 근거다."""

    entry_id: str
    surface: str
    evidence: Evidence

    def as_dict(self) -> dict[str, object]:
        """외부에 필요한 근거만 담은 사전으로 변환한다."""
        return {"entry_id": self.entry_id, "surface": self.surface, "evidence": self.evidence.as_dict()}

