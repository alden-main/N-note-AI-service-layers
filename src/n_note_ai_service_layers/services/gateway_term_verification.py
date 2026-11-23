"""Gateway의 구조화 후보만 검증해 검토 artifact로 변환한다."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from n_note_ai_service_layers.contracts.gateway_payloads import KOREAN_FINDINGS, MEDICAL_CANDIDATES, MEDICAL_SUGGESTIONS, MedicalTermName
from n_note_ai_service_layers.contracts.privacy import PRIVACY_TOKEN
from n_note_ai_service_layers.contracts.term_verification import (
    GatewaySuggestion,
    GatewayTermOccurrence,
    GatewayTermVerificationResult,
    TermCategory,
    TermVerificationPhase,
)
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError
from n_note_ai_service_layers.settings import DEFAULT_TERM_REVIEW_CHUNK_CHARS, DEFAULT_TERM_REVIEW_TOOLS

_HANGUL = re.compile(r"[가-힣]")
# 일반 단어 뒤에 붙으면 용언 활용형으로 보는 어미다('말+라고', '설명+함', '변경+해').
_WORD_ENDINGS = frozenset({"라고", "하기", "하고", "하여", "하며", "했다", "해", "함", "고", "며", "다", "음", "임"})


class GatewayTermVerificationFlow:
    """Gateway가 보존한 upstream 구조화 출력을 원문 범위와 함께 검증한다.

    입력: Gateway client, 검토에 사용할 도구 이름 목록과 한 번에 보낼 최대 문자 수이다.
    출력: 원문 범위가 검증된 표준어·의학용어 후보이며 항상 ``auto_apply=false``이다.
    """

    MAX_CANDIDATES = 200
    MAX_SUGGESTIONS = 10

    def __init__(
        self, gateway: McpGatewayClient, tool_names: tuple[str, ...] = DEFAULT_TERM_REVIEW_TOOLS, chunk_chars: int = DEFAULT_TERM_REVIEW_CHUNK_CHARS,
    ) -> None:
        self.gateway = gateway
        self.tool_names = tool_names
        self.chunk_chars = chunk_chars

    def verify(self, phase: TermVerificationPhase, text: str, request_id: str | None = None, catalog: tuple[dict[str, Any], ...] | None = None) -> GatewayTermVerificationResult:
        """전사문 같은 단일 텍스트를 검토하고 원문 기준 offset 후보를 반환한다."""
        if not text.strip():
            return GatewayTermVerificationResult(phase, "ok")
        if catalog is None:
            try:
                catalog = self.gateway.discover(request_id)
            except McpGatewayError:
                return GatewayTermVerificationResult(phase, "unavailable")
        tools = [tool for tool in catalog if tool.get("name") in self.tool_names and self._accepts_text(tool)]
        if not tools:
            return GatewayTermVerificationResult(phase, "unsupported")
        occurrences: list[GatewayTermOccurrence] = []
        known_words: list[tuple[int, int]] = []
        unsupported = unavailable = False
        compact_tools = frozenset(tool["name"] for tool in tools if self._supports_compact(tool))
        tasks = [(tool["name"], offset, chunk) for tool in tools for offset, chunk in self._chunks(text) if chunk.strip()]
        for (tool_name, offset, chunk), result in zip(tasks, self._call_all(tasks, request_id, compact_tools), strict=True):
            if result is None:
                unavailable = True
                continue
            parsed = self._parse_result(result, phase, chunk, tool_name)
            if parsed is None:
                unsupported = True
                continue
            occurrences.extend(replace(item, start_offset=item.start_offset + offset, end_offset=item.end_offset + offset) for item in parsed)
            known_words.extend((start + offset, end + offset) for start, end in self._known_word_spans(result, chunk))
        occurrences = [
            item for item in occurrences
            if not PRIVACY_TOKEN.search(item.input_text) and not self._ordinary_word_guess(item, known_words)
        ]
        occurrences.sort(key=lambda item: (item.start_offset, item.end_offset, item.tool_name))
        tool_names = tuple(tool["name"] for tool in tools)
        if occurrences:
            return GatewayTermVerificationResult(phase, "ok", tuple(occurrences[:self.MAX_CANDIDATES]), tool_names)
        if unavailable:
            return GatewayTermVerificationResult(phase, "unavailable", tool_names=tool_names)
        return GatewayTermVerificationResult(phase, "unsupported" if unsupported else "ok", tool_names=tool_names)

    def verify_document(self, phase: TermVerificationPhase, document: DraftDocument, request_id: str | None = None, catalog: tuple[dict[str, Any], ...] | None = None) -> GatewayTermVerificationResult:
        """양식 문서를 한 번에 검토하고, 후보를 필드 값 기준 offset으로 변환한다.

        라벨·섹션 제목이나 필드 경계를 넘는 후보는 사용자가 적용할 수 없으므로 제외한다.
        """
        result = self.verify(phase, document.text, request_id, catalog)
        mapped: list[GatewayTermOccurrence] = []
        for item in result.occurrences:
            located = document.locate(item.start_offset, item.end_offset)
            if located is None:
                continue
            span, start, end = located
            mapped.append(replace(item, start_offset=start, end_offset=end, field_id=span.field_id))
        return replace(result, occurrences=tuple(mapped))

    def _call_all(self, tasks: list[tuple[str, int, str]], request_id: str | None, compact_tools: frozenset[str] = frozenset()) -> list[dict[str, Any] | None]:
        """도구·청크 호출을 Gateway 일괄 호출 한 번으로 보내고 입력 순서대로 결과(실패는 ``None``)를 반환한다."""
        calls = [(tool_name, {"text": chunk, **({"detail": "compact"} if tool_name in compact_tools else {})}) for tool_name, _, chunk in tasks]
        outcomes = self.gateway.call_tools(calls, request_id)
        return [None if isinstance(outcome, McpGatewayError) else outcome for outcome in outcomes]

    @staticmethod
    def _accepts_text(tool: dict[str, Any]) -> bool:
        """``{"text": ...}``만으로 호출할 수 있는 원문 범위 검토 도구인지 확인한다."""
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return isinstance(properties, dict) and isinstance(properties.get("text"), dict) and isinstance(required, list) and all(key == "text" for key in required)

    @staticmethod
    def _supports_compact(tool: dict[str, Any]) -> bool:
        """표준어 검토처럼 ``detail="compact"``를 받는 도구인지 확인한다.

        간결 응답에는 이 흐름이 쓰지 않는 사전 근거(뜻풀이 등)가 빠져 크기가 약 1/10이고, 결과 상한도
        교정 후보에만 적용되어 긴 청크가 잘리지 않는다. 이 옵션이 없는 구버전 서버에는 보내지 않는다.
        """
        detail = tool.get("inputSchema", {}).get("properties", {}).get("detail")
        return isinstance(detail, dict) and "compact" in (detail.get("enum") or ())

    def _chunks(self, text: str) -> list[tuple[int, str]]:
        """줄 경계를 우선해 upstream 길이 제한 이하의 조각과 시작 offset을 만든다."""
        chunks: list[tuple[int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_chars, len(text))
            if end < len(text):
                newline = text.rfind("\n", start, end)
                if newline > start:
                    end = newline + 1
            chunks.append((start, text[start:end]))
            start = end
        return chunks

    def _known_word_spans(self, result: dict[str, Any], source_text: str) -> list[tuple[int, int]]:
        """표준국어 검토가 사전 표면형과 정확히 일치(matched)한다고 확인한 원문 범위다."""
        payload = result.get("structuredContent")
        findings = payload.get("findings") if isinstance(payload, dict) else None
        if not isinstance(findings, list):
            return []
        return [
            (item["start"], item["end"])
            for item in findings
            if isinstance(item, dict) and item.get("status") == "matched"
            and self._valid_span(item.get("start"), item.get("end"), item.get("source_text"), source_text)
        ]

    @staticmethod
    def _ordinary_word_guess(item: GatewayTermOccurrence, known_words: list[tuple[int, int]]) -> bool:
        """의학용어 사전의 유사어 추정이 일반 단어 또는 그 활용형이면 오탐으로 본다.

        예: '말라고'→'말라리아', '설명함'→'설명하기'. 일부만 일반 단어인 오인식은 유지한다
        (예: '변증'을 포함한 '강경변증'→'간경변증', '당뇨'로 시작하는 '당뇨뼝'→'당뇨병').
        정확·변형 일치 후보도 유지한다.
        """
        if item.category != "medical_term" or not all(s.reason == "unknown_or_possible_typo" for s in item.suggestions):
            return False
        return any(
            (start <= item.start_offset and item.end_offset <= end)
            or (start == item.start_offset and item.input_text[end - start:] in _WORD_ENDINGS)
            for start, end in known_words
        )

    def _parse_result(self, result: dict[str, Any], phase: TermVerificationPhase, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        payload = result.get("structuredContent")
        # 예전 medic(FastMCP Union 반환)은 구조화 결과를 {"result": ...}로 감쌌다. 지금은 감싸지 않지만 둘 다 받는다.
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict) and not any(key in payload for key in ("findings", "candidates", "suggestions")):
            payload = payload["result"]
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("findings"), list):
            return self._parse_korean_findings(payload["findings"], phase, source_text, tool_name)
        if isinstance(payload.get("candidates"), list):
            return self._parse_medical_candidates(payload["candidates"], phase, source_text, tool_name)
        if isinstance(payload.get("suggestions"), list):
            return self._parse_medical_suggestions(payload["suggestions"], phase, source_text, tool_name)
        return None

    def _parse_korean_findings(self, findings: list[Any], phase: TermVerificationPhase, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        # 형태소 분석기가 복원해 원문에 없는 형태소다. 교정할 구간이 없을 뿐 응답 오류가 아니다.
        findings = [item for item in findings if not (isinstance(item, dict) and item.get("start") == item.get("end") and item.get("source_text") == "")]
        try:
            parsed_findings = KOREAN_FINDINGS.validate_python(findings)
        except ValidationError:
            return None
        parsed: list[GatewayTermOccurrence] = []
        for finding in parsed_findings:
            if not self._valid_span(finding.start, finding.end, finding.source_text, source_text):
                return None
            suggestions = [
                GatewaySuggestion(item.replacement, float(item.confidence) if item.confidence is not None else None, item.reason)
                for item in finding.candidates
            ]
            # 사전과 정확히 일치한 표기(matched)나 근거 없는 미등재어는 수정 후보가 아니다.
            suggestions = self._distinct(suggestions, finding.source_text)
            if suggestions:
                parsed.append(self._occurrence(
                    phase, tool_name, finding.start, finding.end, finding.source_text, finding.analysis_form, suggestions, "standard_korean",
                ))
        return parsed

    def _parse_medical_candidates(self, candidates: list[Any], phase: TermVerificationPhase, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        try:
            parsed_candidates = MEDICAL_CANDIDATES.validate_python(candidates)
        except ValidationError:
            return None
        parsed: list[GatewayTermOccurrence] = []
        for candidate in parsed_candidates:
            if not self._valid_span(candidate.start_offset, candidate.end_offset, candidate.source_text, source_text):
                return None
            if candidate.candidate_type == "exact_match":
                continue
            suggestions = [GatewaySuggestion(self._term_text(term, candidate.source_text), reason=candidate.candidate_type) for term in candidate.terms]
            suggestions = self._distinct(suggestions, candidate.source_text, "medical_term")
            if suggestions:
                parsed.append(self._occurrence(
                    phase, tool_name, candidate.start_offset, candidate.end_offset, candidate.source_text, candidate.source_text, suggestions, "medical_term",
                ))
        return parsed

    def _parse_medical_suggestions(self, suggestions: list[Any], phase: TermVerificationPhase, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        try:
            parsed_suggestions = MEDICAL_SUGGESTIONS.validate_python(suggestions)
        except ValidationError:
            return None
        parsed: list[GatewayTermOccurrence] = []
        for suggestion in parsed_suggestions:
            if not self._valid_span(suggestion.start_offset, suggestion.end_offset, suggestion.source_text, source_text):
                return None
            term = suggestion.recommended_term
            if suggestion.review_status == "no_change" or term is None:
                continue
            items = self._distinct([GatewaySuggestion(self._term_text(term, suggestion.source_text), reason=suggestion.reason)], suggestion.source_text, "medical_term")
            if items:
                parsed.append(self._occurrence(
                    phase, tool_name, suggestion.start_offset, suggestion.end_offset, suggestion.source_text, suggestion.source_text, items, "medical_term",
                ))
        return parsed

    def _occurrence(self, phase: TermVerificationPhase, tool_name: str, start: int, end: int, input_text: str, normalized: str, suggestions: list[GatewaySuggestion], category: TermCategory) -> GatewayTermOccurrence:
        return GatewayTermOccurrence(phase, tool_name, start, end, input_text, normalized, tuple(suggestions[:self.MAX_SUGGESTIONS]), category=category)

    @staticmethod
    def _term_text(term: MedicalTermName, input_text: str) -> str:
        """원문이 한글이면 한글 용어를, 영문 약어·용어이면 영문 용어를 우선 제안한다."""
        preferred, fallback = (term.kor_name, term.eng_name) if _HANGUL.search(input_text) else (term.eng_name, term.kor_name)
        return preferred or fallback

    @staticmethod
    def _same_term(input_text: str, suggestion: str) -> bool:
        """조사·어미만 붙은 같은 용어(통증은→통증)나 더 긴 합성어(수치→수치값)는 교정이 아니다."""
        source, target = input_text.casefold(), suggestion.casefold().rstrip("-")
        return bool(target) and (source.startswith(target) or target.startswith(source))

    @classmethod
    def _distinct(cls, suggestions: list[GatewaySuggestion], input_text: str, category: TermCategory = "standard_korean") -> list[GatewaySuggestion]:
        seen: set[str] = set()
        distinct: list[GatewaySuggestion] = []
        for item in suggestions:
            if not item.text or item.text == input_text or item.text in seen:
                continue
            if category == "medical_term" and cls._same_term(input_text, item.text):
                continue
            seen.add(item.text)
            distinct.append(item)
        return distinct

    @staticmethod
    def _valid_span(start: Any, end: Any, input_text: Any, source_text: str) -> bool:
        """``[start, end)``가 원문 안에 있고 그 구간이 ``input_text``와 같은지다(정수는 bool 제외)."""
        return (
            type(start) is int and type(end) is int and 0 <= start < end <= len(source_text)
            and isinstance(input_text, str) and source_text[start:end] == input_text
        )
