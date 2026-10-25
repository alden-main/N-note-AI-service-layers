"""Gateway의 구조화 후보만 검증해 검토 artifact로 변환한다."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from n_note_ai_service_layers.contracts.term_verification import (
    GatewaySuggestion,
    GatewayTermOccurrence,
    GatewayTermVerificationResult,
    TermCategory,
)
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError

_HANGUL = re.compile(r"[가-힣]")


class GatewayTermVerificationFlow:
    """Gateway가 보존한 upstream 구조화 출력을 원문 범위와 함께 검증한다.

    입력: Gateway client, 검토에 사용할 도구 이름 목록과 한 번에 보낼 최대 문자 수이다.
    출력: 원문 범위가 검증된 표준어·의학용어 후보이며 항상 ``auto_apply=false``이다.
    """

    MAX_CANDIDATES = 200
    MAX_SUGGESTIONS = 10
    _PROTECTED_TOKEN = re.compile(r"\[\[[A-Z_]+_\d+\]\]")

    def __init__(self, gateway: McpGatewayClient, tool_names: tuple[str, ...] | None = None, chunk_chars: int | None = None) -> None:
        from n_note_ai_service_layers.settings import term_review_chunk_chars, term_review_tool_names
        self.gateway = gateway
        self.tool_names = tool_names if tool_names is not None else term_review_tool_names()
        self.chunk_chars = chunk_chars or term_review_chunk_chars()

    def verify(self, phase: str, text: str, request_id: str | None = None, catalog: tuple[dict[str, Any], ...] | None = None) -> GatewayTermVerificationResult:
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
        for tool in tools:
            for offset, chunk in self._chunks(text):
                if not chunk.strip():
                    continue
                try:
                    result = self.gateway.call_tool(tool["name"], {"text": chunk}, request_id)
                except McpGatewayError:
                    unavailable = True
                    continue
                parsed = self._parse_result(result, phase, chunk, tool["name"])
                if parsed is None:
                    unsupported = True
                    continue
                occurrences.extend(replace(item, start_offset=item.start_offset + offset, end_offset=item.end_offset + offset) for item in parsed)
                known_words.extend((start + offset, end + offset) for start, end in self._known_word_spans(result, chunk))
        occurrences = [
            item for item in occurrences
            if not self._PROTECTED_TOKEN.search(item.input_text) and not self._ordinary_word_guess(item, known_words)
        ]
        occurrences.sort(key=lambda item: (item.start_offset, item.end_offset, item.tool_name))
        tool_names = tuple(tool["name"] for tool in tools)
        if occurrences:
            return GatewayTermVerificationResult(phase, "ok", tuple(occurrences[:self.MAX_CANDIDATES]), tool_names)
        if unavailable:
            return GatewayTermVerificationResult(phase, "unavailable", tool_names=tool_names)
        return GatewayTermVerificationResult(phase, "unsupported" if unsupported else "ok", tool_names=tool_names)

    def verify_document(self, phase: str, document: DraftDocument, request_id: str | None = None, catalog: tuple[dict[str, Any], ...] | None = None) -> GatewayTermVerificationResult:
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

    @staticmethod
    def _accepts_text(tool: dict[str, Any]) -> bool:
        """``{"text": ...}``만으로 호출할 수 있는 원문 범위 검토 도구인지 확인한다."""
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return isinstance(properties, dict) and isinstance(properties.get("text"), dict) and isinstance(required, list) and all(key == "text" for key in required)

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
        """의학용어 사전의 유사어 추정이 표준국어 사전의 일반 단어 위에 걸치면 오탐으로 본다.

        예: '말라고'→'말라리아', '설명함'→'설명하기'. 정확·변형 일치 후보는 유지한다.
        """
        if item.category != "medical_term" or not all(s.reason == "unknown_or_possible_typo" for s in item.suggestions):
            return False
        return any(start < item.end_offset and item.start_offset < end for start, end in known_words)

    def _parse_result(self, result: dict[str, Any], phase: str, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        payload = result.get("structuredContent")
        # FastMCP는 Union 반환 타입의 구조화 결과를 {"result": ...}로 감싼다.
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

    def _parse_korean_findings(self, findings: list[Any], phase: str, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        parsed: list[GatewayTermOccurrence] = []
        for finding in findings:
            if not isinstance(finding, dict):
                return None
            start, end = finding.get("start"), finding.get("end")
            input_text, analysis_form, candidates = finding.get("source_text"), finding.get("analysis_form"), finding.get("candidates")
            if not self._valid_span(start, end, input_text, source_text) or not isinstance(analysis_form, str) or not isinstance(candidates, list):
                return None
            suggestions: list[GatewaySuggestion] = []
            for candidate in candidates:
                if not isinstance(candidate, dict) or not isinstance(candidate.get("replacement"), str):
                    return None
                confidence = candidate.get("confidence")
                if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)):
                    return None
                reason = candidate.get("reason") if isinstance(candidate.get("reason"), str) else None
                suggestions.append(GatewaySuggestion(candidate["replacement"], float(confidence) if confidence is not None else None, reason))
            # 사전과 정확히 일치한 표기(matched)나 근거 없는 미등재어는 수정 후보가 아니다.
            suggestions = self._distinct(suggestions, input_text)
            if suggestions:
                parsed.append(self._occurrence(phase, tool_name, start, end, input_text, analysis_form, suggestions, "standard_korean"))
        return parsed

    def _parse_medical_candidates(self, candidates: list[Any], phase: str, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        parsed: list[GatewayTermOccurrence] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("automatic_correction") is not False:
                return None
            start, end, input_text, terms = candidate.get("start_offset"), candidate.get("end_offset"), candidate.get("source_text"), candidate.get("terms")
            if not self._valid_span(start, end, input_text, source_text) or not isinstance(terms, list):
                return None
            if candidate.get("candidate_type") == "exact_match":
                continue
            suggestions: list[GatewaySuggestion] = []
            for term in terms:
                if not isinstance(term, dict) or not isinstance(term.get("kor_name"), str) or not isinstance(term.get("eng_name"), str):
                    return None
                suggestions.append(GatewaySuggestion(self._term_text(term, input_text), reason=candidate.get("candidate_type")))
            suggestions = self._distinct(suggestions, input_text, "medical_term")
            if suggestions:
                parsed.append(self._occurrence(phase, tool_name, start, end, input_text, input_text, suggestions, "medical_term"))
        return parsed

    def _parse_medical_suggestions(self, suggestions: list[Any], phase: str, source_text: str, tool_name: str) -> list[GatewayTermOccurrence] | None:
        parsed: list[GatewayTermOccurrence] = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                return None
            start, end, input_text, term = suggestion.get("start_offset"), suggestion.get("end_offset"), suggestion.get("source_text"), suggestion.get("recommended_term")
            if not self._valid_span(start, end, input_text, source_text):
                return None
            if suggestion.get("review_status") == "no_change" or term is None:
                continue
            if not isinstance(term, dict) or not isinstance(term.get("kor_name"), str) or not isinstance(term.get("eng_name"), str):
                return None
            items = self._distinct([GatewaySuggestion(self._term_text(term, input_text), reason=suggestion.get("reason"))], input_text, "medical_term")
            if items:
                parsed.append(self._occurrence(phase, tool_name, start, end, input_text, input_text, items, "medical_term"))
        return parsed

    def _occurrence(self, phase: str, tool_name: str, start: int, end: int, input_text: str, normalized: str, suggestions: list[GatewaySuggestion], category: TermCategory) -> GatewayTermOccurrence:
        return GatewayTermOccurrence(phase, tool_name, start, end, input_text, normalized, tuple(suggestions[:self.MAX_SUGGESTIONS]), category=category)

    @staticmethod
    def _term_text(term: dict[str, Any], input_text: str) -> str:
        """원문이 한글이면 한글 용어를, 영문 약어·용어이면 영문 용어를 우선 제안한다."""
        preferred, fallback = ("kor_name", "eng_name") if _HANGUL.search(input_text) else ("eng_name", "kor_name")
        return term[preferred] or term[fallback]

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
        return (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and 0 <= start < end <= len(source_text)
            and isinstance(input_text, str) and source_text[start:end] == input_text
        )
