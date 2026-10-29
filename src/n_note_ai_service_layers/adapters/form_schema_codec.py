"""``FormSchema``를 저장소 보관용 JSON 사전으로 변환한다."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from n_note_ai_service_layers.contracts.form_schema import (
    BridgeMapping, FormField, FormSchema, FormSection, PdfPlacement, SchemaStatus,
)


def schema_to_dict(schema: FormSchema) -> dict[str, Any]:
    raw = asdict(schema)
    raw["status"] = schema.status.value
    raw["approved_at"] = schema.approved_at.isoformat() if schema.approved_at else None
    return raw


def _placement(raw: dict[str, Any] | None) -> PdfPlacement | None:
    return PdfPlacement(**raw) if raw else None


def schema_from_dict(raw: dict[str, Any]) -> FormSchema:
    sections = tuple(
        FormSection(
            section["section_id"],
            tuple(
                FormField(**{**item, "options": tuple(item.get("options", ())), "pdf": _placement(item.get("pdf"))})
                for item in section["fields"]
            ),
            section.get("label"),
            section.get("display_order"),
        )
        for section in raw["sections"]
    )
    return FormSchema(
        raw["schema_id"],
        raw["version"],
        raw["purpose"],
        sections,
        tuple(raw.get("validation_rule_ids", ())),
        tuple(BridgeMapping(**item) for item in raw.get("bridge_mapping", ())),
        SchemaStatus(raw.get("status", SchemaStatus.DRAFT.value)),
        raw.get("approved_by"),
        datetime.fromisoformat(raw["approved_at"]) if raw.get("approved_at") else None,
        dict(raw.get("freshness_windows_minutes", {})),
        raw.get("title"),
        raw.get("instructions"),
        raw.get("pdf_template"),
        tuple((name, _placement(placement)) for name, placement in raw.get("pdf_meta", ())),
        tuple(dict(item) for item in raw.get("validation_criteria", ())),
    )
