"""확정 간호기록을 양식 PDF 위에 그리거나 섹션형 PDF 기록지로 렌더링한다."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from n_note_ai_service_layers.contracts.draft_review import FinalizedNursingNote
from n_note_ai_service_layers.contracts.form_schema import FormSchema, PdfPlacement
from n_note_ai_service_layers.services.draft_document import format_value, ordered_fields, ordered_sections

FONT_NAME = "NNoteKorean"
MIN_FONT_SIZE = 6.0
CONTINUED_MARK = " …(계속)"
META_LABELS = {
    "form_title": "양식",
    "patient_id": "환자 ID",
    "encounter_id": "내원 ID",
    "finalized_by": "확정자",
    "finalized_at": "확정 일시",
    "note_id": "기록 ID",
}


class PdfConfigurationError(RuntimeError):
    """한글 글꼴 또는 PDF 양식 설정이 올바르지 않을 때 발생한다."""


@dataclass(frozen=True)
class _Box:
    page: int
    x: float
    y: float
    width: float
    height: float
    font_size: float


class NursingNotePdfRenderer:
    """확정본만 PDF로 렌더링한다. 초안이나 후보는 렌더링 대상이 아니다.

    입력: PDF에 포함할 한글 TrueType 글꼴 경로이다.
    출력: ``render`` 호출 시 양식 PDF 합성본 또는 생성한 섹션형 PDF 바이트이다.
    """

    def __init__(self, font_path: Path | None) -> None:
        self.font_path = font_path

    def render(self, schema: FormSchema, note: FinalizedNursingNote) -> bytes:
        if (note.schema_id, note.schema_version) != (schema.schema_id, schema.version):
            raise ValueError("note does not belong to this form version")
        self._register_font()
        if schema.pdf_template:
            return self._render_on_template(schema, note)
        return self._render_generated(schema, note, [])

    def _register_font(self) -> None:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        if FONT_NAME in pdfmetrics.getRegisteredFontNames():
            return
        if self.font_path is None or not self.font_path.is_file():
            raise PdfConfigurationError("Korean TrueType font is not configured; set N_NOTE_PDF_FONT_PATH")
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(self.font_path)))
        except Exception as exc:
            raise PdfConfigurationError("N_NOTE_PDF_FONT_PATH must be a TrueType (.ttf) font") from exc

    # --- 양식 PDF 합성 -------------------------------------------------

    def _render_on_template(self, schema: FormSchema, note: FinalizedNursingNote) -> bytes:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import ArrayObject, NameObject
        from reportlab.pdfgen import canvas

        reader = PdfReader(schema.pdf_template)
        widgets = _widget_boxes(reader)
        values = {field.field_id: (field.label or field.field_id, format_value(note.fields.get(field.field_id)), field.pdf) for field in schema.fields}
        values.update({f"meta.{key}": (META_LABELS[key], value, placement) for key, value, placement in _meta_items(schema, note)})

        drawings: dict[int, list[tuple[_Box, str, str]]] = {}
        continuation: list[tuple[str, str]] = []
        for label, text, placement in values.values():
            box = _resolve_box(placement, widgets, len(reader.pages)) if placement else None
            if box is None:
                # 배치 정보가 없는 항목도 누락되지 않도록 추가 기록 쪽에 싣는다.
                if text.strip():
                    continuation.append((label, text))
                continue
            drawings.setdefault(box.page, []).append((box, label, text))

        overlay_buffer = BytesIO()
        overlay = canvas.Canvas(overlay_buffer)
        for index, page in enumerate(reader.pages):
            overlay.setPageSize((float(page.mediabox.width), float(page.mediabox.height)))
            for box, label, text in drawings.get(index, []):
                if _draw_in_box(overlay, box, text):
                    continuation.append((label, text))
            overlay.showPage()
        overlay.save()
        overlay_reader = PdfReader(BytesIO(overlay_buffer.getvalue()))

        writer = PdfWriter(clone_from=schema.pdf_template)
        for index, page in enumerate(writer.pages):
            page.merge_page(overlay_reader.pages[index])
            # 빈 입력 위젯이 합성한 글자를 가리지 않도록 양식 필드를 평면화한다.
            if "/Annots" in page:
                annots = [item for item in page["/Annots"] if item.get_object().get("/Subtype") != "/Widget"]
                if annots:
                    page[NameObject("/Annots")] = ArrayObject(annots)
                else:
                    del page[NameObject("/Annots")]
        if "/AcroForm" in writer.root_object:
            del writer.root_object[NameObject("/AcroForm")]
        if continuation:
            extra = PdfReader(BytesIO(self._render_generated(schema, note, continuation)))
            for page in extra.pages:
                writer.add_page(page)
        output = BytesIO()
        writer.write(output)
        return output.getvalue()

    # --- 섹션형 PDF 생성 ------------------------------------------------

    def _render_generated(self, schema: FormSchema, note: FinalizedNursingNote, continuation: list[tuple[str, str]]) -> bytes:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        title_style = ParagraphStyle("title", fontName=FONT_NAME, fontSize=16, leading=22, spaceAfter=4 * mm, wordWrap="CJK")
        heading_style = ParagraphStyle("heading", fontName=FONT_NAME, fontSize=11.5, leading=16, spaceBefore=4 * mm, spaceAfter=2 * mm, wordWrap="CJK")
        cell_style = ParagraphStyle("cell", fontName=FONT_NAME, fontSize=9.5, leading=13.5, wordWrap="CJK")
        label_style = ParagraphStyle("label", parent=cell_style, textColor=colors.HexColor("#333333"))

        def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
            return Paragraph(escape(text).replace("\n", "<br/>") or "&nbsp;", style)

        def table(rows: list[tuple[str, str]]) -> Table:
            data = [[paragraph(label, label_style), paragraph(value, cell_style)] for label, value in rows]
            result = Table(data, colWidths=[42 * mm, None], repeatRows=0)
            result.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9a9a9a")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            return result

        story: list[Any] = []
        if continuation:
            story.append(paragraph(f"{schema.display_title} — 추가 기록", title_style))
            story.append(table([(META_LABELS[key], value) for key, value, _ in _meta_items(schema, note) if key in ("patient_id", "note_id")]))
            story.append(paragraph("양식 영역에 배치되지 않았거나 영역을 넘은 항목", heading_style))
            story.append(table(continuation))
        else:
            story.append(paragraph(schema.display_title, title_style))
            story.append(table([(META_LABELS[key], value) for key, value, _ in _meta_items(schema, note) if key != "form_title"]))
            for section in ordered_sections(schema):
                story.append(paragraph(section.label or section.section_id, heading_style))
                story.append(table([(field.label or field.field_id, format_value(note.fields.get(field.field_id))) for field in ordered_fields(section)]))
            story.append(Spacer(1, 6 * mm))

        def footer(canvas_obj, document) -> None:
            canvas_obj.saveState()
            canvas_obj.setFont(FONT_NAME, 8)
            canvas_obj.drawRightString(A4[0] - 15 * mm, 10 * mm, f"{schema.display_title} · {schema.version} · {canvas_obj.getPageNumber()}쪽")
            canvas_obj.restoreState()

        buffer = BytesIO()
        document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=18 * mm, title=schema.display_title)
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        return buffer.getvalue()


def _meta_items(schema: FormSchema, note: FinalizedNursingNote) -> list[tuple[str, str, PdfPlacement | None]]:
    placements = dict(schema.pdf_meta)
    values = {
        "form_title": f"{schema.display_title} ({schema.version})",
        "patient_id": note.patient_id or "",
        "encounter_id": note.encounter_id or "",
        "finalized_by": note.finalized_by,
        "finalized_at": note.finalized_at.astimezone().strftime("%Y-%m-%d %H:%M"),
        "note_id": note.note_id,
    }
    return [(key, values[key], placements.get(key)) for key in META_LABELS]


def _widget_boxes(reader: Any) -> dict[str, _Box]:
    """AcroForm 위젯의 정규화된 전체 이름과 쪽·사각형 위치를 읽는다."""
    boxes: dict[str, _Box] = {}
    for page_index, page in enumerate(reader.pages):
        for annot in page.get("/Annots") or []:
            widget = annot.get_object()
            if widget.get("/Subtype") != "/Widget" or "/Rect" not in widget:
                continue
            names = []
            node = widget
            while node is not None:
                if "/T" in node:
                    names.append(str(node["/T"]))
                parent = node.get("/Parent")
                node = parent.get_object() if parent is not None else None
            if not names:
                continue
            x1, y1, x2, y2 = (float(value) for value in widget["/Rect"])
            boxes.setdefault(".".join(reversed(names)), _Box(page_index, min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), 10.0))
    return boxes


def _resolve_box(placement: PdfPlacement, widgets: dict[str, _Box], page_count: int) -> _Box | None:
    if placement.field_name:
        widget = widgets.get(placement.field_name)
        if widget is None:
            raise PdfConfigurationError(f"PDF template has no form field named {placement.field_name!r}")
        return _Box(widget.page, widget.x, widget.y, widget.width, widget.height, placement.font_size)
    if placement.has_box:
        if not 1 <= placement.page <= page_count:
            raise PdfConfigurationError(f"PDF placement page {placement.page} is outside the template")
        return _Box(placement.page - 1, placement.x, placement.y, placement.width, placement.height, placement.font_size)
    return None


def _wrap(text: str, font_size: float, width: float) -> list[str]:
    """공백 단위로 줄바꿈하고, 한 단어가 너비를 넘으면 글자 단위로 나눈다."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}" if current else word
            if stringWidth(candidate, FONT_NAME, font_size) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = ""
            for char in word:
                if stringWidth(current + char, FONT_NAME, font_size) > width and current:
                    lines.append(current)
                    current = char
                else:
                    current += char
        lines.append(current)
    return lines


def _draw_in_box(canvas_obj: Any, box: _Box, text: str) -> bool:
    """글자 크기를 줄여 영역에 맞추고, 최소 크기에서도 넘치면 잘린 표시를 남기고 ``True``를 반환한다."""
    if not text.strip():
        return False
    padding = 2.0
    width, height = max(box.width - 2 * padding, 1.0), max(box.height - 2 * padding, 1.0)
    size = box.font_size
    while True:
        leading = size * 1.25
        lines = _wrap(text, size, width)
        if len(lines) * leading <= height or size <= MIN_FONT_SIZE:
            break
        size = max(MIN_FONT_SIZE, size - 0.5)
    capacity = max(int(height // leading), 1)
    overflow = len(lines) > capacity
    if overflow:
        lines = lines[:capacity]
        lines[-1] = lines[-1] + CONTINUED_MARK
    canvas_obj.setFont(FONT_NAME, size)
    y = box.y + box.height - padding - size
    for line in lines:
        canvas_obj.drawString(box.x + padding, y, line)
        y -= leading
    return overflow
