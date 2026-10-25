# 간호기록 양식 정의

AI 서비스는 시작할 때 `N_NOTE_FORMS_DIR`(기본 `forms/`, Docker에서는 `/app/forms`) 아래의 `*/form.json`을 읽어 **활성 양식**으로 등록합니다. 양식 7종은 각각 디렉터리 하나로 추가합니다.

```
forms/
  nursing-daily-note/
    form.json
    template.pdf        # 선택: 병원 PDF 양식
  sbar-handoff/
    form.json
```

| 환경 변수 | 설명 |
|---|---|
| `N_NOTE_FORMS_DIR` | 양식 디렉터리 |
| `N_NOTE_ENABLED_FORMS` | 사용할 `schema_id` 목록(쉼표 구분). 비우면 전체 |
| `N_NOTE_DEFAULT_FORM` | workflow 요청에 `schema_id`가 없을 때 쓸 양식 |

정의 파일이 잘못되면 서비스가 기동하지 않습니다(잘못된 양식으로 기록이 만들어지는 것을 막기 위함).

## form.json

```json
{
  "schema_id": "nursing-daily-note",
  "version": "nursing-daily-note/v1",
  "purpose": "NURSING_DAILY_NOTE",
  "title": "간호일지",
  "instructions": "LLM에 전달할 양식별 작성 지침(선택)",
  "pdf": {
    "template": "template.pdf",
    "meta": {
      "patient_id": {"page": 1, "x": 80, "y": 760, "width": 150, "height": 16},
      "finalized_at": {"field_name": "sign_date"}
    }
  },
  "validation_criteria": [],
  "sections": [
    {
      "section_id": "nursing_process",
      "label": "간호과정",
      "display_order": 1,
      "fields": [
        {
          "field_id": "assessment",
          "type": "text",
          "label": "간호사정 및 평가",
          "required": true,
          "description": "LLM 작성 지침",
          "display_order": 1,
          "pdf": {"field_name": "assessment"}
        }
      ]
    }
  ]
}
```

- `type`: `text`, `string`, `integer`, `number`, `boolean`, `array`, `date`, `datetime`, `enum`(`options` 필수).
- `required`: 확정(finalize) 전에 반드시 채워야 하는 항목입니다. 보고서 검증 MCP의 섹션별 기준으로도 사용됩니다.
- `llm_allowed: false`: LLM이 채우지 않고 사용자가 직접 입력하는 항목입니다.
- 양식을 바꿀 때는 `version`을 올립니다. 확정본은 작성 당시 양식 버전을 기록합니다.

## PDF 출력

- `pdf.template`이 **없으면**: 섹션·라벨·값 표 형태의 PDF를 생성합니다.
- `pdf.template`이 **있으면**: 그 PDF 위에 값을 그려 넣습니다. 필드마다 `pdf` 배치를 지정합니다.
  - `{"field_name": "..."}`: PDF에 입력 필드(AcroForm)가 있으면 그 필드 영역에 그립니다. 출력 후 입력 필드는 제거(평면화)됩니다.
  - `{"page": 1, "x": 72, "y": 600, "width": 400, "height": 60, "font_size": 10}`: 입력 필드가 없는 스캔·평면 PDF용 좌표입니다. 단위는 pt(1/72인치), 원점은 쪽 왼쪽 아래입니다.
  - 글이 영역보다 길면 글자 크기를 6pt까지 줄이고, 그래도 넘치면 `…(계속)`을 표시한 뒤 전체 내용을 마지막 "추가 기록" 쪽에 싣습니다.
  - 배치를 지정하지 않은 항목과 메타데이터(`patient_id`, `encounter_id`, `finalized_by`, `finalized_at`, `note_id`, `form_title`)도 "추가 기록" 쪽에 실리므로 누락되지 않습니다.
- PDF 양식 파일은 반드시 해당 양식 디렉터리 안에 있어야 합니다.

## validation_criteria (선택)

보고서 검증 MCP(`report-validation-standard/v1`)에 섹션별 필수 항목 기준 외의 기준을 추가합니다.

```json
{"criterion_id": "pain-detail", "description": "통증 호소 시 부위·강도·양상을 기록한다.",
 "required_elements": [{"element_id": "pain", "description": "통증 기록", "field_id": "assessment"}]}
```

`field_id`가 있는 요소는 그 필드의 작성 여부로 자동 평가합니다. `field_id`가 없는 기준은 `uncertain`으로 표시되어 간호사가 직접 확인합니다.
