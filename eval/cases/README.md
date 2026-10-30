# 모델 비교 평가 사례

모든 사례는 **합성 자료**입니다. 실제 환자 음성·기록을 넣지 마세요.

```
eval/cases/<case_id>/
  case.json                  # schema_id, language_hint, expected(채점 기준)
  transcript.txt             # 전사문(있으면 STT 없이 LLM만 평가)
  audio.wav                  # 선택: 음성(있으면 STT까지 평가, transcript.txt보다 우선)
  reference_transcript.txt   # 선택: 음성의 정답 전사문(STT 문자 오류율 CER 계산)
```

`case.json`의 `expected`:

- `keywords`: `{필드 ID: [초안에 반드시 들어가야 할 핵심어]}` → `keyword_recall`
- `forbidden`: 초안 어디에도 나오면 안 되는 문자열(구술에 없는 날짜·진단 등) → `forbidden_cases`
