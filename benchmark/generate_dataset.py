"""Create a reproducible, deliberately synthetic evaluation set; not clinical validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval/benchmark/v1'
# Each family is distinct; variants retain the same clinical facts and stay in one split.
SCENARIOS = [
 ('daily','normal','수술 부위 통증을 호소합니다. 심호흡을 안내했습니다.','수술 부위 통증과 심호흡 교육','통증','assessment'),
 ('daily','normal','어지러움을 호소합니다. 침상에서 쉬도록 안내했습니다.','어지러움과 침상 안정 안내','어지러움','assessment'),
 ('daily','normal','메스꺼움을 말합니다. 담당의에게 상태를 보고했습니다.','메스꺼움과 담당의 보고','메스꺼움','assessment'),
 ('daily','numeric','맥박은 분당 92회입니다. 담당의에게 보고했습니다.','맥박 분당 92회와 보고','92','assessment'),
 ('daily','privacy','환자명: 최라온님이 어지러움을 호소합니다. 침상 안정을 안내했습니다.','어지러움과 침상 안정','어지러움','assessment'),
 ('daily','difficult','어제는 기침이 없었고 오늘 기침이 있다고 다시 말씀하십니다. 상태를 관찰합니다.','현재 기침 여부와 관찰','기침','assessment'),
 ('daily','missing','상처 부위가 불편하다고 말합니다. 정확한 통증 점수는 말하지 않았습니다.','상처 불편감, 점수 미언급','불편','assessment'),
 ('daily','missing','속이 불편하다고 말합니다. 원인은 아직 확인되지 않았습니다.','복부 불편감, 원인 미확인','불편','assessment'),
 ('daily','numeric','체온은 38.2도입니다. 담당의에게 보고했습니다.','체온 38.2도와 보고','38.2','assessment'),
 ('daily','numeric','통증 점수는 4점입니다. 자세를 바꾸도록 도왔습니다.','통증 4점과 체위 변경','4','assessment'),
 ('daily','privacy','환자명: 김가람님이 목이 마르다고 말합니다. 물을 제공했습니다.','갈증과 물 제공','갈증','assessment'),
 ('daily','difficult','아침에는 통증이 심했는데 지금은 줄었다고 정정합니다. 상태를 관찰합니다.','정정된 현재 통증과 관찰','통증','assessment'),
 ('sbar','normal','인계합니다. 기침이 지속되고 있습니다. 다음 근무에 기침 상태를 관찰해 주세요.','기침 지속과 다음 근무 관찰','기침','a.status'),
 ('sbar','normal','인계할게요. 수면이 부족했다고 합니다. 다음 근무에 수면 상태를 확인해 주세요.','수면 부족과 다음 근무 확인','수면','a.status'),
 ('sbar','normal','교대 인계입니다. 식사량이 줄었습니다. 다음 근무에 섭취량을 확인해 주세요.','식사량 감소와 섭취량 확인','식사','a.status'),
 ('sbar','missing','인계합니다. 어지러움이 있습니다. 원인은 확인되지 않았고 다음 근무에 관찰해 주세요.','어지러움과 원인 미확인','어지러움','a.status'),
 ('sbar','missing','교대 인계입니다. 상처가 불편하다고 합니다. 통증 점수는 확인하지 못했습니다.','상처 불편과 점수 미확인','불편','a.status'),
 ('sbar','numeric','인계합니다. 체온은 37.8도입니다. 다음 근무에 체온을 다시 확인해 주세요.','체온 37.8도와 재측정','37.8','a.status'),
 ('sbar','privacy','인계합니다. 환자명: 박다온님이 기침합니다. 다음 근무에 상태를 관찰해 주세요.','기침과 다음 근무 관찰','기침','a.status'),
 ('sbar','difficult','인계합니다. 통증 점수를 처음에 6점이라 했다가 3점으로 바로잡았습니다. 다음 근무에 재확인해 주세요.','통증 점수 정정 3점과 재확인','3','a.status'),
]
# Five utterance variants per family, never used to construct the reference answer.
PREFIX = ['네, ', '지금 말씀드리면 ', '기록 부탁드립니다. ', '상태 말씀드립니다. ', '오늘 상황입니다. ']

def raw(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n').encode()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, references = [], []
    # Category counts are encoded in scenario allocation; a family never crosses splits.
    for family, (kind, category, text, summary, anchor, field) in enumerate(SCENARIOS, 1):
        for variant in range(5):
            number = (family-1)*5+variant+1
            case_id = f'syn-{number:03d}'
            transcript = PREFIX[variant]+text
            schema = 'nursing-daily-note' if kind == 'daily' else 'sbar-handoff'
            subject = hashlib.sha256(f'synthetic-patient-v1-{case_id}'.encode()).hexdigest()[:12].upper()
            inputs.append({'case_id':case_id, 'family':f'family-{family:02d}', 'category':category,
                           'split':'holdout' if family in {5,10,12,15,18,20} else 'development',
                           'request':{'patient_id':f'SYN-PATIENT-{subject}', 'encounter_id':f'SYN-ENCOUNTER-{subject}',
                                      'schema_id':schema, 'language_hint':'ko'},
                           'reference_time':'2026-11-18T09:00:00+09:00', 'transcript':transcript})
            references.append({'case_id':case_id, 'summary':summary,
                               'facts':[{'id':'primary', 'field_id':field, 'claim':summary, 'evidence':text,
                                         'anchors':[anchor], 'critical':True}],
                               'sensitive_values': (['최라온'] if family == 5 else ['김가람'] if family == 11 else ['박다온'] if family == 19 else []),
                               'must_be_empty': ['nursing_diagnosis'] if kind == 'daily' else ['s.diagnosis'],
                               'review_status':'unreviewed'})
    for name, rows in [('inputs.jsonl',inputs),('references.jsonl',references)]:
        (OUT/name).write_bytes(b''.join(raw(row) for row in rows))
    manifest = {'version':'synthetic-v1','count':len(inputs),'review_status':'unreviewed',
                'files':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['inputs.jsonl','references.jsonl']}}
    (OUT/'manifest.json').write_bytes(raw(manifest))
    print(f'Generated {len(inputs)} cases in {OUT}')

if __name__ == '__main__': main()
