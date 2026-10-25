"""합성 POC 시나리오 파일에서 임상 원천 레코드를 읽는 저장소 어댑터이다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery, SourceRecord
from n_note_ai_service_layers.policies.pressure_injury import BRADEN_FACT_IDS


class SyntheticScenarioRepository:
    """합성 시나리오에서 실행용 임상 사실만 제공하는 저장소이다.

    입력: 합성 시나리오 JSON 파일 경로이다.
    출력: 조회 시 기대값 JSON을 제외한 ``SourceRecord`` 목록이다.
    """
    def __init__(self, fixture_path: Path) -> None:
        self.data = json.loads(fixture_path.read_text())

    def scenario(self, scenario_id: str) -> dict:
        return next(item for item in self.data["scenarios"] if item["scenario_id"] == scenario_id)

    def transcript(self, scenario_id: str) -> str:
        return self.scenario(scenario_id)["input"]["transcript"]

    def expected_for_evaluation(self, scenario_id: str) -> dict:
        """평가 전용 기대 결과를 반환한다. 이 값은 LLM 입력으로 전달하면 안 된다."""
        return self.scenario(scenario_id)["expected"]

    def fetch(self, query: ClinicalContextQuery, requested_fact_ids: set[str]) -> list[SourceRecord]:
        """요청한 사실 ID에 해당하는 합성 원천 레코드를 조회한다.

        입력: 환자 ID를 시나리오 ID로 사용하는 ``query``와 필요한 사실 ID 집합이다.
        출력: 요청된 Braden 사실의 ``SourceRecord`` 목록이며, 충돌 시나리오는 중복 관찰값을 포함한다.
        """
        scenario = self.scenario(query.patient_id)
        observed_at = datetime.fromisoformat(scenario["scenario_context"]["observation"]["observed_at"])
        braden = scenario["expected"]["fields"]["braden"]
        records: list[SourceRecord] = []
        for short_name in BRADEN_FACT_IDS:
            key = short_name.removeprefix("braden.")
            if short_name not in requested_fact_ids or braden.get(key) is None:
                continue
            records.append(SourceRecord(short_name, braden[key], observed_at, "synthetic-poc"))
        if scenario["scenario_id"] == "POC-PI-005" and "braden.moisture" in requested_fact_ids:
            records.extend((
                SourceRecord("braden.moisture", 4, observed_at, "synthetic-poc"),
                SourceRecord("braden.moisture", 1, observed_at, "synthetic-poc"),
            ))
        return records
