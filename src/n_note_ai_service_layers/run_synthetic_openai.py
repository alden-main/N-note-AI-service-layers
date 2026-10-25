"""합성 시나리오를 OpenAI 기반 초안 생성 흐름으로 실행하는 진입점이다."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.adapters.synthetic_repository import SyntheticScenarioRepository
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.pressure_injury_validation import PressureInjuryValidator
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry, SchemaReviewService
from n_note_ai_service_layers.settings import load_dotenv


FIXTURE = Path("data_sample/poc_pressure_injury_synthetic/pressure_injury_poc_v1.json")


def run(scenario_id: str, fixture_path: Path = FIXTURE) -> dict:
    """합성 시나리오 하나를 실행해 전사·초안·검증 결과를 직렬화한다.

    입력: ``scenario_id``는 실행할 시나리오이고 ``fixture_path``는 합성 데이터 파일 경로이다.
    출력: 전사문, 초안, 검증 결과와 평가용 기대값을 담은 사전이다.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required; set it in .env or the environment")
    repository = SyntheticScenarioRepository(fixture_path)
    registry = FormSchemaRegistry()
    schema = registry.add(pressure_injury_poc_schema())
    review = SchemaReviewService(registry)
    approved = review.approve_all(schema, "synthetic-reviewer", datetime.now().astimezone())
    schema = registry.activate(approved.schema_id, approved.version)
    observed_at = datetime.fromisoformat(repository.scenario(scenario_id)["scenario_context"]["observation"]["observed_at"])
    query = ClinicalContextQuery(scenario_id, "synthetic-encounter", schema.purpose, observed_at, schema.schema_id, f"synthetic-{scenario_id}", schema.version)
    snapshot = ClinicalContextFlow(PurposeAccessPolicy({schema.purpose}), registry, repository, ClinicalContextAssembler()).build(query)
    payload = LlmPayloadProjector().project(snapshot, repository.transcript(scenario_id), schema)
    draft = OpenAiDraftGenerationAdapter(model=os.getenv("OPENAI_DRAFT_MODEL", "gpt-4o-mini")).generate(payload, schema)
    validation = PressureInjuryValidator().validate(draft, snapshot, schema)
    return {
        "scenario_id": scenario_id,
        "draft_fields": draft.proposed_fields,
        "needs_confirmation": draft.needs_confirmation,
        "validation_status": validation.status,
        "validation_issues": [issue.__dict__ for issue in validation.issues],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic pressure-injury scenario through OpenAI draft generation.")
    parser.add_argument("scenario_id", nargs="?", default="POC-PI-003")
    args = parser.parse_args()
    print(json.dumps(run(args.scenario_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
