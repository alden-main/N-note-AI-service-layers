from __future__ import annotations

from datetime import datetime
from pathlib import Path
import unittest

from n_note_ai_service_layers.adapters.synthetic_repository import SyntheticScenarioRepository
from n_note_ai_service_layers.adapters.clinical_normalizer import CodeMappingNormalizer
from n_note_ai_service_layers.adapters.schema_extraction import SchemaSourceDocument, TestSchemaExtractor
from n_note_ai_service_layers.contracts.clinical_context import SourceRecord
from n_note_ai_service_layers.contracts.form_schema import FormSection, SchemaCandidate
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, SchemaStatus
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.services.access_policy import AccessDenied, PurposeAccessPolicy
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.draft_handoff import DraftHandoffProjector
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.pressure_injury_validation import PressureInjuryValidator
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry, SchemaReviewService


FIXTURE = Path(__file__).parents[1] / "data_sample/poc_pressure_injury_synthetic/pressure_injury_poc_v1.json"


class PocFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = SyntheticScenarioRepository(FIXTURE)
        self.registry = FormSchemaRegistry()
        schema = pressure_injury_poc_schema()
        self.registry.add(schema)
        service = SchemaReviewService(self.registry)
        approved = service.approve_all(schema, "reviewer-1", datetime.fromisoformat("2026-10-01T00:00:00+09:00"))
        self.schema = self.registry.activate(approved.schema_id, approved.version)
        self.flow = ClinicalContextFlow(PurposeAccessPolicy({self.schema.purpose}), self.registry, self.repo, ClinicalContextAssembler())

    def query(self, scenario_id: str) -> ClinicalContextQuery:
        observed = self.repo.scenario(scenario_id)["scenario_context"]["observation"]["observed_at"]
        return ClinicalContextQuery(scenario_id, "synthetic-encounter", self.schema.purpose, datetime.fromisoformat(observed), self.schema.schema_id, "trace-1", self.schema.version)

    def test_all_scenarios_have_expected_validation_status_and_payload_is_minimal(self) -> None:
        for scenario_id in [f"POC-PI-00{i}" for i in range(1, 6)]:
            with self.subTest(scenario_id=scenario_id):
                snapshot = self.flow.build(self.query(scenario_id))
                payload = LlmPayloadProjector().project(snapshot, self.repo.transcript(scenario_id), self.schema)
                encoded = repr(payload)
                self.assertNotIn("expected", encoded)
                self.assertNotIn("source_reference", encoded)
                result = PressureInjuryValidator().validate(DraftResult({"risk.level": "NORMAL"}), snapshot, self.schema)
                self.assertEqual(result.status, self.repo.expected_for_evaluation(scenario_id)["validation"]["status"])
                if scenario_id in ("POC-PI-004", "POC-PI-005"):
                    self.assertTrue(payload["needs_confirmation"])

    def test_deterministic_risk_overrides_model_proposal_and_handoff_is_allowlisted(self) -> None:
        result = PressureInjuryValidator().validate(DraftResult({"risk.level": "HIGH_RISK", "unapproved": "no"}), self.flow.build(self.query("POC-PI-002")), self.schema)
        self.assertEqual(result.fields["risk.level"], "RISK")
        handoff = DraftHandoffProjector().project(result, self.schema, "reviewed")
        self.assertNotIn("unapproved", handoff.fields)
        self.assertEqual(handoff.schema_version, self.schema.version)
        self.assertNotIn("emr_write_status", handoff.__dict__)

    def test_access_happens_before_repository_read(self) -> None:
        flow = ClinicalContextFlow(PurposeAccessPolicy(set()), self.registry, self.repo, ClinicalContextAssembler())
        with self.assertRaises(AccessDenied):
            flow.build(self.query("POC-PI-001"))

    def test_active_schema_is_immutable_and_version_switch_is_single_active(self) -> None:
        review = SchemaReviewService(self.registry)
        with self.assertRaises(ValueError):
            review.revise_field(self.schema, "braden.nutrition", FormField("braden.nutrition", "integer", True, fact_id="braden.nutrition"), "x", "r", datetime.now().astimezone())
        newer = pressure_injury_poc_schema(version="pressure-injury-poc/v2")
        self.registry.add(newer)
        newer = review.approve_all(newer, "reviewer-1", datetime.now().astimezone())
        self.registry.activate(newer.schema_id, newer.version)
        self.assertEqual(self.registry.resolve_active(newer.schema_id, newer.purpose).version, newer.version)
        self.assertEqual(self.registry.get(self.schema.schema_id, self.schema.version).status, SchemaStatus.APPROVED)

    def test_unmapped_code_is_unavailable_and_candidate_is_never_active(self) -> None:
        record = SourceRecord("braden.moisture", "UNKNOWN", datetime.now().astimezone(), "hospital-a")
        fact = CodeMappingNormalizer({"braden.moisture": {"DRY": 4}}).normalize([record])[0]
        self.assertEqual(fact.availability.value, "unavailable")
        candidate = SchemaCandidate("test.pdf", datetime.now().astimezone(), "test", (FormSection("x", ()),))
        returned = TestSchemaExtractor(candidate).extract(SchemaSourceDocument("test.pdf"))
        self.assertEqual(returned.status, SchemaStatus.DRAFT)
        with self.assertRaises(ValueError):
            SchemaCandidate("test.pdf", datetime.now().astimezone(), "test", status=SchemaStatus.ACTIVE)


if __name__ == "__main__":
    unittest.main()
