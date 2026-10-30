"""API가 공유하는 저장소·레지스트리를 한 곳에서 조립한다.

처음 사용할 때 한 번 만든다. 모듈 import만으로는 환경 변수·양식 파일을 읽지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.document_store import DocumentStore, open_document_store
from n_note_ai_service_layers.adapters.form_definition_loader import FormDefinitionLoader
from n_note_ai_service_layers.api.handoff_store import HandoffStore
from n_note_ai_service_layers.contracts.privacy import NamedEntityRecognizer
from n_note_ai_service_layers.observability import configure_logging, get_logger, log_event
from n_note_ai_service_layers.services.audit import AuditTrail
from n_note_ai_service_layers.services.draft_review import DraftReviewStore
from n_note_ai_service_layers.services.jobs import JobRunner, JobStore
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry, SchemaReviewService


@dataclass(frozen=True)
class AppServices:
    documents: DocumentStore
    audit: AuditTrail
    template_registry: FormSchemaRegistry
    template_review_service: SchemaReviewService
    draft_review_store: DraftReviewStore
    handoff_store: HandoffStore
    jobs: JobStore
    job_runner: JobRunner
    # 로컬 NER(N_NOTE_NER_MODEL)이다. 없으면 규칙 기반 비식별화만 한다.
    ner: NamedEntityRecognizer | None = None


def build_ner() -> NamedEntityRecognizer | None:
    """설정된 로컬 NER 모델을 기동 시점에 올린다. 설정했는데 못 올리면 기동을 중단한다(조용한 보호 해제 방지)."""
    config = settings.get()
    model = config.ner_model
    if not model:
        return None
    from n_note_ai_service_layers.adapters.transformers_ner import TransformersNerRecognizer

    recognizer = TransformersNerRecognizer(model, config.ner_min_score, config.ner_person_labels, config.ner_threads)
    recognizer.load()
    return recognizer


def build_services() -> AppServices:
    """설정을 읽어 저장소와 양식 레지스트리를 만든다. 잘못된 양식 정의는 기동을 중단한다."""
    load_dotenv(".env")
    config = settings.get()
    configure_logging(config.log_level)
    db_path = config.state_db_path
    documents = open_document_store(db_path)
    audit = AuditTrail(documents)
    registry = FormSchemaRegistry(documents)
    # 운영자가 배포한 양식 정의(N_NOTE_FORMS_DIR)를 활성 양식으로 등록한다.
    for form in FormDefinitionLoader(config.forms_dir).load_all(config.enabled_form_ids):
        registry.add_base(form)
    ner = build_ner()
    jobs = JobStore(documents, config.job_timeout_seconds, config.draft_retention_hours)
    log_event(
        get_logger("startup"), "services_ready",
        state_store="sqlite" if db_path else "memory", forms=len(registry.list_versions()),
        timezone=str(config.local_timezone), require_local_stt=config.require_local_stt,
        ner_model=config.ner_model, job_concurrency=config.job_concurrency,
    )
    return AppServices(
        documents,
        audit,
        registry,
        SchemaReviewService(registry),
        DraftReviewStore(documents, audit, config.draft_retention_hours),
        HandoffStore(documents),
        jobs,
        JobRunner(jobs, config.job_concurrency, config.job_max_pending),
        ner,
    )


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    return build_services()
