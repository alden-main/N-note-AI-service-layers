"""욕창 위험도 POC에서 사용하는 결정적 사실 ID와 예방 정책 상수이다."""

BRADEN_FACT_IDS = (
    "braden.sensory_perception", "braden.moisture", "braden.activity",
    "braden.mobility", "braden.nutrition", "braden.friction_shear",
)

RISK_LEVELS = ((19, 23, "NORMAL"), (10, 18, "RISK"), (6, 9, "HIGH_RISK"))
INTERVENTIONS = {
    "NORMAL": (),
    "RISK": ("REPOSITIONING", "SUPPORT_SURFACE", "MOISTURE_MANAGEMENT", "SKIN_OBSERVATION"),
    "HIGH_RISK": ("REPOSITIONING", "SUPPORT_SURFACE", "MOISTURE_MANAGEMENT", "SKIN_OBSERVATION", "PREVENTIVE_ADHESIVE", "NUTRITION_MANAGEMENT", "PATIENT_EDUCATION"),
}
