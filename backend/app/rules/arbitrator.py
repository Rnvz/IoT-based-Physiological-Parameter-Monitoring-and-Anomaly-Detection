from pydantic import BaseModel
from typing import Optional
from app.config import settings
from app.ml.inference import MLInferenceResult

class ArbitrationResult(BaseModel):
    rule_severity: int
    ml_severity: Optional[int] = None
    ml_score: Optional[float] = None
    published_severity: int
    published_label: str
    published_score: Optional[float] = None
    decision_mode: str
    decision_source: str
    is_fallback: bool = False

def arbitrate(
    rule_severity: int,
    rule_label: str,
    ml_result: Optional[MLInferenceResult],
    mode: str = settings.decision_mode
) -> ArbitrationResult:
    """
    Arbitrates between rule-based and ML-based decisions.
    Setting DECISION_MODE=RULE|ML selects which one is published in /result.
    If ML is selected but unavailable (warm-up or error), falls back to rule_severity.
    Always stores both rule_severity and ml_severity.
    """
    mode_upper = mode.upper()
    ml_sev = ml_result.raw_severity if ml_result else None
    ml_score = ml_result.raw_score if ml_result else None

    if mode_upper == "ML":
        if ml_result is not None and ml_result.raw_severity is not None:
            pub_sev = ml_result.debounced_severity if (settings.ml_use_debounce and ml_result.debounced_severity is not None) else ml_result.raw_severity
            return ArbitrationResult(
                rule_severity=rule_severity,
                ml_severity=ml_sev,
                ml_score=ml_score,
                published_severity=pub_sev,
                published_label=ml_result.label,
                published_score=ml_score,
                decision_mode="ML",
                decision_source="ML",
                is_fallback=False
            )
        else:
            is_warmup = bool(ml_result and getattr(ml_result, 'is_warmup', False))
            # Selama fase warm-up (mengumpulkan window 15 sampel awal), jangan bunyikan alarm buzzer (sev 2)
            # dari detak awal transisi sensor. Tahan sebagai NORMAL (severity 0) hingga window stabil.
            pub_sev = 0 if is_warmup else rule_severity
            pub_label = "NORMAL" if is_warmup else rule_label
            source = "WARMUP" if is_warmup else "RULE_FALLBACK"

            print(f"[Arbitrator] ML result unavailable ({getattr(ml_result, 'label', 'None')}, warmup={is_warmup}). Source={source}, severity={pub_sev}.")
            return ArbitrationResult(
                rule_severity=rule_severity,
                ml_severity=ml_sev,
                ml_score=ml_score,
                published_severity=pub_sev,
                published_label=pub_label,
                published_score=None,
                decision_mode="ML",
                decision_source=source,
                is_fallback=True
            )
    else:
        return ArbitrationResult(
            rule_severity=rule_severity,
            ml_severity=ml_sev,
            ml_score=ml_score,
            published_severity=rule_severity,
            published_label=rule_label,
            published_score=ml_score,
            decision_mode="RULE",
            decision_source="RULE",
            is_fallback=False
        )
