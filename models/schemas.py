"""Pydantic contracts used at every PA-assistant boundary."""
from __future__ import annotations
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

class StrictModel(BaseModel):
    """Reject unexpected data at trust boundaries."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class Determination(str, Enum):
    """Permitted authorization outcomes."""
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING_ADDITIONAL_INFORMATION = "PENDING_ADDITIONAL_INFORMATION"

class PolicyReference(StrictModel):
    """Citation for a policy clause."""
    document: Annotated[str, Field(min_length=1)]
    section_or_page: str | int
    relevant_clause: Annotated[str, Field(min_length=1)]

class CriterionCheck(StrictModel):
    """Evidence-backed result for one explicit policy criterion."""
    criterion: Annotated[str, Field(min_length=1)]
    satisfied: bool
    clinical_evidence_found: str | None = None
    deficiency_note: str | None = None
    @field_validator("deficiency_note")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        """Reject empty deficiency notes."""
        if value is not None and not value.strip(): raise ValueError("deficiency_note cannot be blank")
        return value

class PrerequisiteStatus(StrictModel):
    """Status of high-risk prerequisites."""
    step_therapy_met: bool
    waiting_period_observed: bool

class PriorAuthorizationSummary(StrictModel):
    """Strict externally consumable PA payload."""
    determination: Determination
    policy_reference: PolicyReference
    criteria_checklist: list[CriterionCheck] = Field(min_length=1)
    pre_requisite_status: PrerequisiteStatus
    justification_narrative: Annotated[str, Field(min_length=1)]
    recommended_next_action: Annotated[str, Field(min_length=1)]

class PolicyAnswer(StrictModel):
    """A source-bounded answer for the interactive policy assistant."""
    answer: Annotated[str, Field(min_length=1)]
    citations: list[PolicyReference] = Field(min_length=1)

class PatientProfile(StrictModel):
    """Structured clinical source facts."""
    patient_id: Annotated[str, Field(min_length=1)]
    requested_treatment: Annotated[str, Field(min_length=1)]
    procedure_codes: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    clinical_facts: dict[str, Any] = Field(default_factory=dict)
    supporting_documents: list[str] = Field(default_factory=list)

class PolicyChunk(StrictModel):
    """A source-addressable policy fragment."""
    id: str
    text: str
    document: str
    section_or_page: str | int
    chunk_index: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)
    def chroma_metadata(self) -> dict[str, str | int]:
        """Return Chroma-compatible metadata."""
        return {"document": self.document, "section_or_page": str(self.section_or_page), "chunk_index": self.chunk_index, "tags": ",".join(self.tags)}
