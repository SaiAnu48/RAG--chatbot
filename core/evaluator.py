"""Context-bounded extraction and deterministic PA decisioning."""
from __future__ import annotations
import re

from google import genai
from google.genai import types
from core.vector_store import PolicyVectorStore
from models.schemas import Determination, PatientProfile, PolicyAnswer, PolicyChunk, PriorAuthorizationSummary


def _policy_reference_schema() -> types.Schema:
    """Return the Gemini-compatible schema for a policy citation."""
    return types.Schema(type=types.Type.OBJECT, properties={
        "document": types.Schema(type=types.Type.STRING),
        "section_or_page": types.Schema(type=types.Type.STRING),
        "relevant_clause": types.Schema(type=types.Type.STRING),
    }, required=["document", "section_or_page", "relevant_clause"])


def _prior_authorization_schema() -> types.Schema:
    """Return an API-safe schema without Pydantic's unsupported additionalProperties."""
    criterion = types.Schema(type=types.Type.OBJECT, properties={
        "criterion": types.Schema(type=types.Type.STRING),
        "satisfied": types.Schema(type=types.Type.BOOLEAN),
        "clinical_evidence_found": types.Schema(type=types.Type.STRING, nullable=True),
        "deficiency_note": types.Schema(type=types.Type.STRING, nullable=True),
    }, required=["criterion", "satisfied", "clinical_evidence_found", "deficiency_note"])
    prerequisite = types.Schema(type=types.Type.OBJECT, properties={
        "step_therapy_met": types.Schema(type=types.Type.BOOLEAN),
        "waiting_period_observed": types.Schema(type=types.Type.BOOLEAN),
    }, required=["step_therapy_met", "waiting_period_observed"])
    return types.Schema(type=types.Type.OBJECT, properties={
        "determination": types.Schema(type=types.Type.STRING, enum=["APPROVED", "DENIED", "PENDING_ADDITIONAL_INFORMATION"]),
        "policy_reference": _policy_reference_schema(),
        "criteria_checklist": types.Schema(type=types.Type.ARRAY, items=criterion),
        "pre_requisite_status": prerequisite,
        "justification_narrative": types.Schema(type=types.Type.STRING),
        "recommended_next_action": types.Schema(type=types.Type.STRING),
    }, required=["determination", "policy_reference", "criteria_checklist", "pre_requisite_status", "justification_narrative", "recommended_next_action"])


def _policy_answer_schema() -> types.Schema:
    """Return the Gemini-compatible schema for interactive answers."""
    return types.Schema(type=types.Type.OBJECT, properties={
        "answer": types.Schema(type=types.Type.STRING),
        "citations": types.Schema(type=types.Type.ARRAY, items=_policy_reference_schema()),
    }, required=["answer", "citations"])

class PriorAuthorizationEvaluator:
    """Uses policy context and supplied clinical facts without inference beyond either."""
    def __init__(self, store: PolicyVectorStore, model: str = "gemini-3.1-flash-lite", api_key: str | None = None) -> None:
        self._store, self._model, self._client = store, model, genai.Client(api_key=api_key)
    def _retrieve(self, profile: PatientProfile) -> list[PolicyChunk]:
        """Retrieve broad and mandatory risk-focused evidence."""
        query, found = " ".join([profile.requested_treatment, *profile.procedure_codes, *profile.diagnoses]), {}
        for tag in (None, "exclusion", "step_therapy", "waiting_period", "diagnostic_evidence"):
            for chunk in self._store.search(query, required_tag=tag): found[chunk.id] = chunk
        for chunk in self._store.lexical_search(query): found[chunk.id] = chunk
        return list(found.values())
    def evaluate(self, profile: PatientProfile) -> PriorAuthorizationSummary:
        """Extract a schema-validated summary and conservatively enforce its outcome."""
        chunks = self._retrieve(profile)
        if not chunks: raise RuntimeError("No policy evidence retrieved; authorization cannot be evaluated")
        context = "\n\n".join(f"[SOURCE document={c.document}; page={c.section_or_page}; id={c.id}]\n{c.text}" for c in chunks)
        prompt = f"""You are a policy evidence extractor. Use ONLY POLICY CONTEXT and CLINICAL PROFILE. Never invent criteria, citations, diagnoses, treatment history, or approval. Every absent, ambiguous, or uncited criterion must be satisfied=false and its deficiency_note must begin exactly `UNVERIFIED_OR_MISSING:`. Cite a clause verbatim or near-verbatim from context.\nPOLICY CONTEXT:\n{context}\nCLINICAL PROFILE:\n{profile.model_dump_json()}"""
        response = self._client.models.generate_content(model=self._model, contents=prompt, config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=_prior_authorization_schema()))
        return self._conservative_determination(PriorAuthorizationSummary.model_validate_json(response.text), chunks)
    def answer_question(self, question: str, profile: PatientProfile | None = None) -> PolicyAnswer:
        """Answer a policy question strictly from retrieved chunks and optional clinical facts."""
        if not question.strip(): raise ValueError("A question is required")
        found = {chunk.id: chunk for chunk in self._store.search(question, limit=10)}
        for chunk in self._store.lexical_search(question, limit=10): found[chunk.id] = chunk
        chunks = list(found.values())
        if not chunks: raise RuntimeError("No policy evidence retrieved for this question")
        context = "\n\n".join(f"[SOURCE document={c.document}; page={c.section_or_page}; id={c.id}]\n{c.text}" for c in chunks)
        profile_text = profile.model_dump_json() if profile else "No patient profile was provided."
        prompt = f"""You are a healthcare-policy assistant. Answer ONLY from POLICY CONTEXT and the optional CLINICAL PROFILE. Do not infer missing clinical facts, coverage, eligibility, or approval. When asked to list covered treatments or procedures, list every explicitly named covered or conditionally covered service in the retrieved context; say the list is not necessarily exhaustive unless the policy explicitly says so. Do not mark such a list UNVERIFIED_OR_MISSING merely because the policy does not claim completeness. If the direct answer is absent, say exactly `UNVERIFIED_OR_MISSING: the retrieved policy context does not establish this.` Include only citations whose relevant_clause is quoted or near-verbatim from context.\nPOLICY CONTEXT:\n{context}\nCLINICAL PROFILE:\n{profile_text}\nQUESTION:\n{question}"""
        response = self._client.models.generate_content(model=self._model, contents=prompt, config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=_policy_answer_schema()))
        answer = PolicyAnswer.model_validate_json(response.text)
        source_text = " ".join(c.text for c in chunks).lower()
        if any(citation.relevant_clause.lower() not in source_text for citation in answer.citations):
            raise RuntimeError("Model returned an unsupported citation; no answer was emitted")
        return answer
    @staticmethod
    def _conservative_determination(summary: PriorAuthorizationSummary, chunks: list[PolicyChunk]) -> PriorAuthorizationSummary:
        """Prevent approval whenever any evidence or citation is unverified."""
        citation_valid = PriorAuthorizationEvaluator._citation_supported(summary.policy_reference.relevant_clause, chunks)
        all_met = all(x.satisfied and x.clinical_evidence_found for x in summary.criteria_checklist)
        prerequisites = summary.pre_requisite_status.step_therapy_met and summary.pre_requisite_status.waiting_period_observed
        unknown = any("UNVERIFIED_OR_MISSING" in (x.deficiency_note or "") for x in summary.criteria_checklist)
        failed_with_evidence = any(
            not x.satisfied and bool(x.clinical_evidence_found or x.deficiency_note)
            for x in summary.criteria_checklist
        )
        verified_denial = summary.determination is Determination.DENIED and citation_valid and failed_with_evidence
        verified_approval = summary.determination is Determination.APPROVED and citation_valid and not unknown and all_met and prerequisites
        if not (verified_denial or verified_approval):
            summary.determination = Determination.PENDING_ADDITIONAL_INFORMATION
            if not citation_valid: summary.recommended_next_action = "UNVERIFIED_OR_MISSING: retrieve a policy clause supporting the cited reference before determination."
        return summary

    @staticmethod
    def _citation_supported(clause: str, chunks: list[PolicyChunk]) -> bool:
        """Return true when a citation is exact or strongly grounded in retrieved text."""
        normalized_clause = PriorAuthorizationEvaluator._normalize_text(clause)
        normalized_context = PriorAuthorizationEvaluator._normalize_text(" ".join(c.text for c in chunks))
        if not normalized_clause:
            return False
        if normalized_clause in normalized_context:
            return True
        clause_terms = set(re.findall(r"[a-z0-9]{4,}", normalized_clause))
        if not clause_terms:
            return False
        context_terms = set(re.findall(r"[a-z0-9]{4,}", normalized_context))
        overlap = len(clause_terms & context_terms) / len(clause_terms)
        return overlap >= 0.82

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize policy text before deterministic citation comparison."""
        return re.sub(r"\s+", " ", value.lower().replace("...", " ")).strip()
