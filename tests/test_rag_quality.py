"""Retrieval and optional live-Gemini quality tests for the synthetic PA policy.

Run offline retrieval checks:
    python -m unittest tests.test_rag_quality

Run live answer-quality checks (uses Gemini API quota):
    $env:RUN_LIVE_RAG_TESTS = "1"
    python -m unittest tests.test_rag_quality
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from pathlib import Path

from core.evaluator import PriorAuthorizationEvaluator
from core.vector_store import PolicyVectorStore
from models.schemas import CriterionCheck, Determination, PatientProfile, PolicyReference, PrerequisiteStatus, PriorAuthorizationSummary

ROOT = Path(__file__).resolve().parents[1]
DATABASE = Path(os.getenv("POLICY_DB", ROOT / ".policy_chroma"))


@dataclass(frozen=True)
class QuestionCase:
    """Expected grounded phrases for one policy Q&A evaluation."""

    question: str
    expected_phrases: tuple[str, ...]


QUESTION_CASES = (
    QuestionCase("Is a Lumbar Spine MRI covered under this policy?", ("conditionally covered", "lumbar")),
    QuestionCase("Does the policy cover cosmetic procedures performed only for appearance improvement?", ("non-covered", "cosmetic")),
    QuestionCase("How long must conservative treatment be completed before a Lumbar Spine MRI can be authorized for persistent back pain?", ("6", "week")),
    QuestionCase("Can a patient probably be considered to have completed physical therapy when no documents mention it?", ("not", "document")),
)


@dataclass(frozen=True)
class AuthorizationCase:
    """Profile and expected PA outcome for a representative policy scenario."""

    name: str
    profile: PatientProfile
    expected: Determination


AUTHORIZATION_CASES = (
    AuthorizationCase(
        "acute_mri_denial",
        PatientProfile(patient_id="test-acute-mri", requested_treatment="Lumbar Spine MRI", procedure_codes=["72148"], diagnoses=["Low back pain"], clinical_facts={"symptom_duration": "3 weeks", "neurologic_deficits": "none", "physical_therapy": "not completed"}),
        Determination.DENIED,
    ),
    AuthorizationCase(
        "red_flag_mri_approval",
        PatientProfile(patient_id="test-red-flag", requested_treatment="Lumbar Spine MRI", procedure_codes=["72148"], clinical_facts={"red_flag_pathology": "suspected cauda equina syndrome", "neurologic_findings": "progressive leg weakness", "symptom_duration": "2 days"}),
        Determination.APPROVED,
    ),
    AuthorizationCase(
        "adalimumab_missing_tb",
        PatientProfile(patient_id="test-ra-tb", requested_treatment="Adalimumab", procedure_codes=["J0135"], diagnoses=["moderate rheumatoid arthritis"], clinical_facts={"methotrexate": "15 mg weekly for 16 weeks without adequate response", "hepatitis_b_screening": "available", "tb_screening": "not submitted"}),
        Determination.PENDING_ADDITIONAL_INFORMATION,
    ),
    AuthorizationCase(
        "knee_infection_denial",
        PatientProfile(patient_id="test-knee-infection", requested_treatment="Total Knee Arthroplasty", procedure_codes=["27447"], diagnoses=["Kellgren-Lawrence Grade 4 osteoarthritis"], clinical_facts={"radiographs": "Grade 4, dated 4 months ago", "functional_impairment": "severe", "conservative_management": "5 months", "modalities": "12 PT sessions and corticosteroid injection", "active_infection": "active local knee joint infection"}),
        Determination.DENIED,
    ),
)


class RetrievalQualityTests(unittest.TestCase):
    """Fast, local tests proving policy sections can be found without Gemini."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = PolicyVectorStore(DATABASE)
        if cls.store._collection.count() == 0:
            raise RuntimeError(f"Policy index is empty at {DATABASE}. Run: python main.py index healthcare_prior_authorization_testing_policy.pdf")

    def test_covered_procedures_heading_is_retrieved(self) -> None:
        """Ensure lexical fallback retains the section semantic search previously missed."""
        chunks = self.store.lexical_search("list covered treatments and procedures", limit=10)
        text = " ".join(chunk.text for chunk in chunks).lower()
        self.assertIn("lumbar spine mri", text)
        self.assertIn("adalimumab", text)
        self.assertIn("total knee arthroplasty", text)

    def test_exclusion_retrieval_finds_cosmetic_services(self) -> None:
        """Ensure a blanket exclusion is recoverable from the local index."""
        chunks = self.store.lexical_search("cosmetic procedures appearance improvement", limit=5)
        self.assertIn("non-covered", " ".join(chunk.text for chunk in chunks).lower())

    def test_denial_guard_allows_explicit_unmet_policy_criteria(self) -> None:
        """Ensure failed criteria do not become pending when policy support is present."""
        evaluator = PriorAuthorizationEvaluator(self.store, api_key="unused")
        chunks = evaluator._retrieve(
            PatientProfile(
                patient_id="test-acute-mri",
                requested_treatment="Lumbar Spine MRI",
                procedure_codes=["72148"],
                diagnoses=["Low back pain"],
                clinical_facts={"symptom_duration": "3 weeks", "physical_therapy": "not completed"},
            )
        )
        summary = PriorAuthorizationSummary(
            determination=Determination.DENIED,
            policy_reference=PolicyReference(
                document="healthcare_prior_authorization_testing_policy.pdf",
                section_or_page="Section 3.1",
                relevant_clause="Lumbar Spine MRI is conditionally covered when ANY of the following criteria is met: Conservative Therapy Failure: Persistent lower back pain lasting at least 6 consecutive weeks.",
            ),
            criteria_checklist=[
                CriterionCheck(
                    criterion="Persistent lower back pain lasting at least 6 consecutive weeks",
                    satisfied=False,
                    clinical_evidence_found="Symptom duration is 3 weeks",
                    deficiency_note="UNVERIFIED_OR_MISSING: Pain duration is less than the 6-week requirement.",
                )
            ],
            pre_requisite_status=PrerequisiteStatus(step_therapy_met=False, waiting_period_observed=False),
            justification_narrative="The policy duration criterion is not met.",
            recommended_next_action="Deny as criteria are not met.",
        )

        guarded = PriorAuthorizationEvaluator._conservative_determination(summary, chunks)

        self.assertEqual(Determination.DENIED, guarded.determination)


@unittest.skipUnless(os.getenv("RUN_LIVE_RAG_TESTS") == "1", "Set RUN_LIVE_RAG_TESTS=1 to run billable Gemini quality tests.")
class LiveGeminiQualityTests(unittest.TestCase):
    """End-to-end policy answer and PA determination evaluations using Gemini."""

    @classmethod
    def setUpClass(cls) -> None:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for live RAG tests")
        cls.evaluator = PriorAuthorizationEvaluator(PolicyVectorStore(DATABASE), api_key=os.environ["GOOGLE_API_KEY"])

    def test_policy_questions_are_grounded(self) -> None:
        """Require answer phrases and at least one valid source citation per question."""
        for case in QUESTION_CASES:
            with self.subTest(question=case.question):
                answer = self.evaluator.answer_question(case.question)
                rendered = answer.answer.lower()
                self.assertFalse(rendered.startswith("unverified_or_missing"), answer.answer)
                for phrase in case.expected_phrases:
                    self.assertIn(phrase, rendered, answer.answer)
                self.assertTrue(answer.citations)

    def test_authorization_determinations(self) -> None:
        """Compare representative authorization scenarios with expected policy outcomes."""
        for case in AUTHORIZATION_CASES:
            with self.subTest(case=case.name):
                summary = self.evaluator.evaluate(case.profile)
                self.assertEqual(case.expected, summary.determination, summary.model_dump_json(indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
