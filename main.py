"""CLI entry point for policy indexing, chat, and deterministic PA evaluation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.evaluator import PriorAuthorizationEvaluator
from core.loader import ingest_policy
from core.vector_store import PolicyVectorStore
from models.schemas import PatientProfile


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Deterministic healthcare PA assistant")
    parser.add_argument("--db", default=".policy_chroma")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Ingest a PDF, TXT, or Markdown policy")
    index.add_argument("policy", type=Path)

    evaluate = sub.add_parser("evaluate", help="Evaluate one patient-profile JSON file")
    evaluate.add_argument("profile", type=Path)

    batch = sub.add_parser("batch", help="Evaluate each JSON profile in a directory")
    batch.add_argument("profiles", type=Path)

    chat = sub.add_parser("chat", help="Start an interactive cited policy Q&A session")
    chat.add_argument("--profile", type=Path, help="Optional patient-profile JSON for clinical-context questions")
    return parser


def _timestamp() -> str:
    """Return a filename-safe UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    """Persist JSON output with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_api_key() -> str:
    """Return the configured Gemini API key or fail with a direct setup message."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in .env or in the current terminal environment.")
    return api_key


def _chat(evaluator: PriorAuthorizationEvaluator, profile: PatientProfile | None, output_dir: Path) -> None:
    """Run chat and persist the entire question-answer session."""
    session_path = output_dir / "chat_sessions" / f"chat_{_timestamp()}.json"
    session: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "profile": profile.model_dump() if profile else None,
        "messages": [],
    }
    _write_json(session_path, session)
    print("Policy assistant ready. Ask a question; type exit or quit to close.")
    print(f"Chat log: {session_path}")

    try:
        while True:
            try:
                question = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if question.lower() in {"exit", "quit"}:
                return
            if not question:
                continue
            try:
                answer = evaluator.answer_question(question, profile)
                payload: dict[str, Any] = answer.model_dump(mode="json")
                print("\nAssistant> " + answer.model_dump_json(indent=2))
            except Exception as error:
                payload = {"answer": f"UNVERIFIED_OR_MISSING: {error}", "citations": []}
                print(f"\nAssistant> {payload['answer']}")
            session["messages"].append({"question": question, "answer": payload})
            _write_json(session_path, session)
    finally:
        session["ended_at"] = datetime.now(UTC).isoformat()
        _write_json(session_path, session)


def _batch(evaluator: PriorAuthorizationEvaluator, profiles_dir: Path, output_dir: Path) -> None:
    """Evaluate every profile JSON and persist a batch result file."""
    results = []
    for path in sorted(profiles_dir.glob("*.json")):
        profile = PatientProfile.model_validate_json(path.read_text(encoding="utf-8"))
        summary = evaluator.evaluate(profile)
        results.append({"profile_file": str(path), "patient_id": profile.patient_id, "result": summary.model_dump(mode="json")})

    payload = {"created_at": datetime.now(UTC).isoformat(), "profiles_dir": str(profiles_dir), "results": results}
    output_path = output_dir / "batch_results" / f"batch_{_timestamp()}.json"
    latest_path = output_dir / "batch_results" / "latest.json"
    _write_json(output_path, payload)
    _write_json(latest_path, payload)
    print(json.dumps({"evaluated_profiles": len(results), "output": str(output_path), "latest": str(latest_path)}))


def main() -> None:
    """Run index, chat, evaluate, or batch workflows."""
    load_dotenv()
    args = _parser().parse_args()
    store = PolicyVectorStore(args.db)

    if args.command == "index":
        chunks = ingest_policy(args.policy)
        store.index(chunks)
        print(json.dumps({"indexed_chunks": len(chunks), "document": args.policy.name}))
        return

    evaluator = PriorAuthorizationEvaluator(store, api_key=_require_api_key())

    if args.command == "chat":
        profile = PatientProfile.model_validate_json(args.profile.read_text(encoding="utf-8")) if args.profile else None
        _chat(evaluator, profile, args.output_dir)
        return

    if args.command == "batch":
        _batch(evaluator, args.profiles, args.output_dir)
        return

    profile = PatientProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
    print(evaluator.evaluate(profile).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
