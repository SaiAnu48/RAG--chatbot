# Healthcare Insurance Prior Authorization Assistant

Deterministic retrieval-augmented generation (RAG) for health-insurance prior-authorization summaries. The application uses the **Google Gemini API through `google-genai`**. It does not use Ollama, local models, or a local model server.

> This software assists with policy review. It does not make clinical decisions or replace payer review, licensed clinical judgment, or legal advice.

## Technology stack

| Area | Technology | Purpose |
| --- | --- | --- |
| Runtime | Python 3.11+ | Application runtime and CLI |
| Data contracts | Pydantic v2 | Strict patient, citation, checklist, and output validation |
| Document parsing | PyPDF | Extract text from policy-manual PDFs |
| Vector database | ChromaDB | Persistent local policy index using cosine distance |
| LLM | Google Gemini API / `google-genai` | `gemini-2.5-flash`, temperature-zero, JSON-schema-constrained evidence extraction |
| Interface | `argparse` | Index, single evaluation, and batch evaluation commands |

## RAG pipeline

| Stage | Component | Stack / behavior |
| --- | --- | --- |
| 1. Ingest | `core/loader.py` | PyPDF or UTF-8 text/Markdown extraction |
| 2. Chunk | `core/loader.py` | Deterministic 1,200-character sliding windows with 200-character overlap |
| 3. Tag | `core/loader.py` | Regex metadata tags: covered treatment, exclusions, step therapy, waiting period, diagnostic evidence |
| 4. Index | `core/vector_store.py` | Chroma persistent collection; cosine semantic search |
| 5. Retrieve | `core/evaluator.py` | Broad treatment/code/diagnosis retrieval plus focused retrieval for high-risk policy topics |
| 6. Extract | Google Gemini via `google-genai` | Temperature `0`, supplied policy context only, Pydantic JSON schema response |
| 7. Guard | `core/evaluator.py` | Citation presence and criteria/prerequisite checks prevent unsupported approval |
| 8. Output | `models/schemas.py` | Validated determination, citation, checklist, prerequisites, narrative, and next action |

## Local setup

1. Install Python 3.11 or newer.

2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Create a Gemini API key and configure it in `.env` or as a normal environment variable. This project uses `python-dotenv` to load `.env` locally. It does not use Ollama or any local model server. Do not place real keys in source code, patient-profile JSON, README examples, or Git.

```powershell
# Recommended local setup
Copy-Item .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_real_key

# Current terminal only
$env:GOOGLE_API_KEY = "your_google_gemini_api_key"

# Persist for future Windows terminals; restart the terminal afterwards
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your_google_gemini_api_key", "User")
```

`GOOGLE_API_KEY` is the project's expected variable name. `main.py` calls `load_dotenv()` first, then reads `os.environ.get("GOOGLE_API_KEY")`.

To confirm the active terminal can see the key:

```powershell
python -c "import os; print(bool(os.environ.get('GOOGLE_API_KEY')))"
```

The command should print `True`. If it prints `False`, set `$env:GOOGLE_API_KEY` in the same PowerShell window before running `chat`, `evaluate`, `batch`, or live tests.

## Obtain and secure the Google API key

1. Sign in to [Google AI Studio](https://aistudio.google.com/).
2. Open [Create or view a Gemini API key](https://aistudio.google.com/apikey), choose or import the intended Google Cloud project, and create an API key.
3. Use a newly created authorization key where available, then set it in `GOOGLE_API_KEY` as shown above.
4. Restrict the key to Gemini API use and, for deployed services, restrict permitted origins or IP addresses.
5. Store production credentials in [Google Cloud Secret Manager](https://cloud.google.com/secret-manager), not in environment files committed to a repository.

Useful official references: [Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key), [Google AI Studio](https://aistudio.google.com/), and [Google Cloud credentials](https://console.cloud.google.com/apis/credentials).

## Run the application

Index a policy manual (PDF, TXT, or Markdown):

```powershell
python main.py index .\policies\orthopedic_policy.pdf
```

Evaluate one structured clinical profile:

```powershell
python main.py evaluate .\profiles\patient_001.json
```

Evaluate every `*.json` profile in a directory. A timestamped result file is written to `outputs/batch_results/`, and `outputs/batch_results/latest.json` is refreshed.

```powershell
python main.py batch .\profiles
```

The `profiles/` directory contains 10 synthetic test patients derived from the bundled policy manual's evaluation cases.

Start the interactive policy assistant:

```powershell
python main.py chat
```

Ask normal questions such as `What is required before approving lumbar MRI?` and type `exit` to close it. Each chat creates a full session transcript under `outputs/chat_sessions/` containing every question, complete answer, citations, start time, and end time. To let the assistant refer to one patient's supplied facts, add a profile:

```powershell
python main.py chat --profile .\profiles\patient_001.json
```

Use `--db` to change the persistent local Chroma database directory:

```powershell
python main.py --db .\data\policy_chroma index .\policies\policy.pdf
```

Use `--output-dir` to change where batch results and chat transcripts are saved:

```powershell
python main.py --output-dir .\runs chat
```

## Patient-profile input

Patient JSON is for structured PA evaluation. Each file represents one request. The required fields are `patient_id`, `requested_treatment`, `procedure_codes`, `diagnoses`, `clinical_facts`, and `supporting_documents`.

```json
{
  "patient_id": "patient-001",
  "requested_treatment": "Lumbar spine MRI",
  "procedure_codes": ["72148"],
  "diagnoses": ["Low back pain"],
  "clinical_facts": {
    "conservative_therapy": "Physical therapy completed for 6 weeks",
    "neurologic_findings": "Persistent radicular symptoms"
  },
  "supporting_documents": ["PT discharge summary dated 2026-08-01"]
}
```

Run one JSON profile:

```powershell
python main.py evaluate .\profiles\patient_002_mri_after_conservative_therapy.json
```

Run all sample JSON profiles:

```powershell
python main.py batch .\profiles
```

## Deterministic safety behavior

- The model receives only retrieved policy chunks and the supplied patient profile.
- Missing, ambiguous, or uncited clinical criteria must be marked `UNVERIFIED_OR_MISSING`.
- An approval remains `APPROVED` only if its citation is found in retrieved policy text, all checklist items contain evidence, and both prerequisite flags are true.
- A cited and verified denial can remain `DENIED`; all other unsupported outcomes become `PENDING_ADDITIONAL_INFORMATION`.
- Preserve the generated output and cited policy document as part of the PA audit trail.

## Quality testing

The repository includes [tests/test_rag_quality.py](tests/test_rag_quality.py), based on the synthetic policy's coverage, exclusion, duration, missing-evidence, and authorization scenarios.

Run the fast local retrieval checks first. They use the indexed policy only and make no Gemini API calls:

```powershell
python -m unittest tests.test_rag_quality
```

Run the optional end-to-end Gemini quality checks after setting `GOOGLE_API_KEY`. These consume Gemini API quota and compare answers and determinations to expected policy behavior:

```powershell
$env:RUN_LIVE_RAG_TESTS = "1"
python -m unittest tests.test_rag_quality
```

Optionally point tests at another Chroma database with `POLICY_DB`.
