"""Policy-manual ingestion with deterministic chunking and tagging."""
from __future__ import annotations
import hashlib, re
from pathlib import Path
from pypdf import PdfReader
from models.schemas import PolicyChunk

_TAG_PATTERNS = {"covered_treatment": re.compile(r"\b(covered|coverage|prior authorization|precertification|cpt|hcpcs)\b", re.I), "exclusion": re.compile(r"\b(non-covered|not covered|excluded|exclusion|shall not cover)\b", re.I), "step_therapy": re.compile(r"\b(step therapy|fail[- ]first|conservative therapy|failed .* therapy)\b", re.I), "waiting_period": re.compile(r"\b(waiting period|days? (?:before|after)|weeks? (?:before|after))\b", re.I), "diagnostic_evidence": re.compile(r"\b(mri|imaging|laboratory|lab result|diagnostic|within \d+ days?)\b", re.I)}

def read_policy(path: Path) -> list[tuple[str | int, str]]:
    """Extract pages from UTF-8 text/Markdown or a PDF manual."""
    if path.suffix.lower() == ".pdf": return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
    return [("text", path.read_text(encoding="utf-8"))]

def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Use stable word-boundary sliding windows."""
    if size <= overlap: raise ValueError("chunk_size must exceed chunk_overlap")
    text, result, start = re.sub(r"\s+", " ", text).strip(), [], 0
    while start < len(text):
        end = min(start + size, len(text)); boundary = text.rfind(" ", start, end)
        if end < len(text) and boundary > start: end = boundary
        if text[start:end].strip(): result.append(text[start:end].strip())
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return result

def ingest_policy(path: str | Path, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[PolicyChunk]:
    """Create source-citable, metadata-tagged chunks from a policy document."""
    source, chunks, ordinal = Path(path), [], 0
    for locator, page_text in read_policy(source):
        for text in _windows(page_text, chunk_size, chunk_overlap):
            tags = [name for name, pattern in _TAG_PATTERNS.items() if pattern.search(text)]
            digest = hashlib.sha256(f"{source.name}|{locator}|{ordinal}|{text}".encode()).hexdigest()[:24]
            chunks.append(PolicyChunk(id=digest, text=text, document=source.name, section_or_page=locator, chunk_index=ordinal, tags=tags)); ordinal += 1
    if not chunks: raise ValueError(f"No extractable text found in {source}")
    return chunks
