"""File uploads — save what the user drops into the chat and tell the agent about it.

The browser sends each file as a raw request body (POST /upload?filename=...),
so no multipart dependency is needed. Files land in <workspace>/uploads, which
means the agent can reach them with its normal workspace tools. Readable types
(text, code, data, .docx, and .pdf when pypdf is installed) are also inlined
into the message so the agent can answer without a tool round-trip; everything
else is saved and handed over by path.
"""

import re
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .tools.workspace_tool import WORKSPACE_ROOT

UPLOAD_DIR = WORKSPACE_ROOT / "uploads"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Inlined text is capped so one big file can't swamp the context window; the
# agent can read_file the rest from the saved path.
MAX_INLINE_CHARS = 20_000

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".xml", ".html", ".htm",
    ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1", ".sql",
    ".tex",
}
# Saved and passed by path only — the agent opens them with whatever tool fits.
BINARY_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".odt", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".bmp", ".svg", ".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".mov",
    ".webm", ".zip",
}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | BINARY_EXTENSIONS

router = APIRouter()


def _clean_name(filename: str) -> str:
    """Strip any directory part and anything unsafe, keeping the extension."""
    name = re.sub(r"[^\w.\- ]", "_", Path(filename.replace("\\", "/")).name).strip(" .")
    return name or "upload"


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # optional; ImportError is handled by the caller

    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _extract_text(path: Path) -> str | None:
    """The file's text, or None when it isn't something we can read here."""
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8", errors="replace")
        if ext == ".docx":
            return _read_docx(path)
        if ext == ".pdf":
            return _read_pdf(path)
    except Exception as e:  # missing pypdf, corrupt file, ...
        print(f"[uploads] Could not extract text from {path.name}: {e!r}")
    return None


@router.post("/upload")
async def upload(request: Request, filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"'{ext or filename}' files are not supported.")
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")

    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # The timestamp prefix keeps two uploads of "notes.txt" from overwriting
    # each other.
    saved = f"{int(time.time())}-{_clean_name(filename)}"
    (UPLOAD_DIR / saved).write_bytes(data)
    return {"name": saved, "size": len(data)}


def attach(message: str, files: list[str]) -> str:
    """Return `message` with the uploaded `files` described underneath it."""
    parts = []
    for name in files:
        path = (UPLOAD_DIR / name).resolve()
        # Names come back from the client, so confine them to the upload dir.
        if not path.is_relative_to(UPLOAD_DIR.resolve()) or not path.is_file():
            continue
        rel = f"uploads/{path.name}"
        text = _extract_text(path)
        if text is None:
            parts.append(f"[Attached file: {rel} — not inlined; open it from the workspace if needed.]")
            continue
        clipped = text[:MAX_INLINE_CHARS]
        note = "" if len(text) <= MAX_INLINE_CHARS else f"\n[Truncated at {MAX_INLINE_CHARS} characters — read {rel} for the rest.]"
        parts.append(f"[Attached file: {rel}]\n{clipped}{note}")
    if not parts:
        return message
    return f"{message}\n\n" + "\n\n".join(parts)
