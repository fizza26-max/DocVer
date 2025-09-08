
import io
import re
import json
import math
import time
from typing import List, Dict, Tuple

import streamlit as st
import pdfplumber
from docx import Document as DocxDocument
from transformers import pipeline
from pydantic import BaseModel

st.set_page_config(page_title="Document Verifier Agent", page_icon="🛡️", layout="centered")

# -----------------------------
# Caching model loaders
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_zero_shot_classifier():
    # Lightweight MNLI model for zero-shot classification
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

@st.cache_resource(show_spinner=True)
def load_text2text():
    # Chat-like answers for follow-ups (grounded in extracted text we pass as context)
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)


# -----------------------------
# Utility: file reading
# -----------------------------
import fitz  # PyMuPDF
import easyocr
from PIL import Image

# load once
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def read_pdf(file_bytes: bytes) -> str:
    text_parts = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    reader = load_ocr()

    for page in pdf:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
        else:
            # OCR fallback
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_ocr = " ".join(reader.readtext(np.array(img), detail=0))
            text_parts.append(text_ocr)
    return "\n".join(text_parts).strip()


def read_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = DocxDocument(f)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def read_txt(file_bytes: bytes, encoding_guess: str = "utf-8") -> str:
    try:
        return file_bytes.decode(encoding_guess, errors="ignore").strip()
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore").strip()

def extract_text_from_upload(upload) -> Tuple[str, str]:
    """
    Returns (text, ext) where ext in {'pdf','docx','txt'}
    """
    name = upload.name.lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return read_pdf(data), "pdf"
    elif name.endswith(".docx"):
        return read_docx(data), "docx"
    elif name.endswith(".txt"):
        return read_txt(data), "txt"
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# -----------------------------
# Heuristics for authenticity signals
# -----------------------------
SENSATIONAL_WORDS = {
    "shocking","unbelievable","you won't believe","miracle","cure","instant","secret",
    "exposed","banned","leaked","guaranteed","100%","BREAKING","!!!","act now"
}

def signal_caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)

def signal_excess_punct(text: str) -> float:
    bangs = text.count("!")
    ques  = text.count("?")
    return min(1.0, (bangs + 0.5*ques) / max(1, len(text)/500))  # normalize by length

def signal_sensational(text: str) -> float:
    t = text.lower()
    found = sum(1 for w in SENSATIONAL_WORDS if w.lower() in t)
    return min(1.0, found / 6.0)

DATE_RE = re.compile(r"\b(?:\d{1,2}[/-])?(?:\d{1,2}[/-])?\d{2,4}\b")
def signal_has_dates(text: str) -> float:
    # Presence of plausible dates can (weakly) correlate with real reports / official docs.
    # We cap influence to avoid over-trusting this.
    return 1.0 if len(DATE_RE.findall(text)) >= 2 else 0.0

def signal_official_markers(text: str) -> float:
    markers = [
        "supreme court", "high court", "government of", "ministry of",
        "appellate jurisdiction", "case no.", "judgment", "reporting", "official gazette"
    ]
    t = text.lower()
    found = sum(1 for m in markers if m in t)
    return min(1.0, found / 4.0)

def compute_heuristics(text: str) -> Dict[str, float]:
    return {
        "caps_ratio": round(signal_caps_ratio(text), 3),
        "excess_punct": round(signal_excess_punct(text), 3),
        "sensational": round(signal_sensational(text), 3),
        "dates_present": signal_has_dates(text),
        "official_markers": round(signal_official_markers(text), 3),
        "length_chars": len(text),
    }

def heuristic_adjustment(heur: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw heuristics into nudges for verdict.
    Positive weights nudge 'real', negative nudge 'fake'.
    """
    nudge_real = 0.0
    nudge_fake = 0.0

    # Sensational style + lots of !/? + high CAPS can hint fake
    nudge_fake += 0.6*heur["sensational"] + 0.4*heur["excess_punct"] + 0.3*max(0.0, heur["caps_ratio"] - 0.25)

    # Presence of formal markers + some dates hints real-ish
    nudge_real += 0.5*heur["official_markers"] + 0.2*heur["dates_present"]

    # Extremely short text -> suspicious
    suspicious = 1.0 if heur["length_chars"] < 400 else 0.0

    return {
        "nudge_real": round(nudge_real, 3),
        "nudge_fake": round(nudge_fake, 3),
        "suspicious": suspicious
    }


# -----------------------------
# Zero-shot classification wrapper
# -----------------------------
LABELS = ["real", "fake", "satire", "opinion", "clickbait", "propaganda", "conspiracy", "scam", "suspicious"]

def zero_shot_verdict(zs, text: str) -> Dict:
    if not text.strip():
        return {"top_label": "suspicious", "scores": {l: (1.0 if l=="suspicious" else 0.0) for l in LABELS}}
    # Use multi_label to get probabilities per class
    out = zs(text, LABELS, multi_label=True)
    # Normalize to dict
    scores = {lbl: 0.0 for lbl in LABELS}
    for lbl, score in zip(out["labels"], out["scores"]):
        scores[lbl] = float(score)
    # pick one primary label
    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return {"top_label": top_label, "scores": scores}


# -----------------------------
# Verdict fusion
# -----------------------------
def fuse_verdict(zs_result: Dict, heur_nudges: Dict[str, float], strict_mode: bool=False) -> Tuple[str, Dict[str, float]]:
    # Start with zero-shot scores
    scores = zs_result["scores"].copy()

    # Heuristic nudges: convert to small adjustments
    scores["real"] += 0.15 * heur_nudges["nudge_real"]
    scores["fake"] += 0.15 * heur_nudges["nudge_fake"]
    scores["suspicious"] += 0.25 * heur_nudges["suspicious"]

    # Optional stricter setting to avoid overconfident REAL
    if strict_mode and scores.get("real",0) < 0.45:
        scores["suspicious"] += 0.1

    # Normalize to [0,1]
    # (not strictly necessary, but keeps things tidy)
    mx = max(scores.values()) or 1.0
    norm = {k: float(v/mx) for k,v in scores.items()}

    # Final label constrained to {real,fake,suspicious}
    primary_pool = {k: v for k, v in norm.items() if k in {"real","fake","suspicious"}}
    final_label = max(primary_pool.items(), key=lambda kv: kv[1])[0]

    return final_label, norm


# -----------------------------
# Chunking + quick relevance (keyword overlap)
# -----------------------------
def split_chunks(text: str, chunk_size=900, overlap=150) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def score_chunk(question: str, chunk: str) -> float:
    q_terms = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", question) if len(w) > 2}
    if not q_terms: return 0.0
    c_terms = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", chunk)}
    overlap = len(q_terms & c_terms)
    return overlap / (len(q_terms) + 1e-6)

def select_top_chunks(question: str, chunks: List[str], k: int = 3) -> List[str]:
    ranked = sorted(((score_chunk(question, c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]


# -----------------------------
# Chat-like answer grounded in doc
# -----------------------------
def answer_question(t2t, question: str, full_text: str) -> str:
    if not question.strip():
        return "Please enter a question."
    if not full_text.strip():
        return "No document text found."

    chunks = split_chunks(full_text)
    context_parts = select_top_chunks(question, chunks, k=3)
    context = "\n\n".join(context_parts)[:3500]  # keep prompt modest for CPU

    prompt = (
        "You are a careful assistant. Answer the user's question ONLY using the provided document context. "
        "If the answer is not present, say 'I cannot find evidence for that in the document.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    out = t2t(prompt)[0]["generated_text"].strip()
    return out


# -----------------------------
# Report model
# -----------------------------
class Report(BaseModel):
    verdict: str
    top_scores: Dict[str, float]
    heuristics: Dict[str, float]
    nudges: Dict[str, float]
    notes: List[str]
    char_count: int
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(self.dict(), indent=2)


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Document Verifier Agent")
st.caption("Upload a PDF, DOCX, or TXT. The agent will classify it as Real, Fake, or Suspicious and can answer follow-up questions about its contents.")

with st.sidebar:
    st.header("⚙️ Settings")
    strict_mode = st.toggle("Strict mode (more conservative)", value=True)
    show_raw_scores = st.toggle("Show detailed class scores", value=False)
    st.markdown("---")
    st.markdown("**How it works**")
    st.write(
        "- Zero-shot NLI model estimates likelihood of labels like *real*, *fake*, *suspicious*.\n"
        "- Heuristics nudge the verdict based on style/format cues.\n"
        "- Chat section uses FLAN-T5 to answer questions based on your document text."
    )
    st.markdown("---")
    st.write("**Note:** No web lookups are performed. All judgments are content-only and heuristic; treat results as advisory.")

uploaded = st.file_uploader("Upload a document", type=["pdf","docx","txt"])

zs = load_zero_shot_classifier()
t2t = load_text2text()

if uploaded:
    with st.status("Extracting text...", expanded=False) as status:
        try:
            text, kind = extract_text_from_upload(uploaded)
            status.update(label=f"Parsed {kind.upper()} ✔️", state="complete")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    if len(text) < 40:
        st.warning("Very little text was extracted. This often happens with scanned PDFs (images). OCR is not enabled here—result may be *Suspicious* by default.")

    # Heuristics
    heur = compute_heuristics(text)
    nudges = heuristic_adjustment(heur)

    # Zero-shot
    with st.status("Running verifier...", expanded=False):
        zs_out = zero_shot_verdict(zs, text)

    final_label, fused_scores = fuse_verdict(zs_out, nudges, strict_mode=strict_mode)

    # Human-friendly notes
    notes = []
    if nudges["nudge_fake"] > 0.2:
        notes.append("Writing style shows sensational/clickbaity signals (CAPS, !!!, or buzzwords).")
    if nudges["nudge_real"] > 0.2:
        notes.append("Contains formal markers (e.g., case/judgment language or institutional terms).")
    if nudges["suspicious"] >= 1.0:
        notes.append("Text is very short or extraction was minimal (possible scanned or partial).")
    if heur["dates_present"] >= 1.0:
        notes.append("Multiple date patterns detected.")
    if heur["official_markers"] >= 0.5:
        notes.append("Multiple official/documentary markers found.")

    # Build and show report
    report = Report(
        verdict=final_label,
        top_scores={k: round(v,3) for k,v in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:6]},
        heuristics=heur,
        nudges=nudges,
        notes=notes,
        char_count=len(text),
        timestamp=time.time()
    )

    st.subheader("🔍 Result")
    color_map = {"real": "✅", "fake": "❌", "suspicious": "⚠️"}
    st.markdown(f"### {color_map.get(final_label,'')} **{final_label.upper()}**")

    if show_raw_scores:
        st.write("**Class scores (normalized):**")
        st.json({k: round(v,3) for k,v in fused_scores.items()})

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Heuristics**")
        st.json(heur)
    with col2:
        st.write("**Heuristic nudges**")
        st.json(nudges)

    if notes:
        st.info("**Signals noted:**\n- " + "\n- ".join(notes))

    # Downloadable JSON report
    st.download_button(
        label="⬇️ Download verification report (JSON)",
        data=report.to_json().encode("utf-8"),
        file_name="verification_report.json",
        mime="application/json"
    )

    st.markdown("---")
    st.subheader("💬 Ask about this document")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask a question (e.g., 'What is this judgment about?')", key="q")
    ask = st.button("Answer")

    # Show history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Agent:** {msg}")

    if ask and user_q.strip():
        answer = answer_question(t2t, user_q, text)
        st.session_state.chat_history.append(("user", user_q))
        st.session_state.chat_history.append(("agent", answer))
        st.experimental_rerun()

else:
    st.info("Upload a PDF, DOCX, or TXT to begin. You can also paste raw text below.")

# Optional: Raw text input (no file)
st.markdown("---")
st.subheader("📎 Or paste text manually")
raw_text = st.text_area("Paste content here…", height=150, placeholder="Paste any announcement, notice, article, or snippet…")
if st.button("Verify pasted text"):
    if not raw_text.strip():
        st.warning("Please paste some text first.")
        st.stop()
    heur = compute_heuristics(raw_text)
    nudges = heuristic_adjustment(heur)
    zs_out = zero_shot_verdict(load_zero_shot_classifier(), raw_text)
    final_label, fused_scores = fuse_verdict(zs_out, nudges, strict_mode=True)
    st.markdown(f"### Result: **{final_label.upper()}**")
    st.json({"scores": {k: round(v,3) for k,v in fused_scores.items()}, "heuristics": heur, "nudges": nudges})
