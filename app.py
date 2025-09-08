import io
import re
import json
import time
from typing import List, Dict, Tuple, Any

import streamlit as st
from docx import Document as DocxDocument
from transformers import pipeline
from pydantic import BaseModel
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np
from textblob import TextBlob  # ✅ Replaced language_tool_python

st.set_page_config(page_title="Document Verifier Agent", page_icon="🛡️", layout="centered")

# -----------------------------
# Caching models
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

@st.cache_resource(show_spinner=True)
def load_text2text():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)


# -----------------------------
# File readers
# -----------------------------
def read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF, fallback to OCR if no selectable text."""
    text_parts = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    reader = load_ocr()

    for page in pdf:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
        else:
            # Fallback to OCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_ocr = " ".join(reader.readtext(np.array(img), detail=0))
            if text_ocr.strip():
                text_parts.append(text_ocr)
    return "\n".join(text_parts).strip() or "⚠️ No readable text found in PDF."


def read_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = DocxDocument(f)
    return "\n".join([p.text for p in doc.paragraphs]).strip() or "⚠️ DOCX file contains no text."


def read_image(file_bytes: bytes) -> str:
    reader = load_ocr()
    img = Image.open(io.BytesIO(file_bytes))
    text = " ".join(reader.readtext(np.array(img), detail=0))
    return text.strip() or "⚠️ No readable text found in image."


def extract_text_from_upload(upload) -> Tuple[str, str]:
    name = upload.name.lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return read_pdf(data), "pdf"
    elif name.endswith(".docx"):
        return read_docx(data), "docx"
    elif name.endswith((".jpg", ".jpeg", ".png")):
        return read_image(data), "image"
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or Image.")


# -----------------------------
# Heuristics + Suspicious Rules
# -----------------------------
SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "miracle", "cure", "instant", "secret",
    "exposed", "banned", "leaked", "guaranteed", "breaking", "!!!", "act now"
}

SUSPICIOUS_PATTERNS = [
    r"cash in hand",
    r"permanent residency",
    r"visa sponsorship",
    r"issued for any purpose",
    r"guaranteed job",
    r"work permit",
    r"lottery",
    r"unlimited salary",
]

# Whitelist of trusted institutions (bias toward real)
TRUSTED_INSTITUTIONS = [
    "supreme court of pakistan",
    "high court",
    "government of",
    "ministry of",
    "university",
    "board of",
    "parliament",
    "president of",
    "prime minister",
    "civil appeal",
    "justice",
    "respondent",
    "appellant",
    "judgment"
]

def signal_caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)

def signal_excess_punct(text: str) -> float:
    bangs = text.count("!")
    ques  = text.count("?")
    return min(1.0, (bangs + 0.5*ques) / max(1, len(text)/500))

def signal_sensational(text: str) -> float:
    t = text.lower()
    found = sum(1 for w in SENSATIONAL_WORDS if w in t)
    return min(1.0, found / 6.0)

def signal_suspicious(text: str) -> float:
    t = text.lower()
    matches = sum(1 for pat in SUSPICIOUS_PATTERNS if re.search(pat, t))
    return min(1.0, matches / 3.0)

def compute_heuristics(text: str) -> Dict[str, float]:
    return {
        "caps_ratio": round(signal_caps_ratio(text), 3),
        "excess_punct": round(signal_excess_punct(text), 3),
        "sensational": round(signal_sensational(text), 3),
        "suspicious": round(signal_suspicious(text), 3),
        "length_chars": len(text),
    }

def nudges_from_heuristics(h: Dict[str, float]) -> List[str]:
    nudges = []
    if h["caps_ratio"] > 0.3 and h["length_chars"] < 2000:
        nudges.append("Contains unusually high proportion of capital letters.")
    if h["excess_punct"] > 0.2:
        nudges.append("Has excessive punctuation (!!! or ???).")
    if h["sensational"] > 0.2:
        nudges.append("Uses sensational/trigger words often seen in fake content.")
    if h["suspicious"] > 0.2:
        nudges.append("Contains suspicious claims (e.g., cash in hand, permanent residency).")
    if h["length_chars"] < 50:
        nudges.append("Text is very short; difficult to verify authenticity.")
    if not nudges:
        nudges.append("No obvious red flags detected.")
    return nudges


# -----------------------------
# Legal Quality Check (TextBlob)
# -----------------------------
def check_legal_document_quality(text: str) -> Dict[str, Any]:
    results = {"grammar_issues": [], "missing_essentials": []}

    # Grammar/spelling check
    blob = TextBlob(text)
    corrected = str(blob.correct())
    if corrected != text:
        results["grammar_issues"].append("Possible grammar/spelling improvements suggested.")

    # Legal essentials
    essentials = ["justice", "respondent", "appellant", "judgment", "date", "civil appeal"]
    for item in essentials:
        if item not in text.lower():
            results["missing_essentials"].append(item)

    return results


# -----------------------------
# Classification
# -----------------------------
LABELS = ["real", "fake", "suspicious"]

def zero_shot_verdict(zs, text: str) -> Tuple[str, float, Dict[str, Any]]:
    if not text.strip() or text.startswith("⚠️"):
        return "fake", 0.0, {}

    out = zs(text, LABELS, multi_label=False)
    base_label = out["labels"][0]
    base_score = out["scores"][0]

    susp_score = signal_suspicious(text)

    # ✅ Trusted judicial docs
    if any(marker in text.lower() for marker in TRUSTED_INSTITUTIONS):
        quality_report = check_legal_document_quality(text)
        if quality_report["missing_essentials"] or quality_report["grammar_issues"]:
            return "suspicious", 0.7, quality_report
        return "real", 0.95, quality_report

    # 🚨 Fake employment / residency style docs
    if susp_score > 0.4 or "certificate of employment" in text.lower():
        return "fake", 0.95, {}

    return base_label, round(base_score, 3), {}


# -----------------------------
# Q&A
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

def answer_question(t2t, question: str, full_text: str, max_tokens: int = 80) -> str:
    if not question.strip():
        return "Please enter a question."
    if not full_text.strip():
        return "No document text found."

    chunks = split_chunks(full_text)
    context = "\n\n".join(chunks[:3])[:3500]

    prompt = (
        f"Answer using only the document context:\n\nContext:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    out = t2t(prompt, max_new_tokens=max_tokens)[0]["generated_text"].strip()
    return out


# -----------------------------
# Report Model
# -----------------------------
class Report(BaseModel):
    verdict: str
    heuristics: Dict[str, float]
    confidence: float
    char_count: int
    timestamp: float
    quality_report: Dict[str, Any] = {}

    def to_json(self) -> str:
        return json.dumps(self.dict(), indent=2)


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Document Verifier Agent")

with st.sidebar:
    st.header("⚙️ Settings")
    text_input_mode = st.toggle("Text Input Mode", value=False)
    detail_mode = st.toggle("Explained in detail", value=False)
    enable_discussion = st.toggle("Enable discussion about document", value=False)

zs = load_zero_shot_classifier()
t2t = load_text2text()

# -----------------------------
# Main Input Handling
# -----------------------------
text = ""
if text_input_mode:
    st.subheader("✍️ Enter Text to Verify")
    text = st.text_area("Paste text here", height=200)
else:
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded:
        try:
            text, kind = extract_text_from_upload(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

# -----------------------------
# Processing
# -----------------------------
if text.strip():
    if len(text) < 40 and not text.startswith("⚠️"):
        st.warning("Very little text was provided. This may affect classification.")

    final_label, confidence, quality_report = zero_shot_verdict(zs, text)
    heur = compute_heuristics(text)
    nudges = nudges_from_heuristics(heur)

    report = Report(
        verdict=final_label,
        heuristics=heur,
        confidence=confidence,
        char_count=len(text),
        timestamp=time.time(),
        quality_report=quality_report
    )

    st.subheader("🔍 Result")
    color_map = {"real": "✅", "fake": "❌", "suspicious": "⚠️"}
    st.markdown(f"### {color_map.get(final_label,'')} **{final_label.upper()}**")

    if detail_mode:
        st.write("**Confidence Score:**", confidence)
        st.write("**Heuristics:**")
        st.json(heur)
        st.write("**Nudges:**")
        for n in nudges:
            st.markdown(f"- {n}")

        if quality_report:
            st.write("**Legal Document Quality Checks:**")
            if quality_report.get("grammar_issues"):
                st.warning("Grammar/Spelling Issues:")
                for g in quality_report["grammar_issues"]:
                    st.markdown(f"- {g}")
            if quality_report.get("missing_essentials"):
                st.error("Missing Essential Legal Elements:")
                for m in quality_report["missing_essentials"]:
                    st.markdown(f"- {m}")

    st.download_button(
        label="⬇️ Download verification report (JSON)",
        data=report.to_json().encode("utf-8"),
        file_name="verification_report.json",
        mime="application/json"
    )

    if enable_discussion:
        st.markdown("---")
        st.subheader("💬 Ask about this input")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_q = st.text_input("Ask a question (e.g., 'What is this about?')", key="q")
        ask = st.button("Answer")

        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Agent:** {msg}")

        if ask and user_q.strip():
            answer = answer_question(t2t, user_q, text, max_tokens=80)
            if answer.endswith("..."):
                cont = st.radio("The answer seems cut off. Continue?", ["No", "Yes"], horizontal=True, key="cont")
                if cont == "Yes":
                    answer = answer_question(t2t, user_q, text, max_tokens=100)

            st.session_state.chat_history.append(("user", user_q))
            st.session_state.chat_history.append(("agent", answer))
            st.rerun()
else:
    st.info("Upload a PDF, DOCX, or Image to begin.")
