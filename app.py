import io
import re
import json
import time
from typing import List, Dict, Tuple

import streamlit as st
import pdfplumber
from docx import Document as DocxDocument
from transformers import pipeline
from pydantic import BaseModel
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np

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
    text_parts = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    reader = load_ocr()

    for page in pdf:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_ocr = " ".join(reader.readtext(np.array(img), detail=0))
            text_parts.append(text_ocr)
    return "\n".join(text_parts).strip()


def read_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = DocxDocument(f)
    return "\n".join([p.text for p in doc.paragraphs]).strip()


def read_image(file_bytes: bytes) -> str:
    reader = load_ocr()
    img = Image.open(io.BytesIO(file_bytes))
    text = " ".join(reader.readtext(np.array(img), detail=0))
    return text.strip()


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
# Heuristics (only used for explanations)
# -----------------------------
SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "miracle", "cure", "instant", "secret",
    "exposed", "banned", "leaked", "guaranteed", "breaking", "!!!", "act now"
}

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

def compute_heuristics(text: str) -> Dict[str, float]:
    return {
        "caps_ratio": round(signal_caps_ratio(text), 3),
        "excess_punct": round(signal_excess_punct(text), 3),
        "sensational": round(signal_sensational(text), 3),
        "length_chars": len(text),
    }


# -----------------------------
# Zero-shot classification
# -----------------------------
LABELS = ["real", "fake"]

def zero_shot_verdict(zs, text: str) -> str:
    if not text.strip():
        return "fake"
    out = zs(text, LABELS, multi_label=False)
    return out["labels"][0]


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
    char_count: int
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(self.dict(), indent=2)


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Document Verifier Agent")
st.caption("Upload a PDF, DOCX, or Image. The agent will classify it as Real or Fake.")

with st.sidebar:
    st.header("⚙️ Settings")
    detail_mode = st.toggle("Explained in detail", value=False)
    enable_discussion = st.toggle("Enable discussion about document", value=False)

uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "jpg", "jpeg", "png"])

zs = load_zero_shot_classifier()
t2t = load_text2text()

if uploaded:
    try:
        text, kind = extract_text_from_upload(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if len(text) < 40:
        st.warning("Very little text was extracted. This may affect classification.")

    final_label = zero_shot_verdict(zs, text)

    heur = compute_heuristics(text)

    report = Report(
        verdict=final_label,
        heuristics=heur,
        char_count=len(text),
        timestamp=time.time()
    )

    st.subheader("🔍 Result")
    color_map = {"real": "✅", "fake": "❌"}
    st.markdown(f"### {color_map.get(final_label,'')} **{final_label.upper()}**")

    if detail_mode:
        st.write("**Heuristics:**")
        st.json(heur)

    st.download_button(
        label="⬇️ Download verification report (JSON)",
        data=report.to_json().encode("utf-8"),
        file_name="verification_report.json",
        mime="application/json"
    )

    # -----------------
    # Chat mode
    # -----------------
    if enable_discussion:
        st.markdown("---")
        st.subheader("💬 Ask about this document")
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
