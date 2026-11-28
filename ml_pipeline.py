"""
ml_pipeline.py - Core AIML Engine for Smart Gmail Replier.

Provides:
  1. Intent & Urgency ML Classifiers:
     - TF-IDF + Calibrated Multi-Class Logistic Regression.
     - Categorizes emails into:
         * Intent: Academic/Faculty, Internship/Recruiter, Event/Club, Action Required, Promotional/Spam
         * Urgency: High, Medium, Low
     - Computes calibrated probability/confidence scores.
  2. Tone & Sentiment Analyzer.
  3. Style & Context RAG Engine:
     - Dense sentence-transformers embeddings (all-MiniLM-L6-v2) + FAISS vector index.
     - Indexes past sent email styles (data/sent_style_examples.json) and personal facts (data/my_context.md).
     - Retrieves top-k most semantically relevant few-shot context/style matches.
  4. analyze_email():
     - Consolidated inference API returning full classification & RAG payload.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Try importing sentence_transformers and faiss; provide graceful fallbacks if needed
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA_PATH = DATA_DIR / "training_emails.json"
SENT_STYLE_PATH = DATA_DIR / "sent_style_examples.json"
CONTEXT_PATH = DATA_DIR / "my_context.md"

INTENT_MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

INTENT_CLASSES = [
    "Academic/Faculty",
    "Internship/Recruiter",
    "Event/Club",
    "Action Required",
    "Promotional/Spam",
]

URGENCY_CLASSES = ["High", "Medium", "Low"]


def format_email_text(subject: str, sender: str, body: str) -> str:
    """Combines email fields into an enriched textual representation for NLP vectorization."""
    cleaned_subject = (subject or "").strip()
    cleaned_sender = (sender or "").strip()
    cleaned_body = (body or "").strip()
    return f"Subject: {cleaned_subject}\nFrom: {cleaned_sender}\nBody: {cleaned_body}"


# ---------------------------------------------------------------------------
# 1. Intent & Urgency Classifier Engine
# ---------------------------------------------------------------------------
class EmailClassifierEngine:
    """Handles training, persistence, and inference for Intent and Urgency classification."""

    def __init__(self, retrain: bool = False):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.intent_model: Optional[LogisticRegression] = None
        self.urgency_model: Optional[LogisticRegression] = None
        self._initialize_models(retrain=retrain)

    def _initialize_models(self, retrain: bool = False):
        """Loads models from disk if present, else trains on training_emails.json."""
        if not retrain and INTENT_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            try:
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.intent_model = joblib.load(INTENT_MODEL_PATH)
                self.urgency_model = joblib.load(URGENCY_MODEL_PATH)
                return
            except Exception as e:
                print(f"[ml_pipeline] Error loading cached models ({e}), retraining...", file=sys.stderr)

        self.train()

    def train(self, training_path: Path = TRAINING_DATA_PATH):
        """Trains TF-IDF vectorizer and dual Logistic Regression models on labeled email dataset."""
        if not training_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {training_path.resolve()}")

        with open(training_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        texts: List[str] = []
        intent_labels: List[str] = []
        urgency_labels: List[str] = []

        for item in samples:
            text = format_email_text(item.get("subject", ""), item.get("sender", ""), item.get("body", ""))
            texts.append(text)
            intent_labels.append(item.get("intent", "Action Required"))
            urgency_labels.append(item.get("urgency", "Medium"))

        # Fit TF-IDF Vectorizer with unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3500,
            sublinear_tf=True,
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)

        # Fit Intent Classifier
        self.intent_model = LogisticRegression(
            C=2.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.intent_model.fit(X, intent_labels)

        # Fit Urgency Classifier
        self.urgency_model = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.urgency_model.fit(X, urgency_labels)

        # Save artifacts to models directory
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.intent_model, INTENT_MODEL_PATH)
        joblib.dump(self.urgency_model, URGENCY_MODEL_PATH)

    def predict(self, subject: str, sender: str, body: str) -> Dict[str, Any]:
        """
        Runs vectorization and calibrated probability inference.
        Returns predicted intent, intent confidence, urgency, and urgency confidence.
        """
        if self.vectorizer is None or self.intent_model is None or self.urgency_model is None:
            self._initialize_models(retrain=True)

        # High-precision domain / automated notification overrides
        snd_clean = sender.lower()
        sub_clean = subject.lower()

        # 1. Automated Marketing & Newsletters (Quora Digest, Apilayer News, Medium, Pinterest, etc.)
        if any(k in snd_clean for k in ["quora.com", "digest", "newsletter", "apilayer", "substack", "medium.com", "insideapple", "discover.pinterest.com", "pinterest.com"]) or any(k in sub_clean for k in ["digest", "newsletter", "top stories for you", "weekly roundup", "news:"]):
            return {
                "intent": "Promotional/Spam",
                "intent_confidence": 0.95,
                "urgency": "Low",
                "urgency_confidence": 0.90,
                "intent_distribution": {"Promotional/Spam": 0.95, "Action Required": 0.05},
            }

        # 2. Automated Security & Authentication (Google, GitHub OTP, 2FA)
        if any(k in snd_clean for k in ["accounts.google.com", "no-reply@accounts.google.com", "security@google.com", "account-security"]) or any(k in sub_clean for k in ["verification code", "sudo email", "security alert", "shared some google account data", "2fa", "otp", "password reset"]):
            return {
                "intent": "Action Required",
                "intent_confidence": 0.92,
                "urgency": "High",
                "urgency_confidence": 0.88,
                "intent_distribution": {"Action Required": 0.92, "Promotional/Spam": 0.08},
            }

        text = format_email_text(subject, sender, body)
        X = self.vectorizer.transform([text])

        # Intent prediction & confidence
        intent_probs = self.intent_model.predict_proba(X)[0]
        intent_idx = np.argmax(intent_probs)
        predicted_intent = str(self.intent_model.classes_[intent_idx])
        intent_conf = float(intent_probs[intent_idx])

        # Urgency prediction & confidence
        urgency_probs = self.urgency_model.predict_proba(X)[0]
        urgency_idx = np.argmax(urgency_probs)
        predicted_urgency = str(self.urgency_model.classes_[urgency_idx])
        urgency_conf = float(urgency_probs[urgency_idx])

        return {
            "intent": predicted_intent,
            "intent_confidence": round(intent_conf, 4),
            "urgency": predicted_urgency,
            "urgency_confidence": round(urgency_conf, 4),
            "intent_distribution": {
                str(cls_name): round(float(prob), 4)
                for cls_name, prob in zip(self.intent_model.classes_, intent_probs)
            },
        }


# ---------------------------------------------------------------------------
# 2. Tone & Sentiment Analysis
# ---------------------------------------------------------------------------
def detect_tone_and_sentiment(subject: str, body: str, intent: str, urgency: str) -> Tuple[str, str]:
    """
    Analyzes lexical cues, punctuation, and classified intent to assign tone and sentiment polarity.
    """
    combined = f"{subject} {body}".lower()

    # Urgency & criticality cues
    urgent_words = ["urgent", "immediately", "deadline", "expires", "asap", "critical", "mandatory", "closes today", "action required"]
    promotional_words = ["discount", "free", "sale", "win", "exclusive", "offer", "voucher", "deal", "cashback", "coupon"]
    academic_words = ["professor", "dr.", "lab", "assignment", "viva", "lecture", "cie", "marks", "grade", "department", "curriculum"]
    recruiter_words = ["interview", "internship", "hiring", "applicant", "resume", "stipend", "hackerrank", "assessment", "role"]

    # Sentiment Polarity
    if any(w in combined for w in ["congratulations", "pleased to offer", "shortlisted", "impressed", "great work", "delighted"]):
        sentiment = "Positive"
    elif any(w in combined for w in ["vulnerability", "failed", "broken", "critical", "alert", "warning", "shortage", "overdue"]):
        sentiment = "Critical / Negative"
    else:
        sentiment = "Neutral"

    # Tone categorization
    if intent == "Promotional/Spam" or any(w in combined for w in promotional_words):
        tone = "Promotional / Marketing"
    elif urgency == "High" or any(w in combined for w in urgent_words):
        tone = "Urgent & Direct"
    elif intent == "Academic/Faculty" or any(w in combined for w in academic_words):
        tone = "Formal & Respectful"
    elif intent == "Internship/Recruiter" or any(w in combined for w in recruiter_words):
        tone = "Professional & Inquisitive"
    elif intent == "Event/Club":
        tone = "Collaborative & Enthusiastic"
    else:
        tone = "Standard Informational"

    return tone, sentiment


# ---------------------------------------------------------------------------
# 3. Style & Context RAG Engine
# ---------------------------------------------------------------------------
class StyleAndContextRAG:
    """
    Retrieval-Augmented Generation (RAG) index storing sent email styles
    and personal student context using Sentence Transformers and FAISS.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.dense_model = None

        self._load_documents()
        self._build_index()

    def _load_documents(self):
        """Loads both few-shot sent email examples and markdown context chunks."""
        self.documents = []

        # 1. Load sent style examples
        if SENT_STYLE_PATH.exists():
            try:
                with open(SENT_STYLE_PATH, "r", encoding="utf-8") as f:
                    styles = json.load(f)
                for item in styles:
                    content = (
                        f"Target Intent: {item.get('intent')}\n"
                        f"Tone: {item.get('tone')}\n"
                        f"Subject Pattern: {item.get('subject_pattern')}\n"
                        f"Sample Body:\n{item.get('body')}"
                    )
                    self.documents.append({
                        "id": item.get("id", "style_example"),
                        "type": "style_example",
                        "intent": item.get("intent"),
                        "tone": item.get("tone"),
                        "content": item.get("body", ""),
                        "search_text": content,
                        "notes": f"Sent Email Template ({item.get('tone')})",
                    })
            except Exception as e:
                print(f"[RAG] Warning: Error loading sent styles: {e}", file=sys.stderr)

        # 2. Load and chunk personal context from markdown
        if CONTEXT_PATH.exists():
            try:
                with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                    context_raw = f.read()

                # Split by markdown headers
                sections = re.split(r"\n(?=##\s+)", context_raw)
                for idx, sec in enumerate(sections):
                    cleaned_sec = sec.strip()
                    if not cleaned_sec:
                        continue
                    first_line = cleaned_sec.split("\n")[0].replace("#", "").strip()
                    self.documents.append({
                        "id": f"context_chunk_{idx}",
                        "type": "personal_context",
                        "intent": "All",
                        "tone": "Fact",
                        "content": cleaned_sec,
                        "search_text": f"Arya Personal Background Context: {cleaned_sec}",
                        "notes": f"Personal Context: {first_line}",
                    })
            except Exception as e:
                print(f"[RAG] Warning: Error loading personal context: {e}", file=sys.stderr)

    def _build_index(self):
        """Encodes all loaded documents and indexes them in FAISS or a cosine-matrix fallback."""
        if not self.documents:
            return

        texts = [doc["search_text"] for doc in self.documents]

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.dense_model = SentenceTransformer(self.model_name)
                embs = self.dense_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                self.embeddings = embs.astype(np.float32)

                if HAS_FAISS:
                    dimension = self.embeddings.shape[1]
                    # Inner Product (IP) on L2-normalized embeddings equals Cosine Similarity
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    self.faiss_index.add(self.embeddings)
                return
            except Exception as e:
                print(f"[RAG] Warning: dense embedding failed ({e}), falling back to TF-IDF...", file=sys.stderr)

        # Fallback to TF-IDF Cosine Similarity
        self.dense_model = None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.embeddings = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        # Normalize rows
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant style examples and personal context chunks for query.
        Returns similarity scores between 0.0 and 1.0.
        """
        if not self.documents:
            return []

        cleaned_query = (query or "").strip()
        top_k = min(top_k, len(self.documents))

        if self.dense_model and self.faiss_index:
            query_emb = self.dense_model.encode([cleaned_query], convert_to_numpy=True, normalize_embeddings=True)
            query_emb = query_emb.astype(np.float32)
            similarities, indices = self.faiss_index.search(query_emb, top_k)
            sim_scores = similarities[0]
            hit_indices = indices[0]
        elif self.dense_model and self.embeddings is not None:
            query_emb = self.dense_model.encode([cleaned_query], convert_to_numpy=True, normalize_embeddings=True)
            sim_scores = np.dot(self.embeddings, query_emb[0])
            hit_indices = np.argsort(sim_scores)[::-1][:top_k]
            sim_scores = sim_scores[hit_indices]
        else:
            # TF-IDF Cosine Fallback
            q_vec = self.vectorizer.transform([cleaned_query]).toarray().astype(np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            sim_scores = np.dot(self.embeddings, q_vec[0])
            hit_indices = np.argsort(sim_scores)[::-1][:top_k]
            sim_scores = sim_scores[hit_indices]

        results = []
        for rank, (idx, score) in enumerate(zip(hit_indices, sim_scores)):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append({
                "rank": rank + 1,
                "id": doc["id"],
                "type": doc["type"],
                "similarity": round(float(max(0.0, min(1.0, score))), 4),
                "notes": doc["notes"],
                "content": doc["content"],
            })

        return results


# ---------------------------------------------------------------------------
# Global Singleton Instances for Fast Reuse
# ---------------------------------------------------------------------------
_CLASSIFIER_ENGINE: Optional[EmailClassifierEngine] = None
_RAG_ENGINE: Optional[StyleAndContextRAG] = None


def get_classifier_engine() -> EmailClassifierEngine:
    global _CLASSIFIER_ENGINE
    if _CLASSIFIER_ENGINE is None:
        _CLASSIFIER_ENGINE = EmailClassifierEngine()
    return _CLASSIFIER_ENGINE


def get_rag_engine() -> StyleAndContextRAG:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        _RAG_ENGINE = StyleAndContextRAG()
    return _RAG_ENGINE


def analyze_email(subject: str, sender: str, body: str) -> Dict[str, Any]:
    """
    Core AIML Pipeline Endpoint.
    Analyzes an email to return:
      - Intent & confidence
      - Urgency & confidence
      - Tone & Sentiment
      - Action requirement flag
      - Top-2 retrieved few-shot style and personal context matches
    """
    start_time = time.perf_counter()
    classifier = get_classifier_engine()
    rag = get_rag_engine()

    # 1. Classification
    clf_res = classifier.predict(subject, sender, body)
    intent = clf_res["intent"]
    intent_conf = clf_res["intent_confidence"]
    urgency = clf_res["urgency"]
    urgency_conf = clf_res["urgency_confidence"]

    # 2. Tone & Sentiment
    tone, sentiment = detect_tone_and_sentiment(subject, body, intent, urgency)

    # 3. Action Required determination
    # Non-actionable if promotional/spam or purely low-urgency broadcast
    action_required = intent != "Promotional/Spam" and (
        intent in ["Action Required", "Academic/Faculty", "Internship/Recruiter"]
        or urgency in ["High", "Medium"]
    )

    # 4. RAG Retrieval (Top-2 style/context snippets)
    query_text = f"{subject}\n{body}"
    rag_matches = rag.retrieve(query_text, top_k=2)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "intent": intent,
        "intent_confidence": intent_conf,
        "urgency": urgency,
        "urgency_confidence": urgency_conf,
        "tone": tone,
        "sentiment": sentiment,
        "action_required": action_required,
        "rag_context": rag_matches,
        "inference_latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    print("[ml_pipeline] Running sample test inference...")
    sample_sub = "Interview Invitation: AI Systems Engineering Intern"
    sample_sender = "sarah.recruiting@tensorcore.ai"
    sample_body = "Hi Arya, We were very impressed with your Smart Replier project. Are you available this Friday after 4 PM IST?"

    result = analyze_email(sample_sub, sample_sender, sample_body)
    print("\n--- ML Analysis Output ---")
    print(json.dumps(result, indent=2))
