"""
Intent Agent
-------------
BERT-based intent classifier (4-class: Reveal, Interpret, Revise, Chitchat).
Rule-based extensions handle Revise signals not captured by the encoder.
Inquire is kept outside the encoder as a ProductQA fallback route, since the
model was trained on 4 classes only and PSCoN has very few Inquire examples.

Routing logic:
  - If session stage is already set (not first turn) → skip classification,
    return current intent from session. Caller decides when to override.
  - Revise is detected by rule first (explicit rejection signals), then BERT.
  - Inquire is fallback-only / ProductQA route (not in training set).
  - BERT handles Reveal / Interpret / Chitchat with confidence threshold.
  - Below threshold → fallback to Interpret (safest default mid-session).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BERT_MODEL_DIR = Path(
    os.getenv(
        "MACPRS_INTENT_MODEL_DIR",
        str(_PROJECT_ROOT / "bert_4class_intent_newclassifier"),
    )
)
MAX_HISTORY_TURNS    = 6   # last N turns fed as context to BERT


def _env_float(name: str, default: float, minimum: float = None, maximum: float = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[IntentAgent] Invalid {name}={raw!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


CONFIDENCE_THRESHOLD = _env_float("INTENT_THRESHOLD", 0.65, minimum=0.0, maximum=1.0)
MARGIN_THRESHOLD     = _env_float("INTENT_MARGIN_THRESHOLD", 0.15, minimum=0.0, maximum=1.0)


# ── Classifier ────────────────────────────────────────────────────────────────

class _BERTClassifier:
    """Thin wrapper around the saved BERT artifact. Loaded once."""

    def __init__(self):
        meta_path = BERT_MODEL_DIR / "agentic_model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Intent model metadata is missing: {meta_path}. "
                "Ensure the bundled intent model directory is present."
            )
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(BERT_MODEL_DIR)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.max_len = self.meta.get("max_len", 256)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred   = int(np.argmax(probs))
        return {
            "label":         self.id2label[pred].lower(),
            "confidence":    float(probs[pred]),
            "probabilities": {
                self.id2label[i].lower(): float(probs[i])
                for i in range(len(probs))
            },
        }


# ── Rules ─────────────────────────────────────────────────────────────────────

# PSCoN "revise" = user pivots to a completely different product category or
# brand, not just adding a price/feature constraint.
# Constraint additions ("cheaper", "I don’t like the brand") → interpret.
# Explicit product pivot → revise.
_REVISE_SIGNALS = [
    "actually i want", "actually i need", "actually i’d like",
    "on second thought", "changed my mind",
    "instead i want", "instead i need",
    "no i want", "no i need",
    "i want something else", "i need something else",
    "not that", "not this",
    "forget it", "never mind",
    "different product", "different category",
    "switch to", "change to",
]

# Signals that look like revise but are interpret (adding constraint) or acceptance
_NOT_REVISE = [
    # Acceptance / selection
    "first one", "second one", "third one", "last one",
    "i’ll take", "i will take", "i’ll buy", "i will buy",
    "i’ll go with", "i will go with",
    "i like the first", "i like the second",
    "thank", "thanks",
    # Constraint additions — PSCoN labels these as interpret
    "too expensive", "too cheap", "cheaper", "lower price",
    "too big", "too small", "too heavy", "too loud",
    "don’t like the brand", "don’t like the color",
    "don’t like the price",
]

_INQUIRE_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which",
    "can you", "could you", "do you", "does it", "is it",
    "are there", "is there", "will it",
)

_INQUIRE_DETAIL_TERMS = [
    "warranty", "return", "shipping", "delivery", "available",
    "compatible", "battery", "waterproof", "material", "weight",
    "dimension", "size", "color", "rating", "review", "feature",
    "price", "cost", "how much",
]

_CHITCHAT_EXACT = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "how are you", "how r u", "ok", "okay", "sure", "cool",
    "great", "perfect", "awesome", "nice", "good",
    "i see", "got it", "understood", "noted",
    "yes", "no", "yep", "nope", "yeah", "nah",
    "fine", "alright", "all right",
}

_ACCEPTANCE_SIGNALS = [
    "thank", "thanks", "thank you",
    "this works", "that works", "works for me",
    "sounds good", "looks good", "looks nice",
    "perfect", "great choice", "that will do",
    "i'll take", "i will take", "i'll buy", "i will buy",
    "i'll go with", "i will go with",
    "i like the first", "i like the second", "i like the third",
    "first one is good", "second one is good", "third one is good",
    "the first one is good", "the second one is good", "the third one is good",
    "ok thanks", "okay thanks", "cool thanks",
    "i'm done", "im done", "no thanks", "not needed",
]


def _is_revise(message: str) -> bool:
    m = message.lower().replace("’", "’").replace("’", "’")
    if any(sig in m for sig in _NOT_REVISE):
        return False
    return any(sig in m for sig in _REVISE_SIGNALS)


def _is_inquire(message: str, has_products: bool) -> bool:
    """Inquire = explicit product-detail question after at least one recommendation."""
    if not has_products:
        return False
    m = message.lower().strip()
    # Must be a question (? or question starter)
    if "?" not in message and not any(m.startswith(s) for s in _INQUIRE_QUESTION_STARTERS):
        return False
    return any(term in m for term in _INQUIRE_DETAIL_TERMS)


def _is_chitchat(message: str) -> bool:
    m = " ".join(message.lower().strip().split()).strip(".,!?")
    return m in _CHITCHAT_EXACT


def _is_acceptance(message: str, has_products: bool) -> bool:
    """
    Acceptance/done is a dialog-policy route, not an encoder class. Only fire
    after recommendations exist; otherwise short words like "good" or "fine"
    at the start of a session would incorrectly end the flow.
    """
    if not has_products:
        return False
    m = " ".join(message.lower().replace("’", "'").strip().split()).strip(".,!?")
    if m in {"ok", "okay", "fine", "good", "great", "perfect", "cool", "yes", "yep"}:
        return True
    return any(sig in m for sig in _ACCEPTANCE_SIGNALS)


def _prob_margin(probs: dict) -> float:
    if not probs:
        return 0.0
    vals = sorted((float(v) for v in probs.values()), reverse=True)
    if len(vals) < 2:
        return vals[0]
    return vals[0] - vals[1]


# ── Public agent ──────────────────────────────────────────────────────────────

class IntentAgent:
    """
    Classifies user intent for the taxonomy-aware CRS pipeline.

    Usage:
        agent = IntentAgent()
        result = agent.classify(
            user_message="I need running shoes under $100",
            history=[],
            session_stage="new",          # "new" | "clarifying" | "searching"
                                          # | "recommending" | "revising"
            has_products=False,
        )
        # result["intent"]      → "reveal" | "interpret" | "revise"
        #                          | "inquire" | "chitchat"
        # result["confidence"]  → float
        # result["source"]      → "rule" | "bert" | "fallback" | "fallback_inquire"
        # result["run_bert"]    → bool (False if skipped)
    """

    _instance = None   # singleton — load model once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._classifier = None
        return cls._instance

    def _load(self):
        if self._classifier is None:
            print("[IntentAgent] Loading BERT model...")
            self._classifier = _BERTClassifier()
            print("[IntentAgent] Ready.")

    # ── Main entry ────────────────────────────────────────────────────────────

    def classify(
        self,
        user_message: str,
        history: list,
        session_stage: str = "new",
        has_products: bool = False,
    ) -> dict:
        """
        Returns intent dict. Skips BERT when session stage makes it unnecessary.

        session_stage values:
          "new"          → first turn, run full classification
          "clarifying"   → awaiting slot answer, skip intent — treat as Interpret
          "recommending" → post-recommendation, check for Revise/Inquire only
          "revising"     → mid-revision, treat as Revise
        """

        # ── Rule layer (always checked before BERT) ────────────────────────────
        if _is_acceptance(user_message, has_products):
            return self._result("acceptance", 1.0, "acceptance_gate", run_bert=False)

        if _is_chitchat(user_message):
            return self._result("chitchat", 1.0, "rule", run_bert=False)

        if _is_inquire(user_message, has_products):
            return self._result("inquire", 1.0, "fallback_inquire", run_bert=False)

        if _is_revise(user_message):
            return self._result("revise", 0.9, "rule", run_bert=False)

        # ── BERT classification ────────────────────────────────────────────────
        self._load()
        context = self._format_context(user_message, history)
        pred    = self._classifier.predict(context)

        label      = pred["label"]       # reveal / interpret / revise / chitchat
        confidence = pred["confidence"]
        probs      = pred["probabilities"]
        margin     = _prob_margin(probs)

        if confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
            if label == "revise" and session_stage != "new" and margin < max(MARGIN_THRESHOLD, 0.20):
                # Revise can reset slots/category, so only trust it when the
                # encoder is clearly separated from its runner-up. Explicit
                # revise phrases are still caught by _is_revise above.
                label = "interpret"
                source = "bert_revise_softened"
            else:
                source = "bert"
            return self._result(label, confidence, source, run_bert=True, probs=probs, margin=margin)

        # ── Below threshold fallback ───────────────────────────────────────────
        # Use second-best BERT prediction if it's meaningfully higher than random
        # and more appropriate given session stage.
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        second_label, second_conf = sorted_probs[1] if len(sorted_probs) > 1 else (label, 0.0)

        if session_stage == "new":
            # First turn — prefer reveal/interpret over revise/chitchat
            if label in ("reveal", "interpret"):
                fallback = label
            elif second_label in ("reveal", "interpret"):
                fallback = second_label
            else:
                fallback = "reveal"
        else:
            # Mid-session fallbacks and safeguards
            if session_stage == "revising":
                fallback = "revise"
            elif session_stage == "clarifying":
                fallback = "interpret"
            elif label == "interpret":
                fallback = "interpret"
            elif second_label == "interpret" and second_conf > 0.20:
                fallback = "interpret"
            elif label == "revise" and session_stage == "recommending":
                fallback = "revise"
            else:
                fallback = "interpret"

        return self._result(fallback, confidence, "fallback", run_bert=True, probs=probs, margin=margin)

    def classify_raw_encoder(self, user_message: str, history: list = None) -> dict:
        """
        Direct saved-encoder prediction with no rules, thresholds, fallbacks, or
        session-stage overrides. This is for offline evaluation/reporting so we
        can separate original model quality from pipeline policy behavior.
        """
        self._load()
        context = self._format_context(user_message, history or [])
        pred = self._classifier.predict(context)
        return self._result(
            pred["label"],
            pred["confidence"],
            "raw_bert",
            run_bert=True,
            probs=pred["probabilities"],
            margin=_prob_margin(pred["probabilities"]),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_context(self, user_message: str, history: list) -> str:
        lines = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role    = turn.get("role", "").upper()
            content = str(turn.get("content", "")).strip()
            if role and content:
                lines.append(f"{role}: {content}")
        lines.append(f"USER: {user_message.strip()}")
        return "\n".join(lines)

    def _result(
        self,
        intent: str,
        confidence: float,
        source: str,
        run_bert: bool,
        probs: dict = None,
        margin: float = None,
    ) -> dict:
        return {
            "intent":        intent,
            "confidence":    round(confidence, 4),
            "source":        source,   # rule | bert | fallback | stage_rule
            "run_bert":      run_bert,
            "probabilities": probs or {},
            "margin":        round(float(margin), 4) if margin is not None else None,
        }
