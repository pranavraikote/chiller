"""CHiLLER API tests — no GPU required (model fully mocked)."""

import os
os.environ.setdefault("LOG_FILE", "/tmp/chiller_test/grades.jsonl")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app

# Override require_model so no GPU loading happens in any test
async def _no_model() -> None:
    pass

app.dependency_overrides[main.require_model] = _no_model

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    """Provide a mock tokenizer on mgr so inference helpers don't null-ref."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "mock prompt"
    monkeypatch.setattr(main.mgr, "tokenizer", tok)


# ── Health / Status ──────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_shape():
    r = client.get("/status")
    assert r.status_code == 200
    for key in ("model", "gpu_busy", "grader_gpu", "feedback_model", "idle_timeout"):
        assert key in r.json()


# ── Input validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload,expected_status", [
    ({"question": "  ", "response": "A",   "max_grade": 5},  400),  # blank question
    ({"question": "Q",   "response": "  ", "max_grade": 5},  400),  # blank response
    ({"question": "Q",   "response": "A",  "max_grade": 0},  422),  # max_grade below minimum
    ({"question": "Q",   "response": "A",  "max_grade": 21}, 422),  # max_grade above maximum
    ({"question": "Q"},                                       422),  # missing fields
])
def test_grade_validation(payload, expected_status):
    assert client.post("/grade", json=payload).status_code == expected_status


# ── Grade ────────────────────────────────────────────────────────────────────

_GRADE = {
    "question": "What is photosynthesis?",
    "response": "Plants use sunlight to make food.",
    "max_grade": 5,
}


def test_grade_response_schema():
    scores = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0]  # grade 3 wins
    with patch("main._score", side_effect=scores), patch("main._log"):
        r = client.post("/grade", json=_GRADE)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_grade"] == 3
    assert data["max_grade"] == 5
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["accepted"], bool)
    assert "prompt" in data


def test_grade_busy_returns_503():
    with patch.object(main._gpu_sem, "locked", return_value=True):
        assert client.post("/grade", json=_GRADE).status_code == 503


# ── Attribution ───────────────────────────────────────────────────────────────

def test_attribution_returns_tokens():
    tokens = [["plants", 0.8], ["sunlight", 0.6]]
    with patch("main._compute_attribution", return_value=tokens):
        r = client.post("/attribution", json={
            "question": "Q", "response": "A",
            "prompt": "mock prompt", "grade": 3, "max_grade": 5,
        })
    assert r.status_code == 200
    assert r.json()["tokens"] == tokens


# ── Feedback ──────────────────────────────────────────────────────────────────

def test_feedback_returns_text():
    fb = {"feedback": "Good answer. [DAMI_2021_L10]", "grounded": True, "top_score": 0.71,
          "excerpts": [{"label": "DAMI_2021_L10", "text": "lecture text", "score": 0.71, "cited": True}]}
    with patch("main._generate_feedback", return_value=fb), \
         patch("main.mgr.ensure_retriever"), patch("main._log"):
        r = client.post("/feedback", json={
            "question": "Q", "response": "A", "grade": 3, "max_grade": 5,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["feedback"] == fb["feedback"]
    assert body["grounded"] is True
    assert body["excerpts"][0]["label"] == "DAMI_2021_L10"


# ── LaTeX stripping ─────────────────────────────────────────────────────────────

def test_attribution_cleaning_splits_and_suppresses():
    from inference import _words_from_salience
    # realistic decoded tokens: word starts carry a leading space; compound tokens hold newlines/\xa0
    pieces = [(" The", 0.9), (" cat", 0.5), (" sits\n\non", 0.8), (" the", 0.4),
              (" dog", 0.7), (" .", 0.6), (" alpha\xa0beta", 0.2)]
    out = _words_from_salience(pieces)
    words = {w: s for w, s in out}
    # whitespace/newline-mashed tokens are split into clean words
    assert {"cat", "dog", "sits", "on", "alpha", "beta"} <= set(words)
    assert all("\n" not in w and "\xa0" not in w and " " not in w for w, _ in out)
    # stopwords + punctuation-only zeroed; content words keep salience
    assert words["The"] == 0.0 and words["the"] == 0.0 and words["on"] == 0.0 and words["."] == 0.0
    assert words["cat"] > 0 and words["dog"] > 0 and words["sits"] > 0 and words["alpha"] > 0


def test_strip_latex_removes_markup_keeps_citations():
    from inference import _strip_latex
    raw = (r"The SVM minimizes \(\frac{1}{2} \|w\|^2\) subject to "
           r"\(y_i(w \cdot x_i + b) \geq 1\) [DATM-slides_2025_L03_s31].")
    out = _strip_latex(raw)
    assert "\\" not in out and "$" not in out           # no LaTeX markup left
    assert "[DATM-slides_2025_L03_s31]" in out          # citation preserved
    assert "(1)/(2)" in out and ">=" in out             # frac/macro converted


# ── Correction ────────────────────────────────────────────────────────────────

def test_correct_logs_and_returns_ok():
    payload = {
        "question": "Q", "response": "A", "max_grade": 5,
        "predicted_grade": 3, "human_grade": 4, "comment": "close",
    }
    with patch("main._log") as mock_log:
        r = client.post("/correct", json=payload)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    event = mock_log.call_args[0][0]
    assert event["event"] == "correction"
    assert event["human_grade"] == 4
