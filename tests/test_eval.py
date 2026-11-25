import json
from datetime import datetime, timedelta

import pytest

from app import rag
from app.config import ProviderConfig
from app.llm import GenerationResult, LLMError, ProviderRateLimitedError
from app.prompts import REFUSAL
from app.retrieve import RetrievedChunk
from eval import run_eval
from eval.metrics import (
    abstention_correct,
    answer_correct,
    citation_correct,
    contains_term,
    retrieval_hit,
)

PROVIDER = ProviderConfig("fake", "https://fake.test/v1", "fake-model", "key")
QUESTIONS = [
    {
        "id": "q1",
        "question": "What is the opcode of LUI?",
        "answerable": True,
        "expected_section": "p#a",
        "must_contain": ["0110111"],
    },
    {
        "id": "q2",
        "question": "What does MRET do?",
        "answerable": False,
        "expected_section": None,
        "must_contain": [],
    },
]


def chunk(section="a", score=0.9):
    return RetrievedChunk(
        id=section,
        text=f"Text {section}",
        section=section,
        section_id=f"p#{section}",
        path=section,
        source_url=f"https://example.test/p.html#{section}",
        chunk_index=0,
        score=score,
    )


RETRIEVED = {"q1": [chunk("a")], "q2": [chunk("b")]}


@pytest.mark.parametrize(
    ("answer", "term", "expected"),
    [
        ("The value is 0.", "0", True),
        ("The value is 10.", "0", False),
        ("Uses x1 and x5", "x1", True),
        ("Uses x10", "x1", False),
        ("range of ±1 MiB", "1 MiB", True),
        ("hardware threads", "hardware thread", True),
        ("Filled with ZEROS", "zero", True),
    ],
)
def test_contains_term(answer, term, expected):
    assert contains_term(answer, term) is expected


def test_retrieval_hit():
    assert retrieval_hit("p#a", {"p#a", "p#b"}) is True
    assert retrieval_hit("p#a", {"p#b"}) is False
    assert retrieval_hit(None, {"p#b"}) is None


def test_answer_correct():
    assert answer_correct("opcode 0110111 [S1]", ["0110111"], True, False)
    assert not answer_correct("opcode 0110011 [S1]", ["0110111"], True, False)
    assert not answer_correct(REFUSAL, ["0110111"], True, True)
    assert answer_correct(REFUSAL, [], False, True)
    assert not answer_correct("Invented [S1]", [], False, False)


def test_citation_correct():
    assert citation_correct({"p#a"}, "p#a", True, False) is True
    assert citation_correct({"p#b"}, "p#a", True, False) is False
    assert citation_correct(set(), "p#a", True, True) is False
    assert citation_correct(set(), None, False, True) is None


def test_abstention_correct():
    assert abstention_correct(True, False)
    assert abstention_correct(False, True)
    assert not abstention_correct(True, True)
    assert not abstention_correct(False, False)


def test_evaluate_scores_a_model_and_pauses_between_calls(monkeypatch):
    answers = iter(["The opcode is 0110111 [S1].", REFUSAL])
    monkeypatch.setattr(
        rag,
        "generate",
        lambda question, chunks, provider: GenerationResult(next(answers), "fake-model", 7),
    )
    sleeps = []

    run = run_eval.evaluate(QUESTIONS, RETRIEVED, PROVIDER, "flat", pause=3, sleep=sleeps.append)

    assert run["model"] == "fake-model"
    assert run["retrieval_mode"] == "flat"
    assert run["errors"] == 0
    assert run["metrics"] == {
        "retrieval_hit": 1.0,
        "answer_correct": 1.0,
        "citation_correct": 1.0,
        "abstention_correct": 1.0,
    }
    assert sleeps == [3]


def test_evaluate_records_provider_errors_and_continues(monkeypatch):
    def fail(question, chunks, provider):
        raise LLMError("fake: 503")

    monkeypatch.setattr(rag, "generate", fail)

    run = run_eval.evaluate(QUESTIONS, RETRIEVED, PROVIDER, "flat", pause=0)

    assert run["errors"] == 2
    assert run["metrics"]["answer_correct"] is None
    assert run["metrics"]["retrieval_hit"] == 1.0


def test_rate_limit_waits_and_retries_once(monkeypatch):
    calls = []

    def generate(question, chunks, provider):
        calls.append(provider.name)
        if len(calls) == 1:
            raise ProviderRateLimitedError("fake", datetime.now() + timedelta(seconds=10))
        return GenerationResult("0110111 [S1]", "fake-model", 7)

    monkeypatch.setattr(rag, "generate", generate)
    sleeps = []

    response, error = run_eval.ask_with_retry("q", [chunk()], PROVIDER, sleeps.append)

    assert error is None
    assert response["answer"] == "0110111 [S1]"
    assert len(calls) == 2
    assert 10 <= sleeps[0] <= 11


def test_long_rate_limit_is_not_waited(monkeypatch):
    def generate(question, chunks, provider):
        raise ProviderRateLimitedError("fake", datetime.now() + timedelta(hours=2))

    monkeypatch.setattr(rag, "generate", generate)
    sleeps = []

    response, error = run_eval.ask_with_retry("q", [chunk()], PROVIDER, sleeps.append)

    assert response is None
    assert "rate limited" in error
    assert sleeps == []


def test_retrieval_only_run_has_no_answer_metrics():
    run = run_eval.evaluate(QUESTIONS, RETRIEVED, None, "flat")

    assert run["model"] == run_eval.RETRIEVAL_ONLY
    assert run["metrics"]["answer_correct"] is None
    assert run["metrics"]["citation_correct"] is None
    assert run["metrics"]["retrieval_hit"] == 1.0


def test_merge_runs_replaces_only_the_same_run():
    base = {
        "date": "2026-11-23T10:00:00+00:00",
        "retrieval_mode": "flat",
        "errors": 0,
        "latency_ms": {"mean": 10.0, "median": 10.0},
        "metrics": dict.fromkeys(run_eval.METRICS, 0.5),
    }
    first = {**base, "provider": "a", "model": "ma"}
    second = {**base, "provider": "b", "model": "mb"}
    updated = {**first, "errors": 3}

    merged = run_eval.merge_runs(run_eval.merge_runs([], [first, second]), [updated])

    assert len(merged) == 2
    assert next(run for run in merged if run["provider"] == "a")["errors"] == 3
    table = run_eval.markdown_table(merged)
    assert "| ma | flat | 50.0 % |" in table
    assert json.dumps(merged)


def test_holdout_flag_uses_separate_files():
    args = run_eval.parse_args(["--holdout"])

    assert args.questions == run_eval.HOLDOUT_PATH
    assert args.output_json == run_eval.HOLDOUT_JSON
    assert args.output_md == run_eval.HOLDOUT_MD


def test_holdout_does_not_repeat_tuning_questions():
    tuning = {q["question"] for q in run_eval.load_questions()}
    holdout = {q["question"] for q in run_eval.load_questions(run_eval.HOLDOUT_PATH)}
    assert not tuning & holdout


@pytest.mark.parametrize("path", [run_eval.QUESTIONS_PATH, run_eval.HOLDOUT_PATH])
def test_questions_file_follows_schema(path):
    questions = run_eval.load_questions(path)
    keys = {"id", "question", "answerable", "expected_section", "must_contain"}

    assert all(set(question) == keys for question in questions)
    assert len({question["id"] for question in questions}) == len(questions)
    for question in questions:
        if question["answerable"]:
            assert question["expected_section"] and question["must_contain"]
        else:
            assert question["expected_section"] is None
