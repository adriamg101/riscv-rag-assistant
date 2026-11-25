import pytest

from app.guardrails import (
    ABSTAIN_THRESHOLD,
    extract_citations,
    is_refusal,
    should_abstain,
    validate_citations,
)
from app.prompts import REFUSAL
from app.retrieve import RetrievedChunk


def chunk(score=0.9):
    return RetrievedChunk("id", "text", "Section", "p#s", "path", "https://x.test", 0, score)


def test_abstains_without_chunks():
    assert should_abstain([])


def test_abstains_below_threshold():
    assert should_abstain([chunk(ABSTAIN_THRESHOLD - 0.01), chunk(0.1)])


def test_does_not_abstain_if_best_chunk_reaches_threshold():
    assert not should_abstain([chunk(0.1), chunk(ABSTAIN_THRESHOLD)])


def test_valid_citations():
    check = validate_citations("ADDI adds [S1] and uses I-type [S2].", [chunk(), chunk()])
    assert check.valid
    assert check.cited == [1, 2]
    assert check.invalid == []


def test_invented_citation_is_detected():
    check = validate_citations("True [S1] and invented [S7].", [chunk(), chunk()])
    assert not check.valid
    assert check.invalid == [7]


def test_citation_zero_is_invalid():
    assert validate_citations("Wrong [S0].", [chunk()]).invalid == [0]


def test_answer_without_citations_is_invalid():
    assert not validate_citations("Answer without sources.", [chunk()]).valid


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("a [S1][S3]", [1, 3]),
        ("a [S2, S1]", [2, 1]),
        ("a [S1; S2] b [S1]", [1, 2]),
        ("a [ S4 ]", [4]),
        ("a 【S2】 b 【S1†L3-L5】", [2, 1]),
        ("a [​S5]", [5]),
        ("imm[11:0] and S1 without brackets", []),
    ],
)
def test_extract_citations(answer, expected):
    assert extract_citations(answer) == expected


def test_refusal_is_detected_with_punctuation():
    assert is_refusal(f"{REFUSAL}.")
    assert is_refusal(REFUSAL.replace("'", "’"))
    assert not is_refusal("ADDI adds an immediate [S1].")
