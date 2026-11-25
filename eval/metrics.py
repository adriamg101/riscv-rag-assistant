"""Evaluation metrics. They return None when a metric does not apply to the question."""

import re
from collections.abc import Collection, Sequence


def contains_term(answer: str, term: str) -> bool:
    """Terms with digits need an exact match, so "0" must not match "10"."""
    if not any(character.isdigit() for character in term):
        return term.casefold() in answer.casefold()
    body = re.escape(term).replace(r"\ ", r"\s*")
    return re.search(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", answer, re.IGNORECASE) is not None


def retrieval_hit(expected_section: str | None, retrieved_sections: Collection[str]) -> bool | None:
    """Was the expected section retrieved?"""
    if expected_section is None:
        return None
    return expected_section in retrieved_sections


def answer_correct(
    answer: str | None, must_contain: Sequence[str], answerable: bool, abstained: bool
) -> bool:
    """Answerable: contains every term. Unanswerable: the system abstains."""
    if not answerable:
        return abstained
    if abstained or not answer:
        return False
    return all(contains_term(answer, term) for term in must_contain)


def citation_correct(
    cited_sections: Collection[str],
    expected_section: str | None,
    answerable: bool,
    abstained: bool,
) -> bool | None:
    """Does any cited source belong to the expected section?"""
    if not answerable:
        return None
    return not abstained and expected_section in cited_sections


def abstention_correct(answerable: bool, abstained: bool) -> bool:
    """The system should abstain exactly when the question has no answer."""
    return abstained != answerable
