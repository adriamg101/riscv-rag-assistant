"""Guardrails: similarity-based abstention and citation validation."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.prompts import REFUSAL
from app.retrieve import RetrievedChunk

# Coarse filter. On eval/questions.json similarity does not separate unanswerable questions
# well, so finer abstention relies on the prompt and on citation validation.
ABSTAIN_THRESHOLD = 0.58

# Some models write 【S1†L2】 instead of [S1].
_CITATION_ITEM = r"S\d+(?:†[^\]】]*)?"
_CITATION_GROUP = re.compile(
    rf"[\[【]\s*({_CITATION_ITEM}(?:\s*[,;]\s*{_CITATION_ITEM})*)\s*[\]】]"
)
_CITATION_NUMBER = re.compile(r"S(\d+)")
_INVISIBLE = re.compile("[​-‍⁠﻿]")  # some models emit these inside citations


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    cited: list[int]
    invalid: list[int]


def should_abstain(chunks: Sequence[RetrievedChunk]) -> bool:
    """Abstain when there are no chunks or the best one is below the threshold."""
    return not chunks or max(chunk.score for chunk in chunks) < ABSTAIN_THRESHOLD


def extract_citations(answer: str) -> list[int]:
    """Cited numbers, without repeats, in order of appearance."""
    numbers = (
        int(number)
        for group in _CITATION_GROUP.findall(_INVISIBLE.sub("", answer))
        for number in _CITATION_NUMBER.findall(group)
    )
    return list(dict.fromkeys(numbers))


def validate_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> CitationCheck:
    """Valid when there is at least one citation and all point to chunks that were sent."""
    cited = extract_citations(answer)
    invalid = [number for number in cited if not 1 <= number <= len(chunks)]
    return CitationCheck(valid=bool(cited) and not invalid, cited=cited, invalid=invalid)


def is_refusal(answer: str) -> bool:
    return REFUSAL.casefold() in answer.replace("’", "'").casefold()
