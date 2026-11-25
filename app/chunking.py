"""Split the specification tree into chunks with metadata."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.corpus import BLOCK_SEPARATOR, Node

# At 250 words nearly everything fits in the 512 tokens bge-small-en-v1.5 reads.
MAX_WORDS = 250
MAX_TABLE_WORDS = 200  # binary codes use almost twice the tokens per word
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_INSTRUCTION_ROW = re.compile(r"^\|.*\| ([A-Z][A-Z0-9.]*) \|$")
_OPCODE_FIELD = re.compile(r"\| [01]{7} \|")
# Matches "(assembler pseudoinstruction SEQZ rd, rs)" and "MV rd, rs1 assembler pseudoinstruction".
_PSEUDOINSTRUCTION = re.compile(
    r"pseudoinstruction ([A-Z]+)\b|\b([A-Z]+)(?: [a-z0-9]+,?)* assembler pseudoinstruction"
)


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    section_id: str
    path: str
    source_url: str
    chunk_index: int


def _pack(units: list[str], separator: str, max_words: int) -> list[str]:
    """Group consecutive units up to max_words, unless a single unit already exceeds it."""
    parts: list[str] = []
    current: list[str] = []
    words = 0
    for unit in units:
        count = len(unit.split())
        if current and words + count > max_words:
            parts.append(separator.join(current))
            current, words = [], 0
        current.append(unit)
        words += count
    if current:
        parts.append(separator.join(current))
    return parts


def _table_header_size(lines: list[str]) -> int | None:
    """Number of header lines (caption + first row), or None if the block is not a table."""
    first_row = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    if first_row is None or first_row > 1 or len(lines) <= first_row + 1:
        return None
    if not all(line.startswith("|") for line in lines[first_row:]):
        return None
    return first_row + 1


def _split_block(block: str) -> list[str]:
    """Split an oversized block: tables between rows, everything else between sentences."""
    lines = block.splitlines()
    header_size = _table_header_size(lines)
    limit = MAX_WORDS if header_size is None else MAX_TABLE_WORDS
    if len(block.split()) <= limit:
        return [block]
    if header_size is None:
        return _pack(_SENTENCE_END.split(block), " ", MAX_WORDS)
    header = "\n".join(lines[:header_size])
    budget = MAX_TABLE_WORDS - len(header.split())
    return [f"{header}\n{rows}" for rows in _pack(lines[header_size:], "\n", budget)]


def _instruction_units(block: str) -> list[str] | None:
    """One unit per instruction if the block is a listing table, otherwise None.

    An embedding of a table with dozens of instructions represents none of them well.
    """
    lines = block.splitlines()
    rows = [
        (line, match.group(1))
        for line in lines
        if (match := _INSTRUCTION_ROW.match(line)) and _OPCODE_FIELD.search(line)
    ]
    if not rows or len(rows) < len(lines) - 1:
        return None
    has_title = len(rows) == len(lines) - 1
    label = lines[0].strip("| ") if has_title else "instruction listing"
    return [f"{mnemonic} instruction encoding ({label})\n{line}" for line, mnemonic in rows]


def _pseudoinstruction_units(text: str) -> list[str]:
    """One extra unit per sentence that defines a pseudoinstruction.

    The definition is a single sentence among many instructions, so the embedding of the
    surrounding chunk barely represents it.
    """
    units = []
    for sentence in _SENTENCE_END.split(text):
        if match := _PSEUDOINSTRUCTION.search(sentence):
            name = match.group(1) or match.group(2)
            units.append(f"{name} pseudoinstruction\n{sentence.strip()}")
    return units


def split_long_section(text: str) -> list[str]:
    """Split a section into parts of up to MAX_WORDS words.

    Parts do not overlap, except for the extra units that repeat pseudoinstruction definitions.
    """
    parts: list[str] = []
    pending: list[str] = []
    for block in text.split(BLOCK_SEPARATOR):
        units = _instruction_units(block)
        if units is None:
            pending.extend(_split_block(block))
            continue
        parts.extend(_pack(pending, BLOCK_SEPARATOR, MAX_WORDS))
        pending = []
        parts.extend(units)
    parts.extend(_pack(pending, BLOCK_SEPARATOR, MAX_WORDS))
    parts.extend(_pseudoinstruction_units(text))  # last, so existing chunk indexes do not shift
    return parts


def chunk_node(node: Node) -> list[Chunk]:
    """Build the chunks of a section, each one prefixed with its title."""
    if not node.text.strip():
        return []
    return [
        Chunk(
            text=f"{node.title}{BLOCK_SEPARATOR}{part}",
            section=node.title,
            section_id=node.id,
            path=node.path,
            source_url=node.source_url,
            chunk_index=index,
        )
        for index, part in enumerate(split_long_section(node.text))
    ]


def chunk_tree(nodes: Iterable[Node]) -> list[Chunk]:
    """Walk the nodes in reading order."""
    return [chunk for node in nodes for chunk in chunk_node(node)]
