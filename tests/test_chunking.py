from app.chunking import MAX_WORDS, chunk_node, chunk_tree, split_long_section
from app.corpus import Node


def make_node(text, node_id="p#sec", title="1.1. Section"):
    return Node(
        id=node_id,
        title=title,
        path=f"Chapter > {title}",
        level=2,
        parent="p#cap",
        source_url=f"https://example.test/p.html#{node_id}",
        text=text,
    )


def paragraph(words):
    return " ".join(["word"] * words)


def test_short_section_is_one_chunk_with_title_and_metadata():
    chunk = chunk_node(make_node("Short text."))[0]
    assert chunk.text == "1.1. Section\n\nShort text."
    assert chunk.section == "1.1. Section"
    assert chunk.section_id == "p#sec"
    assert chunk.path == "Chapter > 1.1. Section"
    assert chunk.source_url == "https://example.test/p.html#p#sec"
    assert chunk.chunk_index == 0


def test_long_section_is_split_with_consecutive_indexes_and_title():
    text = "\n\n".join(paragraph(200) for _ in range(5))
    chunks = chunk_node(make_node(text))
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
    assert all(c.text.startswith("1.1. Section\n\n") for c in chunks)


def test_split_has_no_overlap_and_loses_nothing():
    blocks = [f"block{i} " + paragraph(150) for i in range(8)]
    parts = split_long_section("\n\n".join(blocks))
    assert "\n\n".join(parts).split("\n\n") == blocks


def test_small_table_is_not_split():
    table = "Table 1\n| Field | Value |\n| fm | 0000 |"
    text = f"{paragraph(100)}\n\n{table}\n\n{paragraph(100)}"
    assert sum(table in c.text for c in chunk_node(make_node(text))) == 1


def test_oversized_table_splits_between_rows_repeating_header():
    header = "Table 1\n| Field | Value |"
    rows = [f"| row{i} | {paragraph(20)} |" for i in range(40)]
    parts = split_long_section("\n".join([header, *rows]))

    assert len(parts) > 1
    assert all(part.startswith(f"{header}\n") for part in parts)
    assert all(len(part.split()) <= MAX_WORDS for part in parts)
    body_rows = [line for part in parts for line in part.splitlines()[2:]]
    assert body_rows == rows


def test_oversized_note_splits_between_sentences():
    sentences = [f"Sentence{i} {paragraph(40)}." for i in range(10)]
    parts = split_long_section("Note: " + " ".join(sentences))

    assert len(parts) > 1
    assert all(len(part.split()) <= MAX_WORDS for part in parts)
    assert " ".join(parts) == "Note: " + " ".join(sentences)


def test_encoding_list_is_never_split():
    encoding = "Encoding: R-Type\n| bits | 31:25 |\n| field | funct7 |"
    text = f"{paragraph(495)}\n\n{encoding}"
    assert any(encoding in c.text for c in chunk_node(make_node(text)))


def test_instruction_listing_gets_one_chunk_per_instruction():
    listing = (
        "| RV32I Base Instruction Set |\n"
        "| imm[31:12] | rd | 0110111 | LUI |\n"
        "| imm[11:0] | rs1 | 000 | rd | 0010011 | ADDI |"
    )
    parts = split_long_section(f"Text before.\n\n{listing}\n\nText after.")

    assert parts == [
        "Text before.",
        "LUI instruction encoding (RV32I Base Instruction Set)\n"
        "| imm[31:12] | rd | 0110111 | LUI |",
        "ADDI instruction encoding (RV32I Base Instruction Set)\n"
        "| imm[11:0] | rs1 | 000 | rd | 0010011 | ADDI |",
        "Text after.",
    ]


def test_pseudoinstruction_definitions_get_their_own_chunk():
    text = (
        "ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction. "
        "Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero (assembler pseudoinstruction "
        "SEQZ rd, rs). ANDI is a logical operation."
    )

    parts = split_long_section(text)

    assert parts == [
        text,
        "MV pseudoinstruction\n"
        "ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.",
        "SEQZ pseudoinstruction\n"
        "Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero (assembler pseudoinstruction "
        "SEQZ rd, rs).",
    ]


def test_table_without_opcodes_is_not_treated_as_listing():
    table = "Table 3. Fence mode\n| fm field | Meaning |\n| 0000 | Normal Fence |"
    assert split_long_section(table) == [table]


def test_empty_node_produces_no_chunks():
    assert chunk_node(make_node("  ")) == []


def test_chunk_tree_keeps_reading_order():
    nodes = [make_node("a", "p#a", "A"), make_node("", "p#b", "B"), make_node("c", "p#c", "C")]
    assert [c.section_id for c in chunk_tree(nodes)] == ["p#a", "p#c"]
