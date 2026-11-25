"""Convert the specification HTML into a chapter > section > subsection tree.

Run: python -m app.corpus
"""

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from lxml import etree

SPEC_VERSION = "20260120"
BASE_URL = f"https://docs.riscv.org/reference/isa/v{SPEC_VERSION}/unpriv"
PAGES = ("intro", "rv32", "rv-32-64g")
RAW_DIR = Path("data/raw")
RAW_FIGURES_DIR = RAW_DIR / "figures"
CORPUS_DIR = Path("data/corpus")

MAX_LEVEL = 3  # sect1 = chapter, sect2 = section, sect3 = subsection
BLOCK_SEPARATOR = "\n\n"  # between blocks; tables and lists contain none
FIGURE_NAME = re.compile(r"[a-z]+(?:-[a-z]+)*-[0-9a-f]+\.svg")
SVG_TEXT = "{http://www.w3.org/2000/svg}text"
SVG_ROW_TOLERANCE = 3.0
SVG_COLUMN_TOLERANCE = 2.0
_TRANSLATE = re.compile(r"translate\(([-\d.]+)(?:[, ]+([-\d.]+))?\)")


def figure_names(html: str) -> list[str]:
    """Unique, validated SVG names from the article, in order of appearance."""
    article = BeautifulSoup(html, "lxml").select_one("article.doc")
    if article is None:
        return []
    names = (img.get("src", "").rsplit("/", 1)[-1] for img in article.select(".imageblock img"))
    return list(dict.fromkeys(name for name in names if FIGURE_NAME.fullmatch(name)))


@dataclass
class Node:
    id: str
    title: str
    path: str
    level: int
    parent: str | None
    source_url: str
    text: str = ""
    children: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("​", "")).strip()


def _inline(tag: Tag) -> str:
    return _clean(tag.get_text())


def _render_table(table: Tag) -> str:
    lines = []
    caption = table.find("caption")
    if caption is not None:
        lines.append(_inline(caption))
    for row in table.find_all("tr"):
        cells = [_inline(c).replace("|", "\\|") for c in row.find_all(["th", "td"])]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_list(block: Tag) -> str:
    items = block.find("ul") or block.find("ol")
    if items is None:
        return _inline(block)
    return "\n".join(f"- {_inline(li)}" for li in items.find_all("li", recursive=False))


def _render_definitions(block: Tag) -> str:
    lines = []
    for term in block.find_all("dt"):
        definition = term.find_next_sibling("dd")
        body = _inline(definition) if definition is not None else ""
        lines.append(f"- {_inline(term)}: {body}")
    return "\n".join(lines)


def _render_admonition(block: Tag) -> str:
    content = block.select_one("td.content")
    if content is None:
        return _inline(block)
    parts = [_render_block(child) for child in content.find_all(recursive=False)]
    return "Note: " + "\n".join(part for part in parts if part)


def _svg_rows(data: bytes) -> list[list[tuple[float, str]]]:
    """Group SVG texts into rows (y), each sorted by horizontal position (x)."""
    root = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    items = []
    for element in root.iter(SVG_TEXT):
        value = _clean("".join(element.itertext()))
        if not value:
            continue
        x = float(element.get("x") or 0)
        y = float(element.get("y") or 0)
        node = element
        while node is not None:
            match = _TRANSLATE.search(node.get("transform") or "")
            if match:
                x += float(match.group(1))
                y += float(match.group(2) or 0)
            node = node.getparent()
        items.append((y, x, value))

    rows: list[list[tuple[float, float, str]]] = []
    for item in sorted(items):
        if rows and item[0] - rows[-1][0][0] <= SVG_ROW_TOLERANCE:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [sorted((x, value) for _, x, value in row) for row in rows]


def _encoding_columns(
    bits: list[tuple[float, str]], fields: list[tuple[float, str]]
) -> list[tuple[float, float, str, int, str]] | None:
    """Return (left, right, range, width, field), or None if the figure does not fit."""
    columns = []
    for center, name in fields:
        same = [b for b in bits if abs(b[0] - center) <= SVG_COLUMN_TOLERANCE]
        if same:
            left, right = same[0], same[0]
        else:
            before = [b for b in bits if b[0] < center]
            after = [b for b in bits if b[0] > center]
            if not before or not after:
                return None
            left, right = before[-1], after[0]
        high, low = int(left[1]), int(right[1])
        label = str(high) if high == low else f"{high}:{low}"
        columns.append((left[0], right[0], label, high - low + 1, name))
    return columns


def _join_cell(parts: list[str]) -> str:
    # A field's bits are drawn one by one: 0 1 0 0 reads as 0100.
    separator = "" if all(len(p) == 1 and p.isdigit() for p in parts) else " "
    return separator.join(parts)


def render_encoding(data: bytes) -> str:
    """Turn an encoding figure into a table, or return its text if it does not fit."""
    rows = _svg_rows(data)
    plain = "Figure text: " + " ".join(value for row in rows for _, value in row)
    if len(rows) < 2 or not all(value.isdigit() for _, value in rows[0]):
        return plain

    bits = rows[0]
    last_bit = bits[-1][0]
    fields = [c for c in rows[1] if c[0] <= last_bit + SVG_COLUMN_TOLERANCE]
    labels = [value for x, value in rows[1] if x > last_bit + SVG_COLUMN_TOLERANCE]
    columns = _encoding_columns(bits, fields)
    if not columns:
        return plain

    body = []
    for row in rows[2:]:
        cells: list[list[str]] = [[] for _ in columns]
        for x, value in row:
            index = next(
                (
                    i
                    for i, (left, right, *_) in enumerate(columns)
                    if left - SVG_COLUMN_TOLERANCE <= x <= right + SVG_COLUMN_TOLERANCE
                ),
                None,
            )
            if index is None:
                return plain
            cells[index].append(value)
        joined = [_join_cell(cell) for cell in cells]
        if joined != [str(width) for *_, width, _ in columns]:  # the width row is redundant
            body.append(joined)

    title = "Encoding: " + (" ".join(labels) if labels else "instruction format")
    table = [["bits", *[c[2] for c in columns]], ["field", *[c[4] for c in columns]]]
    table += [["value", *row] for row in body]
    lines = ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in table]
    return "\n".join([title, *lines])


def _render_figure(block: Tag) -> str:
    parts = []
    image = block.find("img")
    name = image.get("src", "").rsplit("/", 1)[-1] if image is not None else ""
    path = RAW_FIGURES_DIR / name
    if FIGURE_NAME.fullmatch(name) and path.exists():
        try:
            parts.append(render_encoding(path.read_bytes()))
        except (etree.XMLSyntaxError, ValueError):
            pass
    title = block.select_one(".title")
    if title is not None:
        parts.append(_inline(title))
    return "\n".join(parts)


def _render_block(block: Tag) -> str:
    classes = set(block.get("class", []))
    if "admonitionblock" in classes:
        return _render_admonition(block)
    if block.name == "table":
        return _render_table(block)
    if "listingblock" in classes or "literalblock" in classes:
        pre = block.find("pre")
        if pre is None:
            return _inline(block)
        # No inner blank lines, because chunking splits on them.
        return re.sub(r"\n\s*\n", "\n", pre.get_text().strip("\n"))
    if classes & {"ulist", "olist"}:
        return _render_list(block)
    if "dlist" in classes:
        return _render_definitions(block)
    if "imageblock" in classes:
        return _render_figure(block)
    return _inline(block)


def _build_node(
    section: Tag, level: int, parent: Node | None, page: str, nodes: list[Node]
) -> None:
    heading = section.find(f"h{level + 1}", recursive=False)
    if heading is None:
        raise ValueError(f"{page}: level {level} section has no title")
    title = _inline(heading)
    anchor = heading.get("id")
    node = Node(
        id=f"{page}#{anchor}",
        title=title,
        path=f"{parent.path} > {title}" if parent else title,
        level=level,
        parent=parent.id if parent else None,
        source_url=f"{BASE_URL}/{page}.html#{anchor}",
    )
    nodes.append(node)
    if parent is not None:
        parent.children.append(node.id)

    body = section.find("div", class_="sectionbody", recursive=False) or section
    child_class = f"sect{level + 1}"
    blocks = []
    for child in body.find_all(recursive=False):
        if child is heading:
            continue
        if child_class in child.get("class", []) and level < MAX_LEVEL:
            _build_node(child, level + 1, node, page, nodes)
        else:
            rendered = _render_block(child)
            if rendered:
                blocks.append(rendered)
    node.text = BLOCK_SEPARATOR.join(blocks)


def parse_page(html: str, page: str) -> list[Node]:
    """Return a page's nodes in reading order."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article.doc")
    if article is None:
        raise ValueError(f"{page}: article.doc not found")

    for anchor in article.select("a.anchor"):
        anchor.decompose()
    for mark, prefix in (("sup", "^"), ("sub", "_")):  # keeps 2^20 from reading as 220
        for tag in article.find_all(mark):
            tag.replace_with(f"{prefix}{tag.get_text(strip=True)}")

    nodes: list[Node] = []
    for section in article.find_all("div", class_="sect1", recursive=False):
        _build_node(section, 1, None, page, nodes)
    if not nodes:
        raise ValueError(f"{page}: no sections found")
    return nodes


def build_corpus() -> dict[str, list[Node]]:
    """Parse data/raw and save each page's tree to data/corpus."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = {}
    for page in PAGES:
        raw_path = RAW_DIR / f"{page}.html"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing {raw_path}. Run: python -m scripts.fetch_corpus")
        nodes = parse_page(raw_path.read_text(encoding="utf-8"), page)
        payload = {"spec_version": SPEC_VERSION, "nodes": [asdict(node) for node in nodes]}
        output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (CORPUS_DIR / f"{page}.json").write_text(output, encoding="utf-8", newline="\n")
        corpus[page] = nodes
    return corpus


def load_nodes() -> list[Node]:
    """Read the nodes of every page from data/corpus, in reading order."""
    nodes: list[Node] = []
    for page in PAGES:
        path = CORPUS_DIR / f"{page}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run: python -m app.corpus")
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes.extend(Node(**item) for item in payload["nodes"])
    return nodes


def format_tree(nodes: list[Node]) -> str:
    """Tree outline, titles only."""
    return "\n".join(f"{'  ' * (node.level - 1)}{node.title}" for node in nodes)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for tree in build_corpus().values():
        print(format_tree(tree))
