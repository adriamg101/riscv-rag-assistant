import pytest

from app.corpus import BASE_URL, figure_names, format_tree, parse_page, render_encoding

HTML = """
<html><body>
<nav>Global menu</nav>
<article class="doc">
  <div class="breadcrumbs-container">Breadcrumbs</div>
  <h1 class="page">Chapter</h1>
  <div class="sect1">
    <h2 id="cap"><a class="anchor" href="#cap"></a>1. Chapter</h2>
    <div class="sectionbody">
      <div class="paragraph"><p>Chapter
      text.</p></div>
      <div class="sect2">
        <h3 id="sec">1.1. Section</h3>
        <div class="paragraph"><p>There are 2<sup>20</sup> values.</p></div>
        <table class="tableblock">
          <caption class="title">Table 1. Modes</caption>
          <tr><th>Field</th><th>Value</th></tr>
          <tr><td><p>fm</p></td><td><p>0000</p></td></tr>
        </table>
        <div class="admonitionblock note"><table><tr>
          <td class="icon"><i title="Note"></i></td>
          <td class="content"><div class="paragraph"><p>Comment.</p></div></td>
        </tr></table></div>
        <div class="imageblock kroki"><div class="content"><img src="a.svg" alt="svg"></div></div>
        <div class="sect3">
          <h4 id="sub">1.1.1. Subsection</h4>
          <div class="ulist"><ul><li><p>one</p></li><li><p>two</p></li></ul></div>
        </div>
      </div>
    </div>
  </div>
  <nav class="pagination">Previous Next</nav>
</article>
<footer>Copyright</footer>
</body></html>
"""


@pytest.fixture
def nodes():
    return parse_page(HTML, "mini")


def test_builds_chapter_section_subsection_tree(nodes):
    chapter, section, subsection = nodes
    assert [n.level for n in nodes] == [1, 2, 3]
    assert chapter.parent is None
    assert chapter.children == [section.id]
    assert section.parent == chapter.id
    assert subsection.parent == section.id
    assert subsection.path == "1. Chapter > 1.1. Section > 1.1.1. Subsection"
    assert subsection.source_url == f"{BASE_URL}/mini.html#sub"


def test_node_text_excludes_child_sections(nodes):
    chapter, section, _ = nodes
    assert chapter.text == "Chapter text."
    assert "one" not in section.text


def test_navigation_is_removed(nodes):
    full_text = " ".join(n.text for n in nodes)
    for noise in ("Menu", "Breadcrumbs", "Previous", "Copyright", "svg"):
        assert noise not in full_text


def test_superscript_is_preserved(nodes):
    assert "2^20 values" in nodes[1].text


def test_table_is_a_single_block(nodes):
    table = next(b for b in nodes[1].text.split("\n\n") if b.startswith("Table 1"))
    assert table == "Table 1. Modes\n| Field | Value |\n| fm | 0000 |"


def test_admonition_and_list_are_rendered(nodes):
    assert "Note: Comment." in nodes[1].text
    assert nodes[2].text == "- one\n- two"


def svg_text(x, y, value):
    return f'<g transform="translate({x},{y})"><text>{value}</text></g>'


ENCODING_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    + "".join(svg_text(x, 10, v) for x, v in [(0, 31), (20, 30), (100, 25), (120, 24), (200, 0)])
    + "".join(svg_text(x, 30, v) for x, v in [(0, "[12]"), (60, "imm"), (160, "opcode")])
    + svg_text(240, 32, "B-Type")
    + "".join(svg_text(x, 50, v) for x, v in [(0, 1), (60, 6), (160, 25)])
    + "".join(svg_text(x, 70, v) for x, v in [(0, 1), (40, 0), (80, 1), (160, "BRANCH")])
    + "</svg>"
).encode()


def test_encoding_figure_becomes_a_table():
    assert render_encoding(ENCODING_SVG) == (
        "Encoding: B-Type\n"
        "| bits | 31 | 30:25 | 24:0 |\n"
        "| field | [12] | imm | opcode |\n"
        "| value | 1 | 01 | BRANCH |"
    )


def test_unrecognized_figure_keeps_its_text():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + svg_text(0, 10, "PI")
        + svg_text(50, 10, "SO")
        + "</svg>"
    ).encode()
    assert render_encoding(svg) == "Figure text: PI SO"


def test_figure_names_are_unique_and_safe():
    html = """<article class="doc">
    <div class="imageblock"><img src="_images/svg-abc123.svg"></div>
    <div class="imageblock"><img src="_images/svg-abc123.svg"></div>
    <div class="imageblock"><img src="_images/mem-order-def456.svg"></div>
    <div class="imageblock"><img src="../../evil.exe"></div>
    </article>"""
    assert figure_names(html) == ["svg-abc123.svg", "mem-order-def456.svg"]


def test_page_without_article_fails():
    with pytest.raises(ValueError):
        parse_page("<html><body></body></html>", "empty")


def test_format_tree_only_shows_indented_titles(nodes):
    assert format_tree(nodes).splitlines() == [
        "1. Chapter",
        "  1.1. Section",
        "    1.1.1. Subsection",
    ]
