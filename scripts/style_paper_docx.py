"""Give the generated paper the table formatting version 1 uses.

Pandoc's --reference-doc copies paragraph and character styles, but version 1
formats its tables directly on the table rather than through a named style, so
nothing is copied. This applies the same direct formatting afterwards.

Read from version 1 (ops-docs/plasticity-indicators-paper-draft.docx):

    table   width 9360 twips, a 0.75pt black rule above and below, a 0.25pt
            grey rule between rows, no vertical rules, and no table style.
            Columns are sized to their contents, not split equally.
    header  repeated on each page, filled E8E8E8, bold, 10pt, single spaced
    cells   10pt, padded 3pt top and bottom, 5pt left and right
    body    6pt after each paragraph
    caption a table caption sits 12pt below the text and 4pt above its table;
            a figure sits 12pt below the text, centred; its caption sits 4pt
            below the figure and 12pt above the next text

Run after pandoc:

    python scripts/style_paper_docx.py ops-docs/plasticity-indicators-paper-v3.docx
"""
import re
import sys

import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TABLE_WIDTH = 9360          # twips, matches version 1
FILL = "E8E8E8"             # header shading
FONT_HALF_POINTS = 20       # w:sz is in half points, so 20 means 10pt
PAD = {"top": 60, "left": 100, "bottom": 60, "right": 100}   # twips


def element(tag, **attrs):
    node = OxmlElement(tag)
    for name, value in attrs.items():
        node.set(qn("w:" + name), str(value))
    return node


def replace(parent, tag):
    """Drop any existing child with this tag, then add a fresh one."""
    for old in parent.findall(qn("w:" + tag)):
        parent.remove(old)
    node = element("w:" + tag)
    parent.append(node)
    return node


def column_widths(table):
    """Split the table width between columns by how much text each holds.

    Version 1 sizes its columns to their contents: a column of labels is wide
    and a column of numbers is narrow. Pandoc splits the width equally, which
    wraps the labels and leaves the numbers floating. This copies version 1's
    shape without needing the real font metrics, by using the longest line in
    each column as a stand-in for how wide that column needs to be.
    """
    longest = [0] * len(table.columns)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(longest):
                longest[i] = max(longest[i], len(cell.text.strip()))
    # A floor of 6 characters keeps a column of one-word numbers readable.
    weights = [max(n, 6) for n in longest]
    total = sum(weights)
    widths = [int(TABLE_WIDTH * w / total) for w in weights]
    widths[-1] += TABLE_WIDTH - sum(widths)      # absorb the rounding loss
    return widths


def style_table(table):
    pr = table._tbl.tblPr

    # Pandoc points every table at a style named "Table" that version 1's
    # reference document never defines, so Word falls back to its own default
    # and can draw a full grid. Version 1 names no table style at all.
    for old in pr.findall(qn("w:tblStyle")):
        pr.remove(old)

    widths = column_widths(table)
    grid = table._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for old, size in zip(grid.findall(qn("w:gridCol")), widths):
            old.set(qn("w:w"), str(size))

    width = replace(pr, "tblW")
    width.set(qn("w:w"), str(TABLE_WIDTH))
    width.set(qn("w:type"), "dxa")

    # Horizontal rules only. Version 1 has no vertical rules, which is what
    # makes its tables read as a list rather than a grid.
    borders = replace(pr, "tblBorders")
    borders.append(element("w:top", val="single", sz=6, space=0, color="000000"))
    borders.append(element("w:bottom", val="single", sz=6, space=0, color="000000"))
    borders.append(element("w:insideH", val="single", sz=2, space=0, color="AAAAAA"))

    look = replace(pr, "tblLook")
    for name, value in (("val", "04A0"), ("firstRow", 1), ("lastRow", 0),
                        ("firstColumn", 1), ("lastColumn", 0),
                        ("noHBand", 0), ("noVBand", 1)):
        look.set(qn("w:" + name), str(value))

    for index, row in enumerate(table.rows):
        head = index == 0
        if head:
            # Repeat the header if the table breaks across a page.
            replace(row._tr.get_or_add_trPr(), "tblHeader")
        for cell_index, cell in enumerate(row.cells):
            pr_cell = cell._tc.get_or_add_tcPr()
            cell_width = replace(pr_cell, "tcW")
            cell_width.set(qn("w:w"), str(widths[min(cell_index, len(widths) - 1)]))
            cell_width.set(qn("w:type"), "dxa")
            if head:
                shade = replace(pr_cell, "shd")
                shade.set(qn("w:val"), "clear")
                shade.set(qn("w:color"), "auto")
                shade.set(qn("w:fill"), FILL)
            margins = replace(pr_cell, "tcMar")
            for side, size in PAD.items():
                margins.append(element("w:" + side, w=size, type="dxa"))
            for para in cell.paragraphs:
                # Single spacing inside cells, whatever the body style says.
                spacing = replace(para.paragraph_format.element.get_or_add_pPr(),
                                  "spacing")
                spacing.set(qn("w:line"), "240")
                spacing.set(qn("w:lineRule"), "auto")
                for run in para.runs:
                    run.font.size = docx.shared.Pt(FONT_HALF_POINTS / 2)
                    if head:
                        run.font.bold = True


# Pandoc marks body paragraphs with styles of its own. Version 1's reference
# document does not define them, so Word falls back to the default paragraph
# style and the spacing happens to come out right. The reference is still
# dangling, and version 1 carries no pStyle on body paragraphs at all, so
# remove them and leave the file in the same shape.
PANDOC_BODY_STYLES = {"BodyText", "FirstParagraph", "Compact"}


def normalize_body(document):
    """Strip pandoc's body-paragraph style references, by style id.

    This works on the XML rather than through python-docx, because an
    undefined style id reads back as "Normal" through the library and would
    never match.
    """
    paragraphs = list(document.paragraphs)
    for table in document.tables:          # cell paragraphs carry them too
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)

    changed = 0
    for para in paragraphs:
        props = para._p.find(qn("w:pPr"))
        if props is None:
            continue
        for ref in props.findall(qn("w:pStyle")):
            if ref.get(qn("w:val")) in PANDOC_BODY_STYLES:
                props.remove(ref)
                changed += 1
    return changed


# Gaps version 1 puts around each kind of paragraph, in twips. 20 twips is one
# point, so 120 is a 6pt gap. Pandoc writes no gap at all, which is why the
# text ran together.
SPACE = {
    "body":    {"after": "120"},
    "table":   {"before": "240", "after": "80"},    # caption, sits above its table
    "figure":  {"before": "240", "after": "80"},    # the image itself
    "caption": {"before": "80", "after": "240"},    # caption, sits below its figure
}


def kind_of(para):
    """Which of version 1's spacing rules this paragraph falls under."""
    if para._p.findall(".//" + qn("w:drawing")):
        return "figure"
    text = para.text.strip()
    # A numbered caption only. "Table to include." is a planning stub in an
    # unwritten section, not a caption, and must keep the body's own look.
    if re.match(r"Table \d+\.", text):
        return "table"
    if re.match(r"Figure \d+\.", text):
        return "caption"
    return "body"


def follows_a_table(para):
    """True if the thing just before this paragraph in the document is a table."""
    previous = para._p.getprevious()
    return previous is not None and previous.tag == qn("w:tbl")


def space_body(document):
    """Give each body paragraph the gap version 1 gives it.

    Headings are left alone: their gaps come from the Heading styles, which
    the reference document already supplied.
    """
    changed = 0
    for para in document.paragraphs:
        props = para._p.get_or_add_pPr()
        if props.findall(qn("w:pStyle")):          # Title, Heading 1, Heading 2
            continue
        if not para.text.strip() and kind_of(para) != "figure":
            continue                                # blank spacer paragraph
        kind = kind_of(para)
        spacing = replace(props, "spacing")
        for side, size in SPACE[kind].items():
            spacing.set(qn("w:" + side), size)
        if kind == "body" and follows_a_table(para):
            # A paragraph with only the 6pt gap below it has nothing above it,
            # so prose set straight after a table reads as part of the table.
            # Version 1 never has this case: its tables are followed by a
            # heading or by a short note that sits close on purpose. Give the
            # prose the same 12pt gap a heading would have.
            spacing.set(qn("w:before"), "240")
        if kind in ("table", "figure"):
            # Do not let a caption or a figure be split from what follows it.
            replace(props, "keepNext")
        if kind in ("table", "caption"):
            # Captions are set at 10pt, the same size as the table text, so a
            # caption does not compete with the body for attention.
            for run in para.runs:
                run.font.size = docx.shared.Pt(FONT_HALF_POINTS / 2)
        if kind == "figure":
            align = replace(props, "jc")
            align.set(qn("w:val"), "center")
        changed += 1
    return changed


def title_the_first_heading(document):
    """Give the paper's title the Title style instead of Heading 1.

    Pandoc turns the first `# ` line into a Heading 1 like any other, so the
    title lands in the table of contents beside the sections. Word's Title
    style is the one that stays out of it.
    """
    for para in document.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name == "Heading 1":
            para.style = document.styles["Title"]
            return True
        return False
    return False


def main(path):
    document = docx.Document(path)
    title_the_first_heading(document)
    for table in document.tables:
        style_table(table)
    # Strip pandoc's styles first. space_body leaves a styled paragraph alone,
    # and while pandoc's own styles are still attached every body paragraph
    # looks styled to it.
    paragraphs = normalize_body(document)
    space_body(document)
    document.save(path)
    print(f"styled {len(document.tables)} tables and reset "
          f"{paragraphs} paragraphs to Normal in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
