"""Give the generated paper the table formatting version 1 uses.

Pandoc's --reference-doc copies paragraph and character styles, but version 1
formats its tables directly on the table rather than through a named style, so
nothing is copied. This applies the same direct formatting afterwards.

Read from version 1 (ops-docs/plasticity-indicators-paper-draft.docx):

    table   width 9360 twips, a 0.75pt black rule above and below, a 0.25pt
            grey rule between rows, and no vertical rules at all
    header  repeated on each page, filled E8E8E8, bold, 10pt, single spaced
    cells   10pt, padded 3pt top and bottom, 5pt left and right

Run after pandoc:

    python scripts/style_paper_docx.py ops-docs/plasticity-paper-v3-skeleton.docx
"""
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


def style_table(table):
    pr = table._tbl.tblPr

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
        for cell in row.cells:
            pr_cell = cell._tc.get_or_add_tcPr()
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


def main(path):
    document = docx.Document(path)
    for table in document.tables:
        style_table(table)
    paragraphs = normalize_body(document)
    document.save(path)
    print(f"styled {len(document.tables)} tables and reset "
          f"{paragraphs} paragraphs to Normal in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
