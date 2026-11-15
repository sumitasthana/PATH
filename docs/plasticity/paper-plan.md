# Paper plan: version 3

Written 15 November 2026. This document exists so that work on the paper can
start in a fresh chat with no other context. Read it first, then read
`docs/plasticity/paper-v3-skeleton.md`.

## Goal

Rewrite the plasticity-indicators paper against the revised nine-task design.
One section at a time. Version 2 (`docs/plasticity/paper-draft.md`) is
complete but was rejected on style and will not be edited further. It stays in
the repository as a record of its numbers.

## What was wrong with version 2

Stated by the user on 15 November 2026. These are the rules for version 3.

1. **Too colorful.** Four figures, fifteen tables, and heavy use of bold in
   the body. Version 3 uses the plain layout of version 1.
2. **Sensational headings.** Section titles such as "The size of the fall does
   not predict the ability to learn" and "Indicators call a broken model
   healthy" state the conclusion in the heading. Version 3 uses neutral noun
   phrases: "Relation between indicator change and measured learning".
3. **Not academic in register.** Version 3 keeps the restrained tone of
   version 1.
4. **Hard to follow.** Sentences had to be read more than once, because a
   contrast was carried across two sentences without being marked. Write one
   claim per sentence. Where two findings contrast, say so in the same
   sentence.

## Format

The output format is the format of
`ops-docs/plasticity-indicators-paper-draft.docx`, which is version 1.

Build command, from the repository root. Both steps are needed.

```bash
tail -n +2 docs/plasticity/paper-v3-skeleton.md > /tmp/skel.md && pandoc -f markdown-implicit_figures /tmp/skel.md -o ops-docs/plasticity-paper-v3-skeleton.docx --reference-doc=ops-docs/plasticity-indicators-paper-draft.docx --resource-path=docs/plasticity --metadata title="Rank-based plasticity indicators and directly measured learning ability in a hypernetwork continual learner"
```

```bash
python scripts/style_paper_docx.py ops-docs/plasticity-paper-v3-skeleton.docx
```

What each part is for:

- `tail -n +2` removes the markdown H1, so Word does not print the title twice.
- `--reference-doc` copies version 1's paragraph styles: Title, Heading 1,
  Heading 2, Normal, and nothing else.
- `-f markdown-implicit_figures` stops pandoc turning a lone image into a
  figure with its own caption, which would print the caption twice.
- `--resource-path` lets the image path in the markdown resolve.
- `scripts/style_paper_docx.py` does what `--reference-doc` cannot. Version 1
  formats its tables and its spacing directly rather than through named
  styles, so none of it is copied. The script applies the same rules:

  - Tables: a rule above and below, a light grey rule between rows, no
    vertical rules, a shaded and bold header that repeats across pages, 10pt
    text, 3pt by 5pt cell padding, and columns sized to their contents rather
    than split equally. It also removes pandoc's reference to a table style
    named "Table", which the reference document never defines.
  - Spacing: 6pt after each body paragraph. A table caption gets 12pt above
    and 4pt below, so it sits against its table. A figure is centred with 12pt
    above, and its caption gets 4pt above and 12pt below.
  - It strips the paragraph styles pandoc adds, which version 1 does not use.
    This runs before the spacing pass, because a paragraph that still carries
    a style is left alone by it.

## Captions

Version 1 puts table captions above the table and figure captions below the
figure. Keep that. Both are plain Normal paragraphs, not Word caption fields.

## Standing rules for every section

- **No number in prose that is not in a table on the same page.** This rule
  was set by the user for version 2 and still applies. Version 2 broke it
  once, in the sentence "Dormant units rise on every run of every ordering",
  which no table supports. Do not repeat that.
- **No number from the retired eight-task design**, anywhere except the
  Section 8 table of work not yet repeated, and the Section 5.6 retraction
  where the old value is the thing being retracted.
- **Do not invent a number.** If a record gives a direction but no value, read
  the value from the stored records or leave the row out and say so.
- No em dash. Plain English for a non-native reader.

## Title

Working title, used in the skeleton:

> Rank-based plasticity indicators and directly measured learning ability in a
> hypernetwork continual learner

Two alternatives, both more neutral than version 2's title:

> Measuring plasticity in a hypernetwork continual learner: rank-based
> indicators against a direct probe

> What rank-based plasticity indicators respond to in a hypernetwork continual
> learner

Version 2's title, "Rank-based plasticity indicators do not track the ability
to learn", is the one rejected as sensational.

## Order to fill the sections

Write 5.3 first. It is the central result, and the rest of the paper is
arranged around it. Then the other result sections, then the method sections
they depend on, then the framing sections last.

| Order | Section | Why here |
| --- | --- | --- |
| 1 | 5.3 Relation between indicator change and measured learning | The central result. Fixes what the paper claims. |
| 2 | 5.1 Indicator trajectories | Sets up 5.3. Needs the dormant-unit table that version 2 lacked. |
| 3 | 5.2 Directly measured learning ability | Sets up 5.3. |
| 4 | 5.4 Indicator response to a damaged model | Independent of 5.1 to 5.3. One srank number still to extract. |
| 5 | 5.5 The contribution of weight scale | Depends on 5.4. |
| 6 | 5.6 Orderings and the retraction | Independent. |
| 7 | 5.7 Reproducibility of the probe | Independent. Short. |
| 8 | 3.1, 3.2, 3.5, 3.6 Method sections | Write once the results fix what must be described. |
| 9 | 4.1, 4.2 Experimental design | Same reason. |
| 10 | 1 Introduction, 6 Discussion, 7 Limitations | Last. Claims must already be fixed. |
| 11 | Appendix A | Assemble from the result tables. |

Sections 2, 3.3, 3.4 and References are carried over from version 1 with small
changes. They do not need to be rewritten.

## Where the data is

All numbers for version 3 are already recorded. No run is needed to write any
section in the table above, with the single exception noted under 5.4.

| Section | Record | Notebook |
| --- | --- | --- |
| 5.1 | `docs/wiki/Experiment-trajectory-2026-11-13.md`, `-2026-11-14.md` | `44_order_A_five_seeds`, `45_orders_B_and_C` |
| 5.2, 5.3, 5.7 | `docs/wiki/Experiment-trajectory-2026-11-14c.md` | `47_probe_on_the_revised_streams` |
| 5.4, 5.5 | `docs/wiki/Experiment-trajectory-2026-11-14b.md` | `46_rescaling_control` |
| 5.6 | `docs/wiki/Experiment-trajectory-2026-11-14.md` | `44`, `45` |
| 3.2 | `docs/wiki/Experiment-trajectory-2026-11-13.md` | `43_learning_rate_sweep` |

A plain-English walkthrough of all six experiments, with every table, is in
`ops-docs/revised-regime-explainer.md`. It is a useful source for the result
sections, but its headings are declarative and must not be copied into the
paper.

## The one number still missing

Section 5.4 needs srank's movement at the unlearn step. The 14 November record
states that srank moves in the same direction as effective rank but gives no
value. It can be read from the rescaling-control records in R2 with no new
run. Do this before writing 5.4.

## Where version 3 stands against the 14 November shortlist

`ops-docs/paper-findings-shortlist-2026-11-14.md` ranked five findings. Two
are fully re-measured on the present design, two partly, one not at all.

| Shortlist finding | Section in version 3 | State |
| --- | --- | --- |
| 1. The diagnostics do not match direct measurement | 5.2, 5.3 | One of six arms repeated |
| 2. The diagnostics call a damaged model healthy | 5.4 | Repeated, fifteen pairs |
| 3. The mechanism | 5.5 | Rescaling arm repeated, random-direction arm not |
| 4. Curvature depends on the measuring point | not in version 3 | Not repeated |
| 5. Single-run comparisons manufacture effects | 5.6 | Repeated |

Findings 3 and 4 are therefore thin. Section 8 lists what would close them.
Whether to run those arms before submission is an open decision, not something
this plan settles.

## Files

| File | What it is |
| --- | --- |
| `docs/plasticity/paper-v3-skeleton.md` | The skeleton being filled. The working file. |
| `ops-docs/plasticity-paper-v3-skeleton.docx` | Built output, version 1 format. |
| `ops-docs/plasticity-indicators-paper-draft.docx` | Version 1. The format reference. Do not edit. |
| `docs/plasticity/paper-draft.md` | Version 2. Rejected on style. Kept for its numbers. |
| `ops-docs/revised-regime-explainer.md` | Plain walkthrough of the six experiments. |
| `ops-docs/paper-findings-shortlist-2026-11-14.md` | The ranked findings list. |
