# Paper plan: version 3

Written 15 November 2026. This document exists so that work on the paper can
start in a fresh chat with no other context. Read it first, then read
`docs/plasticity/paper-v3.md`.

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
tail -n +2 docs/plasticity/paper-v3.md > /tmp/skel.md && pandoc -f markdown-implicit_figures /tmp/skel.md -o ops-docs/plasticity-indicators-paper-v3.docx --reference-doc=ops-docs/plasticity-indicators-paper-draft.docx --resource-path=docs/plasticity --metadata title="Rank-based plasticity indicators and directly measured learning ability in a hypernetwork continual learner"
```

```bash
python scripts/style_paper_docx.py ops-docs/plasticity-indicators-paper-v3.docx
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
  - Spacing: 6pt after each body paragraph, and 12pt above a paragraph that
    comes straight after a table, so prose is not read as part of the table. A
    table caption gets 12pt above and 4pt below, so it sits against its table.
    A figure is centred with 12pt above, and its caption gets 4pt above and
    12pt below.
  - Captions: a paragraph that starts "Table 7." or "Figure 3.", with a
    number, is set at 10pt, the same size as the table text. A planning stub
    that starts "Table to include." is left as body text.
  - It strips the paragraph styles pandoc adds, which version 1 does not use.
    This runs before the spacing pass, because a paragraph that still carries
    a style is left alone by it.

## Numbering

Tables run in reading order: Table 1 in Section 4, Tables 2 and 3 in 5.1,
Tables 4 and 5 in 5.2, Tables 6 and 7 in 5.3. Figures the same way: Figure 1
in 5.1, Figure 2 in 5.2, Figure 3 in 5.3. A section written out of order must
take the numbers its position gives it, not the next free number.

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
- **Describe the probe from the records, not from memory.** Every record of
  the revised run has `arm="full"`, `code_only=false`, `train_code=true`, and
  `uncle/probe.py` puts the generator's parameters in the trainable list when
  `code_only` is false. The probe therefore trains the generator together
  with a fresh task code, inside a copy that is restored afterwards. Version 2
  and the 14 November record both said the generator was frozen; both are
  wrong, the record now carries a correction, and version 2 is not being
  edited. Do not reintroduce this.
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
| 1 | 5.3 Relation between indicator change and measured learning | Written. Tables 6 and 7, Figure 3. |
| 2 | 5.1 Indicator trajectories | Written. Tables 2 and 3, Figure 1. |
| 3 | 5.2 Directly measured learning ability | Written. Tables 4 and 5, Figure 2. |
| 4 | 5.4 Indicator response to a damaged model | Written 17 November. Table 8. |
| 5 | 5.5 The contribution of weight scale | Written 17 November. Table 9. |
| 6 | 5.6 Orderings and the retraction | Independent. |
| 7 | 5.7 Reproducibility of the probe | Independent. Short. |
| 8 | 5.8 The reading protocol and the indicator values | Written 17 and 18 November. Tables 10 to 14. Table 14 still a stub. |
| 8b | 5.9 Learning speed on the held-out group | Written 17 November. Tables 15 and 16. |
| 9 | 3.1, 3.2, 3.5, 3.6 Method sections | Write once the results fix what must be described. |
| 10 | 4.1, 4.2 Experimental design | Same reason. |
| 11 | 1 Introduction, 6 Discussion, 7 Limitations | Last. Claims must already be fixed. |
| 12 | Appendix A | Assemble from the result tables. |

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
| 5.8 | `docs/wiki/Experiment-trajectory-2026-11-16.md` | `51_an_ordinary_network_on_the_same_stream`, `52_both_buffer_routes_on_the_hypernetwork` |

A plain-English walkthrough of all six experiments, with every table, is in
`ops-docs/revised-regime-explainer.md`. It is a useful source for the result
sections, but its headings are declarative and must not be copied into the
paper.

## Section 5.8 and what it does to the rest

Section 5.8 was written on 17 November, after this plan. It reads the Section
5.1 models a second way, through BatchNorm statistics measured from their own
weights rather than the fixed initialisation template, and adds an ordinary
ResNet50 control trained on the same nine groups. On the matched route the
indicators move the opposite way, and the control reproduces the published fall
while learning normally.

On 18 November the section gained the depth profile and the four-arm split.
The protocol's error compounds to about thirteen orders of magnitude across the
sixteen blocks, and correcting either half of it alone recovers 0.2 and 4.3
percent of the gap at boundary 9, so the failure cannot be attributed to the
subtraction or to the division on its own. An earlier draft said the error
compounds layer by layer as if that explained the rank. It explains the size of
the values and not the rank, which is already collapsed after one block, and
the section now reports the two separately.

This changes the reading of Section 5.1 but not its numbers. The one sentence
in 5.1 that treated the fall as a property of the stream now points forward to
5.8. Two decisions are open and are the user's to make.

1. **Where the section goes.** It is at the end of Results because that needed
   no renumbering. Its argument conditions Sections 5.1 and 5.3, so it reads
   better immediately after 5.1. Moving it renumbers six sections and three
   tables.
2. **What the abstract claims.** The abstract still says the indicators fall in
   every run and treats that as the finding. With 5.8 the stronger and more
   defensible claim is that the protocol decides the direction, demonstrated on
   a control that has no plasticity problem. The abstract has not been touched.

## The number that was missing is now read

Section 5.4 needed srank's movement at the unlearn step, which earlier records
gave only as a direction. It was read from the rescaling-control file in R2 on
17 November with no new run. srank moves +73.6 at the unlearn in 15 of 15
streams, against +1.580 for effective rank, and the two interventions give
48.8 percent and 73.3 percent against effective rank's 52.3 and 83.2.

The size of an ordinary learning step, which Section 5.4 needs for scale, was
read the same way from the per-stream indicator logs: 0.162 for effective rank
and 10.025 for srank, as the average absolute change over the 120 learning
steps after the first.

## Table numbering, and the pass still owed

Tables now run 2 to 14 in reading order with no gaps. Sections 5.6 and 5.7 are
still stubs and 5.7 carries a table. When they are written they take the numbers
their position gives them, which shifts the tables in 5.8 and 5.9 again. Do that
shift once, when both are written, rather than after each one.

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
| `docs/plasticity/paper-v3.md` | Version 3. The working file. |
| `ops-docs/plasticity-indicators-paper-v3.docx` | Built output, version 1 format. |
| `ops-docs/plasticity-indicators-paper-draft.docx` | Version 1. The format reference. Do not edit. |
| `docs/plasticity/paper-draft.md` | Version 2. Rejected on style. Kept for its numbers. |
| `ops-docs/revised-regime-explainer.md` | Plain walkthrough of the six experiments. |
| `ops-docs/paper-findings-shortlist-2026-11-14.md` | The ranked findings list. |
