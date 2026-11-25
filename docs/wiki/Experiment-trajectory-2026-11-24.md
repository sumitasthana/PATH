# 24 November 2026

Random-label fitting finished on all fifteen streams. It agrees with the
direct probe everywhere, and nothing anywhere rose past its floor.

## What ran

Notebook 56, on a rented GPU rather than Colab. The protocol is the one
ordering A used on 22 November, unchanged, and the ordering A runs were reused
from the resume file rather than repeated.

Group 9, the held-out group, 2,000 training images, labels shuffled once with
seed 1020 and then fixed. A fork of the model is trained on those shuffled
labels and the final training loss is recorded. The rate is chosen from a grid
on the same loss that is reported. There is no test set and no validation set:
the question is only how much of the noise the model can memorise.

Two reading points per stream, boundary 0 before any task and boundary 9 after
the ninth, five repetitions each, fifteen streams. 450 runs in all, of which
300 were new.

The direction is backwards from everything else in the study. Lower loss means
the model memorised more, so lower means more capacity to change.

## The result

| Stream | t = 0 | t = 9 | Change | Floor | Margin |
| --- | --- | --- | --- | --- | --- |
| A seed 0 | 1.5870 | 0.0093 | -1.5778 | 0.3495 | 4.5x |
| A seed 1 | 1.6468 | 0.0003 | -1.6465 | 0.4694 | 3.5x |
| A seed 2 | 1.6398 | 0.0005 | -1.6393 | 0.2077 | 7.9x |
| A seed 3 | 1.5659 | 0.0041 | -1.5618 | 0.2108 | 7.4x |
| A seed 4 | 1.7199 | 0.0002 | -1.7197 | 0.3205 | 5.4x |
| B seed 0 | 1.6601 | 0.0113 | -1.6489 | 0.3560 | 4.6x |
| B seed 1 | 1.6759 | 0.0007 | -1.6753 | 0.2149 | 7.8x |
| B seed 2 | 1.6574 | 0.0003 | -1.6571 | 0.5346 | 3.1x |
| B seed 3 | 1.6720 | 0.0065 | -1.6655 | 0.3421 | 4.9x |
| B seed 4 | 1.7133 | 0.0020 | -1.7113 | 0.1804 | 9.5x |
| C seed 0 | 1.6885 | 0.0010 | -1.6875 | 0.3645 | 4.6x |
| C seed 1 | 1.8988 | 0.0005 | -1.8983 | 0.3642 | 5.2x |
| C seed 2 | 1.7216 | 0.0008 | -1.7209 | 0.2346 | 7.3x |
| C seed 3 | 1.6133 | 0.0209 | -1.5924 | 0.2715 | 5.9x |
| C seed 4 | 1.5407 | 0.0005 | -1.5402 | 0.2220 | 6.9x |

Across the fifteen the mean change is -1.6628 with a spread of 0.0860. Every
stream fell past its own floor, by 3.1 to 9.5 times. None rose past one.

The three orderings agree. Ordering A means -1.6290 with a spread of 0.0628,
ordering B -1.6716 with 0.0243, ordering C -1.6879 with 0.1381. The gaps
between the orderings are smaller than the spread inside ordering C, so the
order the groups were learned in does not change the answer.

Nothing started near chance. Chance for twenty classes is 2.996 and the
boundary-0 means run from 1.5407 to 1.8988, so the runs measured the model and
not the step budget.

## What the fifteen streams add

The 22 November record carried ordering A alone. Two things are different now.

The direction held on every stream, not on five. That matters because this is
the one direct measurement with nothing left to transfer: the labels are noise,
so a model cannot be helped by knowing anything about the pictures. Capacity to
change rose while the rank indicators fell, on all fifteen.

The ordering is now ruled out as an explanation. Ordering A is the ordering in
which the indicators fall least, which left room to argue that the agreement
was particular to it.

## What is still true from 22 November

The end point is saturated. Thirteen of the fifteen streams end below 0.01 and
the lowest is 0.0002, which is the bottom of the measurement's scale. The run
establishes a direction. It cannot rank one late boundary against another, and
it cannot say how much capacity rose.

The starting-point confound is unchanged. At boundary 0 the generator is
untrained, so some part of the gap is the optimiser having an easier job rather
than the model having more capacity. Every direct measurement in this study
carries the same confound.

## What this changes for the paper

Appendix H lists random-label fitting as work not yet carried out. That is now
wrong. The measurement is complete on the same fifteen streams the rest of the
paper uses, with the same floor rule, so it can be reported rather than
promised.

The decision not yet taken is where it goes. It is a sixth direct measurement
agreeing with the five already reported, so it adds weight rather than a new
claim, and the saturation limit has to be stated wherever it appears.

## It is now in the paper

As Appendix K, with the fifteen-stream table, the per-ordering means, the
chance check and both limits stated. Section 3.3 points at it in one short
paragraph, placed after the probe's two controls, because it answers the same
objection they answer and answers it more plainly: the labels are noise, so
there is nothing to transfer.

The row came out of Appendix H, which lists work not yet carried out, and
notebook 56 joins the other two notebooks named in the code availability
statement.

## The unsupported claim about other papers came out

The paper said in two places that which BatchNorm statistics a reading used is
almost never reported, or not usually reported. No audit of other papers' method
sections was ever carried out, so there was nothing behind it.

Both sentences now say only what the paper can show. The abstract says the value
itself does not record which set of statistics it was read through, and the
discussion says an indicator value carries no record of which protocol produced
it. That is a fact about the measure, not a claim about what other authors
write, and the argument does not need the stronger version.

One nearby sentence in the introduction still says the direct measurement is
usually run once, reported without an error bar, and rarely repeated. It was
left as it is.

## The body was shortened and wired up for LaTeX

The paper body, meaning everything before the appendices, went from 6,922 words
to 6,156, a cut of 766. Every cut was a place where the paper said something
twice. No number, margin, noise floor, citation or table row was removed, and
the counts were checked after the pass rather than trusted.

The abstract was five paragraphs and is now one. All six of its numbers
appeared again in Section 3. Related work lost its four subsection headings and
the topic sentence under each, which restated the heading it sat under; every
citation and every claim survives. Sections 3.2, 3.3 and 3.6 lost passages that
Appendices I, J and F already carry in full, and each now states the result and
points at the appendix. The discussion lost its opening summary of the
abstract.

Nothing structural changed. No table was merged, no section reordered, nothing
moved into an appendix. An earlier plan proposed merging Table 1 into Table 6
and Table 4 into Table 7; both were dropped, because the appendices quote body
numbers in about thirty places and Appendix J checks itself against Table 4 by
name.

The same pass prepared the file for the move to AAAI two-column LaTeX.
`refs.bib` held 7 of the 21 works the paper cites and the citation map in
`build_main.py` knew 8 of the 25 prose forms, so a rebuild produced plain-text
citations and a third of a bibliography. All 21 entries are in, taken from the
paper's own reference list, and the map now covers every form and uses natbib's
`\citep` and `\citet` instead of `\cite`. The rebuilt `main.tex` has 30
citation commands, every key resolves, and no citation is left in plain text.
`main.tex` was also stale: it had no Related work section and no Appendix K.

Five body tables carry a comment marking them to be set across both columns,
since they have six columns or long headers and will not fit one AAAI column.
Pandoc drops the comment, so it reaches neither the tex nor the docx. Table 2's
headers lost up to 40 characters each, its row labels use the b0 and b9
shorthand the paper already used in Table 1, and its caption now says the cells
use e notation. The two pieces of maths in the Method are written as maths.

## What is still over

For a seven page AAAI body, 6,160 words with seven tables and one figure is
roughly half a page to a page too long. The prose has no slack left that does
not cost a result. Closing the rest needs either results moved into appendices
or tables merged, and both change the structure, so both wait for a decision.

## Next decision

The appendices, 13,300 words, have not been touched. Appendix B at 3,263 words
and Appendix C at 2,724 are more than half of that between them. Also still
open on the paper are the two Related work sentences that rest on second-hand
reading.
