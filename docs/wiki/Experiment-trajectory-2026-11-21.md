# 21 November 2026

No run today. This page records a repair to the manuscript, because the two
copies of the paper had drifted apart and neither one was complete.

## The two copies had diverged

A reviewer went through `plasticity-indicators-paper-v3.docx` and left 18
tracked edits and six comments. Those edits were applied to the LaTeX draft in
`ops-docs/AAAI-Draft/WIP/main.tex` and never to the Markdown source in
`docs/plasticity/paper-v3.md`.

Meanwhile the matched-statistics result of 20 November went into the Markdown
and never into the LaTeX. An audit of both files confirmed it: all 16 text
edits are in the LaTeX and none in the Markdown, and the LaTeX contains no
occurrence of Section 3.6, Appendix J or the matched readings.

The Markdown is the single source, because the LaTeX is generated from it. So
the review edits were brought across to the Markdown rather than the other way
round.

## What was applied

All 16 tracked text edits. Each corrected a claim the paper's own tables
contradict. The ones that change what the paper says:

| Edit | Was | Is |
| --- | --- | --- |
| Abstract | the largest correlation is 0.212 and carries the wrong sign | that figure is the fifteen-run one, and no correlation at any count exceeds 0.35 or is separable from zero |
| Appendix B | the collapse is present after one block | after two blocks, because block 1 reads 5.135 and block 2 reads 2.337 |
| Appendix B | a factor of ten at each block | about seven, since thirteen orders over fifteen steps is seven times a step |
| Appendix C | more than ten times the largest floor | more than five, since 15.90 over 2.72 is 5.8 |
| Appendix F | still in the unhealthy direction | still in the direction the measures report as healthy, which is what Appendix E actually found |

The last one had the sentence saying the opposite of its own result.

Four of the six comments are now closed in the Markdown.

The uncited claim that the literature reads a rise in curvature as a loss of
plasticity now cites Lyle et al. The Introduction gained a short related-work
paragraph. Three references were added after checking each one against its
arXiv record: Dohare et al. 2024, Lyle et al. 2023 and von Oswald et al. 2020.
The leaked draft note in the reference list is gone.

Mean row share is now defined in the Method, where it was used throughout and
never explained.

The Table B5 arithmetic is fixed. The column is a signed sum of base-ten
logarithms over 53 layers, so 20.42 and 13.17 give 0.385 and 0.248 per layer.
The 0.393 and 0.268 the text quoted are the mean of the absolute value, which
is a different quantity. Both are now reported and the ordering is the same
either way, so the conclusion stands.

Script paths, notebook names and the "7 November record" are collected into one
code and data availability section. Appendix H lost three rows that list work
this paper already reports: learning speed and the boundary-4 probe are in
Appendix G, and the three codes at boundary 4 are in Appendix C.

## The five smaller open items

All five are closed. The Table I1 floor sentence now says where the floors come
from, since Table I1 prints no floor column. srank's 10.025 is now 10.0 in
three places, because the quantity it is compared against is given to three
figures. Appendix D no longer points at Appendix A for the per-run values,
since Appendix A points at the stored records. The "single factor of two" in
Section 3.2 now states that it is about the fixed arm, and that the trained arm
changes sign between seeds so no factor describes it. The Limitations section
is four paragraphs instead of one block, and `scripts/style_paper_docx.py` now
gives the title Word's Title style so it stays out of a table of contents.

## What is still open

Comment 5, the structure of Appendix H, is only half closed. The three rows
that duplicated reported work are gone, but the appendix still cites a reading
the paper does not otherwise carry.

The LaTeX draft is now behind the Markdown by both the matched-statistics work
and this repair. It has to be regenerated before it is used again.

Figure 1 still plots the fixed route alone and was not regenerated. Table 2's
curvature split still has three variables differing at once.

## Next decision

Blocks A, B and D of the staged plan are closed and block C dissolved. Blocks E
and F are the remaining experiments: the random-label probe, about three hours
of GPU and new code, and the induced plasticity-loss stress test, fourteen
hours or more.
