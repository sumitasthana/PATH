# Paper writing brief

The brief for the paper-writing fork. Point that chat at this file.

Companions: `docs/plasticity/methodology.md` is the written methodology
section, and `docs/plasticity/run-plan.md` is the code fork's brief, which says
which numbers will exist and when.

## What this fork is for

Write the plasticity-indicators paper against the revised experimental design.
Prose only. This fork does not change code and does not run experiments.

**This is a standalone paper.** Write it as a complete piece of work with its
own introduction, contribution list, results and conclusion. Do not write any
part of it as a section that might later be folded into a larger paper, and do
not hedge the framing to keep that option open. If the decision changes later,
that is a later problem.

## The paper

Working title: "Rank-based plasticity indicators partly measure weight scale,
random initialization and task order".

Current draft: `ops-docs/plasticity-indicators-paper-draft.docx`. It is
complete end to end, abstract through limitations, and every number in it comes
from a recorded run.

The claim has three parts. Effective rank converges to a common band whatever
value it starts at, so most of its reported fall tracks the random
initialization. Holding the initialization fixed and changing only the task
order changes the reported fall by a large factor while the generator travels
the same distance. Both rank measures are partly a readout of the scale of the
generated weights, which a four-way rescaling control isolates in four forward
passes with nothing trained. The sharpest case is a model with one task
deliberately destroyed to chance accuracy, where all three indicators moved in
the direction they treat as healthy.

The paper deliberately proposes no replacement measure. Keep it that way.

## Feedback being answered

1. Use all 200 Tiny ImageNet classes for continual learning. Probe on a
   separate imagery dataset.
2. Tune hyperparameters, starting with the learning rate, for high accuracy and
   low variance. Target: 5 seeds within 3 percentage points.
3. Restructure the task split: 9 tasks of 20 classes, not 8, plus 1 probing
   task on out-of-distribution data. No part of the dataset may be silently
   dropped; any subset use must be justified in the paper.
4. Write the methodology section first. It serves double duty if the work
   merges into the main paper or becomes a second paper.

Item 4 is already done, see below. On the "double duty" part, the user has
since decided: this is a standalone paper. Write it that way.

## Decisions already made

The feedback's arithmetic does not close on its own: 9 tasks of 20 classes is
180, not 200. The user chose this reading, and it is what the methodology
section is written against.

- 200 classes split into **ten** groups of twenty. Nothing is dropped.
- **Nine** groups are the learning stream.
- The **tenth** group is held out, never trained on, used as an
  in-distribution probe.
- **EuroSAT** stays as the out-of-distribution probe. It was already wired up
  and already appears in the draft's results.

So there are two probes answering different questions. The held-out Tiny
ImageNet group asks what the model does with familiar-looking images it has not
learned. The satellite images ask what it does with images unlike anything in
the stream.

Do not reopen these choices without asking the user.

## What already exists

`docs/plasticity/methodology.md` is the written methodology section,
covering data, model, hyperparameter tuning, task orderings, reading protocol,
measures, controls and compute. It is the contract this fork writes against and
the other fork implements against. If the paper needs a methodology detail that
is not in that file, add it there rather than inventing it in the paper body.

`ops-docs/paper/main.tex` is a **different paper**, a reproduction study of
UnCLe. Do not edit it. It is useful only as a reference for the TMLR style
files and the house prose voice.

## The thing to be careful about

Every result currently in the draft, Sections 4.1 through 4.3, was measured
under the **old** design of eight tasks of ten classes, which used 80 of the
200 classes. Those numbers do not carry over to the new design.

This means the paper cannot be finished in this fork alone. The results
sections depend on re-runs that the other fork's code change makes possible.

Write in this order:

1. Methodology, already drafted, refine it.
2. Introduction and related work, which barely depend on the numbers.
3. Results, as a skeleton with the claims and the table shapes but with the
   numbers left as clearly marked placeholders.
4. Discussion and limitations, which follow the results.

Never fill a placeholder with a plausible-looking number. If a number is not
from a recorded run, it does not go in the paper. The draft's own header says
this and it is the rule the work is built on.

## How much evidence each claim will have

The arms differ in weight, and the paper should not flatten them:

- **Seed convergence (4.1).** Five seeds on one ordering. This is the
  best-supported claim and the tuning target gives it a noise floor.
- **Task order (4.2).** Three orderings, one seed each on the two extra ones.
  This stays a controlled comparison rather than a sampled one. The direction
  is supported; the size is one measurement per ordering. Keep the draft's
  existing honesty here, that what makes the comparison clean is that the runs
  differ in exactly one factor and share a verified starting value.
- **Weight scale (4.3).** Seven pairs of checkpoints, one per stream run,
  across three orderings and five seeds. This is stronger than the draft's
  single pair, because the unlearn is cheap enough to append to every run. The
  draft's current hedging about resting on one pair can be dropped **only if
  the seven pairs actually agree**. Check before relaxing the language.
- **The destroyed-task case.** Seven models for the same reason. Still a
  demonstration rather than a statistic, but no longer a single case.

The limitations section must keep saying this. Tuning reduces the noise within
an arm; it does not turn a single run into several.

## Writing rules

From `AGENTS.md` and the user's global instructions:

- Plain English for a non-native freshman reader. Short sentences.
- Never use an em dash. Use a colon, a comma, a semicolon or a separate
  sentence.
- Introduce every new term in the same sentence that first uses it.
- Do not invent results, metadata, citations or conclusions.
- Several references in the draft are marked "to verify" and need their full
  bibliographic details checked before submission. Leave the marks until the
  details are actually confirmed.

## Open question for the user

- The draft's unlearning result depends on UnCLe. Confirm which version is
  being cited, since a journal version exists and the project has only read the
  2025 preprint.

## What being standalone changes

The paper now has to stand on its own introduction rather than borrowing
context from a host paper. Two things follow.

The hypernetwork setting needs explaining from scratch, early, in a paragraph
or two. A reader arriving cold does not know that a generator writes the whole
target network from a short code, and the three results do not make sense
without it. The current draft already does this in Section 3.1 but the
introduction leans on the reader catching up later.

The contribution list is the paper's whole claim, not a subset of a larger
one. The draft's four numbered contributions already read as a complete piece
of work, so this is mostly a matter of not weakening them.

The reproduction study in `ops-docs/paper/main.tex` stays a separate paper.
Cite it if useful, do not merge into it.
