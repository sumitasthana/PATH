# 23 November 2026

No new run finished today. Random labels on orderings B and C is running. Two
literature searches came back, and they change what the paper is allowed to
claim.

## The LaTeX and the docx caught up

The LaTeX draft was regenerated from the Markdown and now carries Section 3.6,
Appendix J, the matched readings and all seven references. The docx was rebuilt
from the same source and saved as version 4. The three copies agree again.

## Two searches on the Related Work section

The paper has seven references and no Related Work section. Two deep research
tools were given the paper and the same prompt, which asked for four threads,
a verifiable identifier for every item, and an empty thread reported as empty
rather than padded.

### The closest prior work was already cited, for the wrong thing

Lyle et al. 2023 is in the reference list. It is cited once, in the
Introduction, for the existence of plasticity loss. Section 5 of that paper
puts feature rank, saturated units, weight norm and weight rank next to
directly measured plasticity, finds no robust relationship, and reports the
correlation signs flipping with the observation space and the reward
structure.

That is this paper's third finding, three years earlier, in reinforcement
learning. The result here is still stronger, because the direct measurement is
repeated 990 times and carries a per-model noise floor, and because the reading
protocol is identified as what decides the sign. But the paper has to be
positioned as continuing that work rather than opening the question.

### Finding 1 has precedents, in a different field

One search reported the measurement-protocol thread as empty. The other found a
literature on which statistics a network is read through: AdaBN
(arXiv:1603.04779), prediction-time BatchNorm (arXiv:2006.10963), covariate
shift adaptation (arXiv:2006.16971), Wu and Johnson on the batch choice
(arXiv:2105.07576), Continual Normalization (arXiv:2203.16102), and Kalani et
al. (arXiv:2609.08901), who show a weight-preserving pass that rewrites
BatchNorm running statistics reversing machine-unlearning evaluation metrics by
up to 78 points.

The difference between the two searches is the search terms, not the
literature. One looked inside continual learning, where BatchNorm work is about
forgetting. The other looked at test-time adaptation, where it is about exactly
this.

So the claim that nobody has looked at this is gone. What survives is narrower
and cleaner: nobody has shown it for plasticity indicators, and nobody has
shown it decide the sign at this size.

### What the novelty claim narrows to

Both searches, independently, leave the same three things unclaimed: the
per-model noise floor from 990 repetitions; BatchNorm statistics identified as
the sign-determining protocol for these indicators, with curvature splitting
the same way; and the hypernetwork setting, where nothing at all was found on
how to measure a network whose weights are generated rather than trained.

### Two reference corrections are now settled

Dohare et al. is confirmed as Nature 632:768-774, 2024, DOI
10.1038/s41586-024-07711-7. This closes the question left open on 21 November,
when the Nature page could not be reached. The journal version is the one to
cite, and its author list adds Qingfeng Lan, who is missing from the current
entry.

Adhikari et al. now has a journal version: Machine Learning 115:219, 2026, DOI
10.1007/s10994-026-07125-8, published 15 September 2026.

Both entries are now corrected in the Markdown, with the preprint kept as a
note on each so the arXiv identifier is still findable, and the two in-text
years for Adhikari moved from 2025 to 2026. `refs.bib` carries the same two
changes, under the same citation keys, so the LaTeX needs no edit. The docx was
rebuilt from the corrected source.

### One sentence has to come out or be backed

The abstract and Section 3.1 say that which set of BatchNorm statistics was
used is almost never reported. Neither search supports it, because neither
audited method sections. It is a claim about what other papers do and nobody
has looked. Either audit a named set of papers and report what each states, or
soften it to a statement about what we could not recover from the works we
tried.

### One warning in the reports is wrong

One report says Sokar et al. define dormancy relative to the maximum
activation, and warns that our normalisation differs. Checked against
`uncle/plasticity_indicators.py`: our score is a unit's mean absolute
activation divided by the mean over all units in the same site, which is
Sokar's definition, and the code already explains why a network-wide mean would
be wrong. The warning misdescribes the source. The only real difference is
where the reading is taken, which the 19 November audit already recorded.

Five of the most quotable new items are 2026 arXiv preprints with no venue, one
single-author, and both tools read only their abstracts. They are also the
items that most support the paper's position, so each needs reading before it
is cited.

### The code availability section now says where the paths point

The SciSpace review asked for the script and notebook names scattered through
Appendices A, D, E and I to be collected into one statement. That was already
done. What was missing is the repository itself: the section listed six paths
and never said what they were paths into. It now says the repository is
withheld for review and will be linked in the published version, so the names
are not dangling. The word "retired" in Appendix I became "earlier", because
retired is our word for a study, not a reader's.

Both notebooks named there, 53 and 55, are still in the repository. Nothing was
removed.

## Next decision

The reference corrections are done. What is left from the searches is the
Related Work section itself, the sentence about the reading protocol being
almost never reported, and reading the 2026 preprints before any of them is
cited. Both searches are kept at `ops-docs/related-work/` and the reading of
them at `ops-docs/related-work-reading-2026-11-23.html`.
