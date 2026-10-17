# Plasticity After Task-unlearning in Hypernetworks (PATH)

A reimplementation of UnCLe, a continual-learning method where a hypernetwork
generates a classifier for each task, and tasks can be forgotten on request.
The open research question is whether unlearning a task restores the model's
ability to learn new ones, as the paper claims, and by what mechanism. A
second, supporting question is whether a forgotten task's information is gone
or merely hidden.

## Language

### Measuring a run

**Retain accuracy**:
Mean final accuracy across tasks that were seen and never forgotten.
_Avoid_: retention, preserved accuracy

**Forget accuracy**:
Mean final accuracy across tasks that were forgotten.
_Avoid_: unlearn accuracy

**Spill**:
The change in accuracy on other tasks caused by one forget request.
_Avoid_: collateral damage, interference

**Relapse**:
The change in a forgotten task's accuracy between the moment it was forgotten
and the end of the run.
_Avoid_: recovery, resurgence, rebound

### Measuring what survives a forget request

**Residue**:
Information about a forgotten task that is still present in the model after the
forget request. Distinct from Retain accuracy, which is about tasks that were
never forgotten.
_Avoid_: retained information, leakage, memory

**Probe**:
A measurement procedure that adapts a copy of the model on a forgotten task and
records how its accuracy moves. It never changes the model being measured.
_Avoid_: attack, recovery run, fine-tune

**Recovery advantage**:
How much better the model that forgot a task does under a probe than a matched
model that never learned it. The working measure of Residue.
_Avoid_: recovery gap, recovery score

**Unlearned condition**:
A model that learned a task and then forgot it. The subject of a probe.

**Never-learned control**:
A matched model that never learned the task, run under the same probe. The
baseline that Recovery advantage is measured against.
_Avoid_: baseline model, clean model, fresh model

### Attributing residue

**Component**:
One named group of the model's state, treated as a single unit for attribution.
The five are Task code, Chunk codes, Shared generator, Output heads, and
Normalization buffers.
_Avoid_: module, part, layer

**Task code**:
The learned vector that identifies one task to the hypernetwork.
_Avoid_: task embedding, task vector, task ID

**Shared generator**:
The part of the hypernetwork that every task passes through.
_Avoid_: trunk, backbone, body

**Normalization buffers**:
The per-task BatchNorm running statistics, which live outside the
hypernetwork and are not changed by a forget request.
_Avoid_: BN stats, running stats

**Ablation**:
Replacing one Component's values with donor values chosen to mean "as if this
task had never been learned", in order to see how much Recovery advantage
depends on it.
_Avoid_: intervention, knockout, reset

**Component share**:
The fraction of Recovery advantage that disappears when one Component is
ablated. The attribution number.
_Avoid_: contribution, importance, attribution score

**Probe sensitivity**:
Whether a probe can detect Residue at all, established by running it on a case
where the answer is known in advance.
_Avoid_: probe power, validity
