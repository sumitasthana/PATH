"""The continual-learning metrics the adapter plasticity study reports.

Every number here comes from one accuracy matrix per run. A[i][j] is the
accuracy on task j after the stream has trained through task i, with i and j
counted by position in the learning order rather than by task name.

    task position      0     1     2    ...
    after task 0     A00     -     -
    after task 1     A10   A11     -
    after task 2     A20   A21   A22

Only the lower triangle exists. A task has no accuracy before it is learned.

The four summary numbers follow `reference/adapter-plasticity-study`, which
takes them from the continual-learning literature it cites:

  Final accuracy      mean accuracy over every task at the end of the stream
  Learning accuracy   mean of the diagonal: how well each task was learned at
                      the moment it was learned. The adapter study calls this
                      a direct within-stream measure of plasticity
  Forgetting          mean drop from each task's best score to its last
  Backward transfer   mean change from each task's first score to its last

The diagonal on its own is the new-task accuracy curve. A falling curve means
each new task is learned less well than the one before, which is plasticity
loss measured inside the stream rather than by a separate probe.

One difference from the adapter study, which matters when the two are compared.
That study reports every metric twice, once class-incremental and once
task-incremental, because its model has a single head over all classes. This
framework has no such head: a task code generates a classifier with ten
outputs, and the code has to be supplied to get any prediction at all. So only
the task-incremental form exists here. The adapter study calls that form the
cleaner read, so this is the comparable one, but it is not a free choice.
"""

import math
import random


def accuracy_matrix(history, order):
    """Build A[i][j] from a saved learning history.

    history   the saved request list. Each row needs "action", "task" and
              "after", where "after" maps task name to accuracy.
    order     task names in the order the stream learned them.

    Only "learn" requests are used. A stream containing a forget request is
    refused, because these four metrics are defined for a learning stream and
    a removed task would be counted as catastrophic forgetting.

    Returns a square list of lists. Entries above the diagonal are None.
    """
    rows = [row for row in history if row.get("action") == "learn"]
    if len(rows) != len(history):
        raise ValueError("This history contains non-learning requests. "
                         "These metrics are defined for a learning stream.")
    if [row["task"] for row in rows] != list(order):
        raise ValueError("History task sequence does not match the given order.")

    size = len(order)
    matrix = [[None] * size for _ in range(size)]
    for i, row in enumerate(rows):
        for j, task in enumerate(order):
            if j > i:
                continue
            if task not in row["after"]:
                raise ValueError(f"Request {i} has no accuracy for task {task}.")
            matrix[i][j] = float(row["after"][task])
    return matrix


def _check(matrix):
    size = len(matrix)
    if size < 2:
        raise ValueError("Need at least two tasks.")
    for i, row in enumerate(matrix):
        if len(row) != size or any(row[j] is None for j in range(i + 1)):
            raise ValueError(f"Row {i} is missing an accuracy at or below the diagonal.")
    return size


def diagonal(matrix):
    """Each task's accuracy at the moment it was learned: the A(t,t) curve."""
    return [matrix[i][i] for i in range(_check(matrix))]


def final_accuracy(matrix):
    """Mean accuracy over every task after the last one is learned."""
    size = _check(matrix)
    return sum(matrix[size - 1]) / size


def learning_accuracy(matrix):
    """Mean of the diagonal. Higher means new tasks are learned well."""
    values = diagonal(matrix)
    return sum(values) / len(values)


def forgetting(matrix):
    """Mean drop from each task's best score to its score at the end.

    The best is taken over the rows from the task's own up to but not
    including the last, so a task is compared against its own peak rather
    than against the final row it appears in.
    """
    size = _check(matrix)
    drops = [max(matrix[i][j] for i in range(j, size - 1)) - matrix[size - 1][j]
             for j in range(size - 1)]
    return sum(drops) / len(drops)


def backward_transfer(matrix):
    """Mean change from each task's first score to its last. Negative is loss."""
    size = _check(matrix)
    changes = [matrix[size - 1][j] - matrix[j][j] for j in range(size - 1)]
    return sum(changes) / len(changes)


def slope(values):
    """Least-squares slope of values against position, in units per step.

    Fitting a straight line to the diagonal summarises whether new tasks are
    being learned better or worse as the stream goes on. A negative slope
    means worse.
    """
    count = len(values)
    if count < 2:
        raise ValueError("Need at least two points to fit a line.")
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    spread = sum((x - mean_x) ** 2 for x in range(count))
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    return covariance / spread


def summary(matrix):
    """All five numbers for one run."""
    return {
        "tasks": _check(matrix),
        "final_accuracy": final_accuracy(matrix),
        "learning_accuracy": learning_accuracy(matrix),
        "forgetting": forgetting(matrix),
        "backward_transfer": backward_transfer(matrix),
        "learning_curve_slope_per_task": slope(diagonal(matrix)),
        "learning_curve_slope_per_10_tasks": 10 * slope(diagonal(matrix)),
        "diagonal": diagonal(matrix),
    }


def mean_and_sd(values):
    """Mean and sample standard deviation, as the adapter study reports them."""
    values = list(values)
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance)


def bootstrap_interval(values, *, resamples=10000, seed=12345, level=0.95):
    """A percentile bootstrap interval for the mean, over seeds.

    The adapter study fixes 10,000 resamples and a 95% interval, so those are
    the defaults here.

    A bootstrap resamples the seeds with replacement many times and takes the
    mean each time, then reads the interval off those means. With only a
    handful of seeds the interval is wide and its ends fall on a few distinct
    values, so it says little on its own. Report the seed count beside it.
    """
    values = list(values)
    if not values:
        raise ValueError("Need at least one value.")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(sum(generator.choices(values, k=count)) / count
                   for _ in range(resamples))
    low = means[int((1 - level) / 2 * resamples)]
    high = means[min(int((1 + level) / 2 * resamples), resamples - 1)]
    return low, high


def across_seeds(matrices, *, resamples=10000, seed=12345):
    """Per-seed summaries plus the mean, standard deviation and interval.

    matrices   one accuracy matrix per seed, keyed by seed.
    """
    per_seed = {name: summary(matrix) for name, matrix in matrices.items()}
    names = sorted(per_seed)
    keys = ["final_accuracy", "learning_accuracy", "forgetting", "backward_transfer",
            "learning_curve_slope_per_task", "learning_curve_slope_per_10_tasks"]
    combined = {}
    for key in keys:
        values = [per_seed[name][key] for name in names]
        mean, sd = mean_and_sd(values)
        low, high = bootstrap_interval(values, resamples=resamples, seed=seed)
        combined[key] = {"mean": mean, "sd": sd, "values": values,
                         "ci_low": low, "ci_high": high}
    return {"seeds": names, "n_seeds": len(names),
            "per_seed": per_seed, "across_seeds": combined}
