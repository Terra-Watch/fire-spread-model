"""Scores for a forecast of where a fire will grow. Truth is the cells of newly detected pixels; everything else in the candidate area counts as 'not seen growing',
which is NOT the same as 'did not burn' (small fires are missed, clouds hide), so the scores ask a ranking question rather than a calibration one:
do the cells that were seen growing sit higher in the forecast than the others?"""
import numpy as np
from scipy.stats import rankdata


def auc(score, label) -> float:
    """Probability that a growing cell outranks a non-growing one (ties count half). NaN when one of the two groups is empty."""
    score, label = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def area_for_capture(score, label, capture: float = 0.5) -> float:
    """Share of the candidate area one must take, highest score first, to contain `capture` of the growing cells. Lower is better; 0.5 would be no skill. Ties are taken together (average)."""
    score, label = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel()
    n1 = int(label.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, l = score[order], label[order]
    need = capture * n1
    cum = np.cumsum(l)
    k = int(np.searchsorted(cum, need, side="left"))                    # first index reaching the target
    # extend to the end of the tie group, then average within it: the expected area under random tie-breaking
    end = k
    while end + 1 < len(s) and s[end + 1] == s[k]:
        end += 1
    start = k
    while start > 0 and s[start - 1] == s[k]:
        start -= 1
    before_pos = cum[start - 1] if start > 0 else 0
    group_pos = int(cum[end] - before_pos)
    frac = (need - before_pos) / max(group_pos, 1)
    return float((start + frac * (end - start + 1)) / len(s))


def bootstrap_mean(values, n: int = 2000, seed: int = 0):
    """Mean and 95 % interval of a list of per-unit values (cases, or better, blocks) by resampling the units."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auc_within_bins(score, label, dist, edges, min_pos: int = 3, min_neg: int = 20) -> float:
    """Direction skill: the AUC computed separately among cells at about the same distance from the fire (so distance cannot help), averaged over the distance bins
    weighted by how many growing cells each holds. NaN when no bin has enough of both."""
    score, label, dist = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel(), np.asarray(dist, float).ravel()
    total, weight = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        pos, neg = int(label[m].sum()), int((~label[m]).sum())
        if pos < min_pos or neg < min_neg:
            continue
        a = auc(score[m], label[m])
        total += pos * a
        weight += pos
    return float(total / weight) if weight else float("nan")


def iou_at_true_area(score, truth) -> float:
    """Overlap of the k highest-scored cells with the truth, k = the number of truth cells (so the predicted and true areas are equal and the choice of cut-off is not the model's to make).
    Intersection over union = hit / (2k - hit). 1 is a perfect footprint, a random pick gives about k / cells."""
    score, truth = np.asarray(score), np.asarray(truth, bool)
    k = int(truth.sum())
    if k == 0:
        return float("nan")
    top = np.argpartition(-score, k - 1)[:k]
    hit = int(truth[top].sum())
    return hit / (2 * k - hit)
