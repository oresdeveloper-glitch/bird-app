import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Any


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    normalize: bool = False,
) -> Dict[str, Any]:
    """Build a confusion matrix.

    Args:
        y_true: ground-truth label ids.
        y_pred: predicted label ids.
        labels: ordered set of all label ids to include.
        normalize: if True, normalize rows (per true label).

    Returns JSON-serializable dict.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = [[0 for _ in range(n)] for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        if t not in label_to_idx or p not in label_to_idx:
            # skip unknown labels
            continue
        mat[label_to_idx[t]][label_to_idx[p]] += 1

    if normalize:
        norm_mat = []
        for i in range(n):
            row_sum = sum(mat[i])
            if row_sum == 0:
                norm_mat.append([0.0 for _ in range(n)])
            else:
                norm_mat.append([mat[i][j] / row_sum for j in range(n)])
        mat_out = norm_mat
    else:
        mat_out = mat

    total = sum(sum(row) for row in mat)
    return {
        "labels": list(labels),
        "matrix": mat_out,
        "normalize": normalize,
        "total": total,
    }


def top_misclassifications(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    top_k: int = 25,
) -> List[Dict[str, Any]]:
    """Return top (true->pred) pairs with highest counts, excluding correct predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    counts = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        counts[(t, p)] += 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in ranked
    ]

