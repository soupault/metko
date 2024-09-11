import numpy as np
import torch


def apply_batched(func_score, input_, target, num_classes):
    """

    Args:
        func_score: func
            Function that derives scores from a pair of (d0, ..., dn) arrays.
        input_: (b, d0, ..., dn) ndarray or tensor
            First function argument.
        target: (b, d0, ..., dn) ndarray or tensor
            Second function argument.
        num_classes: int
            Total number of classes.

    Returns:
        scores: (b, ...) ndarray
            Results of applying `score_fn` sample-wise to (b, d0, ..., dn) data.
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = []
    for sample_idx in range(num_samples):
        scores.append(func_score(input_, target, num_classes))
    scores = np.asarray(scores)
    return scores
