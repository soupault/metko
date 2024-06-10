"""Many functions below are modified from
https://github.com/ternaus/robot-surgery-segmentation"""
import torch
import numpy as np


def confusion_matrix(input_, target, num_classes):
    """

    Args:
        input_: (d0, ..., dn) ndarray or tensor
        target: (d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (num_classes, num_classes) ndarray
            Confusion matrix.
    """
    if torch.is_tensor(input_):
        input_ = input_.detach().to("cpu").numpy()
    if torch.is_tensor(target):
        target = target.detach().to("cpu").numpy()

    replace_indices = np.vstack((
        target.flatten(),
        input_.flatten())
    ).T
    cm, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes-1), (0, num_classes-1)]
    )
    return cm.astype(np.uint32)


def dice_score_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class Dice scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 2 * float(true_positives) / denom
        scores.append(score)
    return scores


def jaccard_score_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class IoU scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def precision_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class precision scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        denom = true_positives + false_positives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def recall_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class recall scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def sensitivity_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class sensitivity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_negatives = cm[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_positives) / denom
        scores.append(score)
    return scores


def specificity_from_cm(cm):
    """

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class sensitivity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        true_negatives = np.trace(cm) - true_positives
        false_positives = cm[:, index].sum() - true_positives
        denom = false_positives + true_negatives
        if denom == 0:
            score = 0
        else:
            score = float(true_negatives) / denom
        scores.append(score)
    return scores


def volume_similarity_from_cm(cm):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/

    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class volumetric similarity scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 1 - abs(false_negatives - false_positives) / denom
        scores.append(score)
    return scores


def _template_score(func_score_from_cm, input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        cm = confusion_matrix(input_=input_[sample_idx],
                              target=target[sample_idx],
                              num_classes=num_classes)
        scores[sample_idx, :] = func_score_from_cm(cm)
    return scores


def dice_score(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(dice_score_from_cm, input_, target, num_classes)


def jaccard_score(input_, target, num_classes):
    """Jaccard similarity score, also known as Intersection-over-Union (IoU).

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(jaccard_score_from_cm, input_, target, num_classes)


def precision_score(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(precision_from_cm, input_, target, num_classes)


def recall_score(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(recall_from_cm, input_, target, num_classes)


def sensitivity_score(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(sensitivity_from_cm, input_, target, num_classes)


def specificity_score(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(specificity_from_cm, input_, target, num_classes)


def volume_similarity(input_, target, num_classes):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (b, c) ndarray
    """
    return _template_score(volume_similarity_from_cm, input_, target, num_classes)


def volume_error(input_, target, class_vals):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor of int
        target: (b, d0, ..., dn) ndarray or tensor of int
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.

    Returns:
        out: (b, c) ndarray
    """

    def _ve(i, t):
        numer = 2 * np.count_nonzero(np.bitwise_xor(i, t))
        denom = np.count_nonzero(i) + np.count_nonzero(t)
        if denom == 0:
            return 0
        else:
            return numer / denom

    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    num_classes = len(class_vals)
    scores = np.zeros((num_samples, num_classes), dtype=float)
    for sample_idx in range(num_samples):
        for class_idx, class_val in enumerate(class_vals):
            sel_input_ = input_[sample_idx] == class_val
            sel_target = target[sample_idx] == class_val

            scores[sample_idx, class_idx] = _ve(sel_input_, sel_target)
    return np.squeeze(scores)


def volume_total(input_, class_vals, spacing_rw=(1, 1, 1), mode="straight"):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.
        spacing_rw: 3-tuple
            Pixel spacing in real world units, one per each spatial dimension of `input_`.
        mode: {"straight", "subpix"}

    Returns:
        out: (b, c) ndarray
            Total volume for each class in each batch sample.
    """
    def _v_straight(i, s):
        n = np.count_nonzero(i)
        vox = np.prod(s)
        # Naive approach
        return vox * n

    def _v_subpix(i, s):
        # TODO: implement via surface elements volume (more accurate). See `surface`
        raise NotImplementedError()

    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    num_classes = len(class_vals)
    scores = np.zeros((num_samples, num_classes), dtype=float)
    for sample_idx in range(num_samples):
        for class_idx, class_val in enumerate(class_vals):
            sel_input_ = input_[sample_idx] == class_val

            if mode == "straight":
                scores[sample_idx, class_idx] = _v_straight(sel_input_, spacing_rw)
            else:
                # TODO:
                raise ValueError()

    return scores
