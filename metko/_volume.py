"""Many functions below are modified from
https://github.com/ternaus/robot-surgery-segmentation"""
import torch
import numpy as np


def confusion_matrix(input_, target, num_classes):
    """Construct a confusion matrix from a pair of arrays.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
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
    return cm.astype(np.int64)


def dice_score(input_, target, num_classes):
    """Dice score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.
    Returns:
        out: (c, ) list
            Class-wise Dice scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def jaccard_score(input_, target, num_classes):
    """Jaccard score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.
    Returns:
        out: (c, ) list
            Class-wise IoU scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def precision_score(input_, target, num_classes):
    """Precision score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.

    Returns:
        out: (d, ) list
            Class-wise precision scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def recall_score(input_, target, num_classes):
    """Recall score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.
    
    Returns:
        out: (d, ) list
            Class-wise recall scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def sensitivity_score(input_, target, num_classes):
    """Sensitivity score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.

    Returns:
        out: (d, ) list
            Class-wise sensitivity scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def specificity_score(input_, target, num_classes):
    """Specificity score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.

    Returns:
        out: (d, ) list
            Class-wise sensitivity scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def volume_similarity(input_, target, num_classes):
    """Volume similarity score.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        num_classes: int
            Total number of classes.

    Returns:
        out: (d, ) list
            Class-wise volumetric similarity scores.
    """
    cm = confusion_matrix(input_=input_, target=target, num_classes=num_classes)

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


def volume_error(input_, target, class_vals):
    """Volume error score.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor of int
        target: (d0, ..., dn) ndarray or torch.Tensor of int
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.

    Returns:
        out: (c, ) ndarray
    """
    def _ve(i, t):
        numer = 2 * np.count_nonzero(np.bitwise_xor(i, t))
        denom = np.count_nonzero(i) + np.count_nonzero(t)
        if denom == 0:
            return 0
        else:
            return numer / denom

    num_classes = len(class_vals)
    scores = np.zeros((num_classes, ), dtype=float)
    for class_idx, class_val in enumerate(class_vals):
        sel_input_ = input_ == class_val
        sel_target = target == class_val

        scores[class_idx] = _ve(sel_input_, sel_target)
    return scores


def volume_total(input_, class_vals, spacing_rw=(1., 1., 1.), mode="straight"):
    """Total volume.

    Args:
        input_: (d0, ..., dn) ndarray or tensor
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.
        spacing_rw: 3-tuple
            Pixel spacing in real world units, one per each spatial dimension of `input_`.
        mode: {"straight", "subpix"}

    Returns:
        out: (c, ) ndarray
            Total volume for each class in each batch sample.
    """
    def _v_straight(i, s):
        # Naive approach
        n = np.count_nonzero(i)
        vox = np.prod(s)
        return vox * n

    def _v_subpix(i, s):
        # TODO: implement via surface elements volume (more accurate). See `surface`
        raise NotImplementedError()

    num_classes = len(class_vals)
    scores = np.zeros((num_classes, ), dtype=float)
    for class_idx, class_val in enumerate(class_vals):
        sel_input_ = input_ == class_val

        if mode == "straight":
            scores[class_idx] = _v_straight(sel_input_, spacing_rw)
        else:
            raise NotImplementedError(f"Mode `{mode}` is not supported")
    return scores
