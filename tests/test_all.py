import numpy as np
import pytest

import metko


@pytest.mark.parametrize("diff_mode", ["unit", "percentage"])
def test_bland_altman(diff_mode):
    m1 = [1, 2, 3, 4, 5]
    m2 = [2, 3, 4, 5, 6]

    _ = metko.bland_altman(m1, m2, diff_mode=diff_mode)


@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_distance_transform(ndim):
    class_vals = (0, 1, 2)
    # num_classes = len(class_vals)
    image = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.distance_transform(image, class_vals=class_vals)


# TODO: surface distance:
# "avg_surf_dist",
# "avg_symm_surf_dist",
# "rms_symm_surf_dist",
# "robust_hausdorff_dist",
# "surf_dice_at_tol",
# "surf_overlap_at_tol",


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("num_classes", (1, 2, 3))
def test_volume_scores(ndim, num_classes):
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.confusion_matrix(input_=input_, target=target, num_classes=num_classes)


@pytest.mark.parametrize("fn", (
    metko.dice_score,
    metko.jaccard_score,
    metko.precision_score,
    metko.recall_score,
    metko.sensitivity_score,
    metko.specificity_score,
    metko.volume_similarity
))
@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_volume_scores(fn, ndim):
    num_classes = 3
    input_ = np.random.randint(low=0, high=num_classes, size=(30,) * ndim)
    target = np.random.randint(low=0, high=num_classes, size=(30,) * ndim)

    _ = fn(input_=input_, target=target, num_classes=num_classes)


@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_volume_scores(ndim):
    class_vals = (0, 1, 2)
    num_classes = len(class_vals)
    input_ = np.random.randint(low=0, high=num_classes, size=(30,) * ndim)
    target = np.random.randint(low=0, high=num_classes, size=(30,) * ndim)

    _ = metko.volume_error(input_=input_, target=target, class_vals=class_vals)


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("spacing_rw", ((1., 1., 1.), (.1, .1, .1)))
def test_volume_total(ndim, spacing_rw):
    class_vals = (0, 1, 2)
    num_classes = len(class_vals)

    image = np.random.randint(low=0, high=num_classes, size=(30,) * ndim)
    _ = metko.volume_total(input_=image, class_vals=class_vals, spacing_rw=spacing_rw)
