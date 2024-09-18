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
    image = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.distance_transform(image, class_vals=class_vals)


@pytest.mark.parametrize("fn", (
    metko.avg_surf_dist,
    metko.avg_symm_surf_dist,
    metko.rms_symm_surf_dist
))
@pytest.mark.parametrize("num_classes", (1, 3))
@pytest.mark.parametrize("spacing_rw", ((1., 1., 1.), (.1, .1, .1)))
def test_surface(fn, num_classes, spacing_rw):
    ndim = 3
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = fn(input_=input_, target=target, num_classes=num_classes,
           spacing_rw=spacing_rw)


@pytest.mark.parametrize("num_classes", (1, 3))
@pytest.mark.parametrize("spacing_rw", ((1., 1., 1.), (.1, .1, .1)))
@pytest.mark.parametrize("percent", (50, 100))
def test_robust_hausdorff_dist(num_classes, spacing_rw, percent):
    ndim = 3
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.robust_hausdorff_dist(input_=input_, target=target,
                                    num_classes=num_classes,
                                    spacing_rw=spacing_rw, percent=percent)


@pytest.mark.parametrize("fn", (
    metko.surf_dice_at_tol,
    metko.surf_overlap_at_tol
))
@pytest.mark.parametrize("num_classes", (1, 3))
@pytest.mark.parametrize("spacing_rw", ((1., 1., 1.), (.1, .1, .1)))
@pytest.mark.parametrize("tolerance_rw", (0.9, 1.0))
def test_surface_at_tol(fn, num_classes, spacing_rw, tolerance_rw):
    ndim = 3
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = fn(input_=input_, target=target, num_classes=num_classes,
           spacing_rw=spacing_rw, tolerance_rw=tolerance_rw)


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("num_classes", (1, 2, 3))
def test_cm(ndim, num_classes):
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.confusion_matrix(input_=input_, target=target,
                               num_classes=num_classes)


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
    class_vals = list(range(num_classes))
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = fn(input_=input_, target=target, num_classes=num_classes)


@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_volume_error(ndim):
    class_vals = (0, 1, 2)
    input_ = np.random.choice(class_vals, size=(30,) * ndim)
    target = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.volume_error(input_=input_, target=target, class_vals=class_vals)


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("spacing_rw", ((1., 1., 1.), (.1, .1, .1)))
def test_volume_total(ndim, spacing_rw):
    class_vals = (0, 1, 2)
    image = np.random.choice(class_vals, size=(30,) * ndim)

    _ = metko.volume_total(input_=image, class_vals=class_vals,
                           spacing_rw=spacing_rw)


# TODO: add tests for local_thickness - exec
# TODO: add tests for local_thickness - numerical
