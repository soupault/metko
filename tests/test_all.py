import numpy as np
import pytest

import metko


# def test_bland_altman():
#     m1 = [1, 2, 3, 4, 5]
#     m2 = [2, 3, 4, 5, 6]
#     res = bland_altman_naive(m1, m2)
#     assert res["mean_diff"] == 1


# @pytest.parametrize("ndim", [2, 3, 4])
# def test_distance_transform(ndim):
#     image = np.random.randint(low=0, high=3, size=(100, ) * ndim)
#     res = metko.distance_transform(image, class_vals=(0, 1, 2))
#     assert res.shape == image.shape

# TODO: surface distance
