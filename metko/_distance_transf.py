import numpy as np
import scipy.ndimage as ndi
import torch


def distance_transform(input_, class_vals, spacing_rw=None):
    """Euclidean distance transform.

    Args:
        input_: (d0, ..., dn) ndarray or torch.Tensor
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.
        spacing_rw: n-tuple
            Pixel spacing in real world units, one per each spatial dimension of `input_`.

    Returns:
        out: (d0, ..., dn) ndarray
            Distance transform for each class in each batch sample.
    """
    if torch.is_tensor(input_):
        dims = tuple(input_.size())
    else:
        dims = input_.shape
    spacing_rw = spacing_rw or (1.,) * len(dims)

    dt_map = np.zeros_like(input_, float)
    for class_val in class_vals:
        sel_input_ = input_ == class_val
        dt_map_class = ndi.distance_transform_edt(sel_input_, sampling=spacing_rw)
        dt_map[sel_input_] = dt_map_class[sel_input_]

    return dt_map
