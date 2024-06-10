import numpy as np
import scipy.ndimage as ndi
import torch


def distance_transform(input_, class_vals, spacing_rw=(1., 1., 1.)):
    """Euclidean distance transform.

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        class_vals: (c, ) iterable
            Intensity values corresponding to classes.
        spacing_rw: 3-tuple
            Pixel spacing in real world units, one per each spatial dimension of `input_`.

    Returns:
        out: (b, d0, ..., dn) ndarray
            Thickness map for each class in each batch sample.
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
        dims = tuple(input_.size())[1:]
    else:
        num_samples = input_.shape[0]
        dims = input_.shape[1:]

    th_maps = np.zeros((num_samples, *dims), float)

    for sample_idx in range(num_samples):
        th_map = np.zeros_like(input_[sample_idx], float)

        for class_val in class_vals:
            sel_input_ = input_[sample_idx] == class_val

            th_map_class = ndi.distance_transform_edt(sel_input_, sampling=spacing_rw)

            th_map[sel_input_] = th_map_class[sel_input_]
        th_maps[sample_idx, :] = th_map

    return th_maps
