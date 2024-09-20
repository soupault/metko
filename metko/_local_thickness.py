from warnings import warn

import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
from numba import jit, prange
import torch


@jit(nopython=True)
def _lt_from_ma_dt_2d(mask, med_axis, distance, spacing_rw, search_extent):
    out = np.zeros_like(mask, dtype=np.float32)

    nonzero_x = np.nonzero(mask)
    ii = nonzero_x[0]
    jj = nonzero_x[1]
    num_pts_mask = len(ii)

    nonzero_med_axis = np.nonzero(med_axis)
    mm = nonzero_med_axis[0]
    nn = nonzero_med_axis[1]
    num_pts_med_axis = len(mm)

    for e in range(num_pts_mask):
        i = ii[e]
        j = jj[e]

        best_val = 0

        if search_extent is not None:
            r0 = max(i - search_extent[0], 0)
            r1 = min(i + search_extent[0], mask.shape[0] - 1)
            c0 = max(j - search_extent[1], 0)
            c1 = min(j + search_extent[1], mask.shape[1] - 1)

        for w in range(num_pts_med_axis):
            m = mm[w]
            n = nn[w]

            if search_extent is not None:
                if m < r0 or m > r1 or n < c0 or n > c1:
                    continue

            dist_curr = ((spacing_rw[0] * (i - m)) ** 2 +
                         (spacing_rw[1] * (j - n)) ** 2)
            if dist_curr <= (distance[m, n] ** 2):
                if distance[m, n] > best_val:
                    best_val = distance[m, n]

        out[i, j] = best_val
    return out


@jit(nopython=True, parallel=True)
def _lt_from_ma_dt_3d(mask, med_axis, distance, spacing_rw, search_extent):
    out = np.zeros_like(mask, dtype=np.float32)

    nonzero_mask = np.nonzero(mask)
    ii = nonzero_mask[0]
    jj = nonzero_mask[1]
    kk = nonzero_mask[2]
    num_pts_mask = len(ii)

    nonzero_med_axis = np.nonzero(med_axis)
    mm = nonzero_med_axis[0]
    nn = nonzero_med_axis[1]
    oo = nonzero_med_axis[2]
    num_pts_med_axis = len(mm)

    best_vals = np.zeros((num_pts_mask, ), dtype=np.float32)

    for e in prange(num_pts_mask):
        i = ii[e]
        j = jj[e]
        k = kk[e]

        if search_extent is not None:
            r0 = max(i - search_extent[0], 0)
            r1 = min(i + search_extent[0], mask.shape[0] - 1)
            c0 = max(j - search_extent[1], 0)
            c1 = min(j + search_extent[1], mask.shape[1] - 1)
            p0 = max(k - search_extent[2], 0)
            p1 = min(k + search_extent[2], mask.shape[2] - 1)

        for w in range(num_pts_med_axis):
            m = mm[w]
            n = nn[w]
            o = oo[w]

            if search_extent is not None:
                if m < r0 or m > r1 or n < c0 or n > c1 or o < p0 or o > p1:
                    continue

            dist_curr = ((spacing_rw[0] * (i - m)) ** 2 +
                         (spacing_rw[1] * (j - n)) ** 2 +
                         (spacing_rw[2] * (k - o)) ** 2)
            if dist_curr <= (distance[m, n, o] ** 2):
                if distance[m, n, o] > best_vals[e]:
                    best_vals[e] = distance[m, n, o]

        out[i, j, k] = best_vals[e]
    return out


def _lt_sphere_fitting(mask, spacing_rw):
    """Compute the local thickness of a binary image.

    Args:
        mask: (d0, d1[, d2]) ndarray of bool
            Mask where object is marked with `True`, background - with `False`.
        spacing_rw: tuple of ``mask.ndim`` elements
            Size of ``mask`` voxels in real world units. Defaults to 1.0 for
            each dimension of ``mask``.
    Returns:
        out: (d0, d1[, d2]) ndarray of float
            Local thickness map. Has the same shape as `mask`.
    """
    # Step 1: Compute the Euclidean distance transform
    distance_map = ndi.distance_transform_edt(mask, sampling=spacing_rw)

    # Step 2: Initialize a local thickness map (same shape as the mask)
    thickness = np.zeros_like(distance_map)

    coords = [np.arange(s) for s in mask.shape]
    meshgrid = np.meshgrid(*coords, indexing='ij')

    # Step 3: Sort the voxels by decreasing distance (largest spheres first)
    sorted_indices = np.argsort(-distance_map.ravel())  # sort by descending distance
    sorted_indices = np.unravel_index(sorted_indices,
                                      mask.shape)  # convert to multi-dimensional indices

    # Step 4: For each voxel, propagate the thickness from the largest sphere
    for idx in zip(*sorted_indices):
        radius = distance_map[idx]
        if radius > 0:
            # Compute squared distance to avoid sqrt
            squared_dist = np.sum(
                [((grid - idx_dim) * spacing_dim) ** 2
                 for grid, idx_dim, spacing_dim in zip(meshgrid, idx, spacing_rw)],
                axis=0)

            # Create mask and update thickness
            mask = squared_dist <= radius ** 2

            # Propagate the sphere's diameter to all points in the mask
            thickness[mask] = np.maximum(thickness[mask], 2 * radius)

    return thickness


def local_thickness_base(mask, *, algorithm_2d=None, algorithm_3d=None,
                         spacing_rw=None, stack_axis=None,
                         thickness_max_rw=None,
                         return_med_axis=False, return_distance=False):
    """
    Inspired by https://imagej.net/Local_Thickness .

    Args:
        mask: (D0, D1[, D2]) ndarray
        algorithm_2d: {"med2d_dist2d_lth2d", "sphere_fitting"} or None
            Implementation algorithm  for 2D ``mask``. Ignored for 3D.
        algorithm_3d: {"skel3d_dist3d_lth3d", "stacked_2d",
                      "med2d_dist2d_lth3d", "med2d_dist3d_lth3d",
                      "sphere_fitting"} or None
            Implementation algorithm  for 3D ``mask``. Ignored for 2D.
        spacing_rw: tuple of ``mask.ndim`` elements
            Size of ``mask`` voxels in real world units. Defaults to 1.0 for
            each dimension of ``mask``.
        stack_axis: None or int
            Index of axis to perform slice selection along. Ignored for 2D.
        thickness_max_rw: None or float
            Hypothesised maximum thickness in real world units.
            Used to constrain local ROIs to speed up best candidate search.
        return_med_axis: bool
            Whether to return the medial axis. Ignored for ``sphere_fitting``.
        return_distance: bool
            Whether to return the distance transform. Ignored for ``sphere_fitting``.

    Returns:
        out: ndarray
            Local thickness.
        med_axis: ndarray
            Medial axis. Returned only if ``return_med_axis`` is True.
        distance: ndarray
            Distance transform. Returned only if ``return_distance`` is True.
    """
    # For medial thinning-based algorithms, the workflow as follows:
    # 1. Compute the distance transform
    # 2. Find the distance ridge (/ exclude the redundant points)
    # 3. Compute local thickness

    if spacing_rw is None:
        spacing_rw = (1.,) * mask.ndim
    spacing_rw = np.asarray(spacing_rw)

    if thickness_max_rw is None:
        search_extent = None
    else:
        # Size of the neighborhood in voxels where to look for the sphere centers
        radius_max_rw = thickness_max_rw / 2.
        search_extent = np.ceil(radius_max_rw / spacing_rw).astype(np.int32)

    if mask.ndim == 2:
        if algorithm_2d == "med2d_dist2d_lth2d":
            med_axis = morphology.medial_axis(mask)
            distance = ndi.distance_transform_edt(mask, sampling=spacing_rw)
            out = _lt_from_ma_dt_2d(mask=mask, med_axis=med_axis, distance=distance,
                                    spacing_rw=spacing_rw, search_extent=search_extent)
            out = 2. * out  # Thickness is twice the distance to the closest surface point

        elif algorithm_2d == "sphere_fitting":
            if thickness_max_rw is not None:
                msg = f"`thickness_max_rw` is not supported in algorithm `{algorithm_2d}`"
                raise NotImplementedError(msg)
            med_axis, distance = None, None
            out = _lt_sphere_fitting(mask=mask, spacing_rw=spacing_rw)

        else:
            raise ValueError(f"Invalid algorithm: `{algorithm_2d}`")

    elif mask.ndim == 3:
        if algorithm_3d == "skel3d_dist3d_lth3d":
            msg = "Local thickness based on straight skeleton is only an approximation"
            warn(msg)
            if thickness_max_rw is not None:
                msg = f"`thickness_max_rw` is not supported in algorithm `{algorithm_3d}`"
                raise NotImplementedError(msg)
            search_extent = None

            skeleton = morphology.skeletonize_3d(mask)
            distance = ndi.distance_transform_edt(mask, sampling=spacing_rw)
            out = _lt_from_ma_dt_3d(mask=mask, med_axis=skeleton, distance=distance,
                                    spacing_rw=spacing_rw, search_extent=search_extent)
            med_axis = skeleton
            out = 2. * out  # Thickness is twice the distance to the closest surface point

        elif algorithm_3d == "stacked_2d":
            if thickness_max_rw is not None:
                msg = f"`thickness_max_rw` is not supported in algorithm `{algorithm_3d}`"
                raise NotImplementedError(msg)
            assert stack_axis in range(mask.ndim), "`stack_axis` must be a valid dimension index"

            acc_out = []
            acc_med = []
            acc_dist = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                sel_mask = mask[sel_idcs]
                sel_spacing_rw = (list(spacing_rw[:stack_axis]) +
                                  list(spacing_rw[stack_axis+1:]))
                sel_res = local_thickness_base(sel_mask, algorithm_2d="med2d_dist2d_lth2d",
                                               spacing_rw=sel_spacing_rw,
                                               thickness_max_rw=thickness_max_rw,
                                               return_med_axis=True, return_distance=True)
                acc_out.append(sel_res[0])
                acc_med.append(sel_res[1])
                acc_dist.append(sel_res[2])

            out = np.stack(acc_out, axis=stack_axis)  # Is already twice the distance, no need to x2
            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = np.stack(acc_dist, axis=stack_axis)

        elif algorithm_3d == "med2d_dist2d_lth3d":
            if thickness_max_rw is not None:
                msg = f"`thickness_max_rw` is not supported in algorithm `{algorithm_3d}`"
                raise NotImplementedError(msg)
            search_extent = None
            assert stack_axis in range(mask.ndim), "`stack_axis` must be a valid dimension index"

            acc_med = []
            acc_dist = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                sel_mask = mask[sel_idcs]
                sel_spacing_rw = (list(spacing_rw[:stack_axis]) +
                                  list(spacing_rw[stack_axis + 1:]))
                sel_med = morphology.medial_axis(sel_mask)
                sel_dist = ndi.distance_transform_edt(sel_mask, sampling=sel_spacing_rw)
                acc_med.append(sel_med)
                acc_dist.append(sel_dist)

            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = np.stack(acc_dist, axis=stack_axis)

            out = _lt_from_ma_dt_3d(mask=mask, med_axis=med_axis, distance=distance,
                                    spacing_rw=spacing_rw, search_extent=search_extent)
            out = 2. * out  # Thickness is twice the distance to the closest surface point

        elif algorithm_3d == "med2d_dist3d_lth3d":
            assert stack_axis in range(mask.ndim), "`stack_axis` must be a valid dimension index"
            acc_med = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                sel_res = morphology.medial_axis(mask[sel_idcs])
                acc_med.append(sel_res)

            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = ndi.distance_transform_edt(mask, sampling=spacing_rw)
            out = _lt_from_ma_dt_3d(mask=mask, med_axis=med_axis, distance=distance,
                                    spacing_rw=spacing_rw, search_extent=search_extent)
            out = 2. * out  # Thickness is twice the distance to the closest surface point

        elif algorithm_3d == "sphere_fitting":
            med_axis, distance = None, None
            out = _lt_sphere_fitting(mask=mask, spacing_rw=spacing_rw)

        else:
            raise ValueError(f"Invalid algorithm: {algorithm_3d}")

    else:
        msg = "Only 2D and 3D arrays are supported"
        raise ValueError(msg)

    if return_med_axis:
        if return_distance:
            return out, med_axis, distance
        else:
            return out, med_axis
    else:
        if return_distance:
            return out, distance
        else:
            return out


def local_thickness(input_, num_classes, stack_axis, spacing_rw=(1., 1., 1.),
                    skip_classes=None):
    """

    Args:
        input_: (d0, d1[, d2]) ndarray or torch.Tensor
        num_classes: int
            Total number of classes.
        stack_axis: int
            Index of axis to perform slice selection along. Ignored for 2D.
        spacing_rw: tuple of ``mask.ndim`` elements
            Pixel/voxel spacing in real world units, one per each spatial
            dimension of `input_`.
        skip_classes: None or tuple of ints

    Returns:
        out: (d0, d1[, d2]) ndarray
            Thickness map for each class in the input array.
    """
    if skip_classes is None:
        skip_classes = tuple()

    if torch.is_tensor(input_):
        input_ = input_.numpy()

    th_map = np.zeros_like(input_, float)

    for class_idx in range(num_classes):
        if class_idx in skip_classes:
            continue

        sel_input_ = input_ == class_idx

        if input_.ndim == 2:
            th_map_class = _local_thickness(
                sel_input_, algorithm_2d="med2d_dist2d_lth2d",
                spacing_rw=spacing_rw, stack_axis=stack_axis,
                return_med_axis=False, return_distance=False)
        else:
            th_map_class = _local_thickness(
                sel_input_, algorithm_3d="med2d_dist3d_lth3d",
                spacing_rw=spacing_rw, stack_axis=stack_axis,
                return_med_axis=False, return_distance=False)

        th_map[sel_input_] = th_map_class[sel_input_]

    return th_map
