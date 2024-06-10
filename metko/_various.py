import math
import numpy as np

import torch
import torch.nn.functional as F


# Modified from https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
def ncc(y_true, y_pred, *, win=None, device="cpu"):
    """Local (over window) normalized cross correlation loss."""
    y_true = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=device)

    y_true = y_true[None, None, ...]
    y_pred = y_pred[None, None, ...]

    i_i = y_true
    j_i = y_pred

    # get dimension of volume
    # assumes i_i, j_i are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(i_i.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to(device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1, )
        padding = (pad_no, )
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    i_2 = i_i * i_i
    j_2 = j_i * j_i
    ij = i_i * j_i

    i_sum = conv_fn(i_i, sum_filt, stride=stride, padding=padding)
    j_sum = conv_fn(j_i, sum_filt, stride=stride, padding=padding)
    i_2_sum = conv_fn(i_2, sum_filt, stride=stride, padding=padding)
    j_2_sum = conv_fn(j_2, sum_filt, stride=stride, padding=padding)
    ij_sum = conv_fn(ij, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_i = i_sum / win_size
    u_j = j_sum / win_size

    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
    i_var = i_2_sum - 2 * u_i * i_sum + u_i * u_i * win_size
    j_var = j_2_sum - 2 * u_j * j_sum + u_j * u_j * win_size

    cc = cross * cross / (i_var * j_var + 1e-5)

    return -torch.mean(cc).numpy()


def mse(y_1, y_2):
    """Mean squared error.

    Args:
        y_1: (d0, ..., dn) ndarray
        y_2: (d0, ..., dn) ndarray

    Returns:
        out: float
    """
    y_1 = torch.as_tensor(y_1, dtype=torch.float32)
    y_2 = torch.as_tensor(y_2, dtype=torch.float32)
    return torch.mean((y_1 - y_2) ** 2).numpy()
