# -*- coding: utf-8 -*-
"""
Torch backend for STL.

Provides a minimal API used across the codebase, with optional device
selection (CPU / GPU) via a `device` argument.

Exposed functions
-----------------
- from_numpy(x, device=None, dtype=None)
- zeros(shape, device=None, dtype=torch.float32)
- mean(x, dim)
- dim(x)
- shape(x, axis=None)
- nan
"""

import numpy as np
import torch

# ---------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------

# Global default device: GPU if available, else CPU
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device(device=None) -> torch.device:
    """
    Resolve a device spec into a torch.device.

    device:
        - None        -> _DEFAULT_DEVICE
        - "gpu"       -> "cuda" if available, else "cpu"
        - "cuda", "cuda:0", "cpu", ... -> passed to torch.device
        - torch.device -> returned as-is
    """
    if device is None:
        return _DEFAULT_DEVICE

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        d = device.lower()
        if d in ("gpu", "cuda"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if d == "cpu":
            return torch.device("cpu")
        # Let torch.device handle strings like "cuda:0"
        return torch.device(device)

    # Fallback: let torch.device figure it out / raise
    return torch.device(device)


# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------

def from_numpy(x: np.ndarray, device=None, dtype=None):
    """
    Convert a NumPy array to a torch.Tensor on the requested device.

    Parameters
    ----------
    x : np.ndarray
    device : None, str, or torch.device
        If "gpu"/"cuda" -> CUDA (if available), otherwise CPU.
        If "cpu"       -> CPU.
        If None        -> default device (_DEFAULT_DEVICE).
    dtype : torch.dtype or None
        If not None, cast to this dtype.
    """
    t = torch.from_numpy(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(_get_device(device))


def zeros(shape, device=None, dtype=torch.float32):
    """
    Return a tensor of zeros on CPU or GPU depending on `device`.

    Parameters
    ----------
    shape : tuple or list
    device : None, str, or torch.device
    dtype : torch.dtype
    """
    return torch.zeros(shape, dtype=dtype, device=_get_device(device))


def mean(x, dim):
    """
    Compute the mean of `x` along the given dimension.

    NOTE: `dim` has NO default value on purpose, so calls like
    `bk.mean(t, -1)` behave exactly as written.
    """
    return torch.mean(x, dim=dim)


def dim(x) -> int:
    """
    Return the number of dimensions of a tensor-like object.
    """
    if hasattr(x, "dim"):
        return x.dim()
    return np.array(x).ndim


def shape(x, axis=None):
    """
    Return the shape of `x`, or the size along a given axis.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
    axis : int or None
        If None, return the full shape tuple.
        Otherwise, return the size along the given axis.
    """
    s = x.shape
    return s if axis is None else s[axis]


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Scalar NaN; e.g. bk.zeros(...)+bk.nan -> NaN-filled tensor
nan = float("nan")