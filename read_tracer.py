import os, sys
import numpy as np

def read_tracer_trajectory(mm, position, n_species, n_params=None,
                           return_params=False):
    """Reconstruct one tracer's stored chemistry solve from a streamed
    feature matrix with the ``[idx | t | params | species]`` layout.

    ``position`` is the tracer's *local* index (column 0). 
    Rows for a tracer are contiguous and ``idx`` is non-decreasing,
    so the block is located with a binary search on column 0; absolute time is
    read straight from column 1. A failed/unwritten tracer (zero-filled block,
    since real density ``nH`` in the first params column is strictly positive)
    returns empty arrays.

    Returns ``t`` shape ``(R,)`` and ``y`` shape ``(n_species, R)``; also the
    ``(R, n_params)`` params block if ``return_params=True``.
    """
    if n_params is None:
        n_params = int(mm.shape[1]) - 2 - int(n_species)
        if n_params <= 0:
            raise ValueError(
                f"cannot infer n_params from shape {mm.shape} "
                f"and n_species={n_species}."
            )
    start, end = np.searchsorted(mm[:, 0], [position, position + 1])
    block = np.asarray(mm[start:end])

    if block.shape[0] == 0 or block[0, 2] == 0.0:     # failed/unwritten tracer
        t = np.empty(0)
        y = np.empty((n_species, 0))
        if return_params:
            return t, y, np.empty((0, n_params))
        return t, y

    t = block[:, 1]
    y = block[:, 2 + n_params:2 + n_params + n_species].T

    if return_params:
        return t, y, block[:, 2:2 + n_params]
    return t, y

def remove_duplicates(t, y, params):
    _, keep = np.unique(t, return_index=True)   # first occurrence of each ticme
    t, y, params = t[keep], y[:, keep], params[keep,:]
    return t,y, params

def number_tracers(feat_path):
    mm = np.load(feat_path, mmap_mode="r")
    n_tracers = int(mm[-1, 0]) + 1
    print(f"There are {n_tracers} tracers")
    return n_tracers
