import numpy as np


def regular_grid(ar_shape, n_points, return_type='tuple'):
    """Find `n_points` regularly spaced along `ar_shape`.

    The returned points (as slices) should be as close to cubically-spaced as
    possible. Essentially, the points are spaced by the Nth root of the input
    array size, where N is the number of dimensions. However, if an array
    dimension cannot fit a full step size, it is "discarded", and the
    computation is done for only the remaining dimensions.

    Parameters
    ----------
    ar_shape : array-like of ints
        The shape of the space embedding the grid. ``len(ar_shape)`` is the
        number of dimensions.
    n_points : int
        The (approximate) number of points to embed in the space.
    return_type : str {None, 'tuple', 'list'}
        The container type that should contain the slices.
        'list' is now considered deprecated behaviour to conform with numpy
        1.15 slicing guidelines
        https://docs.scipy.org/doc/numpy/release.html#numpy-1-15-0-release-notes
        If ``None``, a list will be returned in scikit-image version 0.15.
        In scikit-image version 0.16, a tuple will be returned by default.

    Returns
    -------
    slices : tuple (or list) of slice objects
        A slice along each dimension of `ar_shape`, such that the intersection
        of all the slices give the coordinates of regularly spaced points.

    Examples
    --------
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 8, return_type='tuple')
    >>> g
    (slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 32, return_type='tuple')
    >>> g
    (slice(2, None, 5), slice(2, None, 5))
    >>> ar[g] = 1
    >>> ar.sum()
    32.0
    >>> ar = np.zeros((3, 20, 40))
    >>> g = regular_grid(ar.shape, 8, return_type='tuple')
    >>> g
    (slice(1, None, 3), slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    """
    if return_type is 'list':
        warn("The ``return_type`` parameter will be removed in scikit-image "
             "0.18 and ``" +  __name__ + "`` will always return a tuple to "
             "conform with slicing guidelines introduced in numpy 1.15. "
             "See "
             "https://docs.scipy.org/doc/numpy/release.html#numpy-1-15-0-release-notes"
             " for more details.")
    if return_type is None:
        warn("The default value of ``return_type`` will be changed to "
             "``'tuple'`` in 0.16 to conform with guidelines in numpy "
             "slicing introduced in numpy 1.15. See "
             "https://docs.scipy.org/doc/numpy/release.html#numpy-1-15-0-release-notes"
             " for more details.")
        return_type = 'list'
    if return_type not in ('list', 'tuple'):
        raise ValueError("``return_typle`` should be either ``'list'`` "
                "or ``'tuple'``.")

    ar_shape = np.asanyarray(ar_shape)
    ndim = len(ar_shape)
    unsort_dim_idxs = np.argsort(np.argsort(ar_shape))
    sorted_dims = np.sort(ar_shape)
    space_size = float(np.prod(ar_shape))
    if space_size <= n_points:
        return (slice(None), ) * ndim
    stepsizes = (space_size / n_points) ** (1.0 / ndim) * np.ones(ndim)
    if (sorted_dims < stepsizes).any():
        for dim in range(ndim):
            stepsizes[dim] = sorted_dims[dim]
            space_size = float(np.prod(sorted_dims[dim + 1:]))
            stepsizes[dim + 1:] = ((space_size / n_points) **
                                   (1.0 / (ndim - dim - 1)))
            if (sorted_dims >= stepsizes).all():
                break
    starts = (stepsizes // 2).astype(int)
    stepsizes = np.round(stepsizes).astype(int)
    slices = [slice(start, None, step) for
              start, step in zip(starts, stepsizes)]
    slices = tuple(slices[i] for i in unsort_dim_idxs)
    if return_type == 'list':
        return list(slices)
    else:
        return slices


def regular_seeds(ar_shape, n_points, dtype=int):
    """Return an image with ~`n_points` regularly-spaced nonzero pixels.

    Parameters
    ----------
    ar_shape : tuple of int
        The shape of the desired output image.
    n_points : int
        The desired number of nonzero points.
    dtype : numpy data type, optional
        The desired data type of the output.

    Returns
    -------
    seed_img : array of int or bool
        The desired image.

    Examples
    --------
    >>> regular_seeds((5, 5), 4)
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 2, 0],
           [0, 0, 0, 0, 0],
           [0, 3, 0, 4, 0],
           [0, 0, 0, 0, 0]])
    """
    grid = regular_grid(ar_shape, n_points, return_type='tuple')
    seed_img = np.zeros(ar_shape, dtype=dtype)
    seed_img[grid] = 1 + np.reshape(np.arange(seed_img[grid].size),
                                    seed_img[grid].shape)
    return seed_img
