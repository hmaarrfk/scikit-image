import numpy as np
from skimage.util import regular_grid
from skimage._shared.testing import assert_equal
from skimage._shared._warnings import expected_warnings


def test_regular_grid_full():
    ar = np.zeros((2, 2))
    g = regular_grid(ar, 25, return_type='tuple')
    assert_equal(g, (slice(None, None, None), slice(None, None, None)))
    ar[g] = 1
    assert_equal(ar.size, ar.sum())


def test_regular_grid_2d_8():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 8, return_type='tuple')
    assert_equal(g, (slice(5.0, None, 10.0), slice(5.0, None, 10.0)))
    ar[g] = 1
    assert_equal(ar.sum(), 8)


def test_regular_grid_2d_32():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 32, return_type='tuple')
    assert_equal(g, (slice(2.0, None, 5.0), slice(2.0, None, 5.0)))
    ar[g] = 1
    assert_equal(ar.sum(), 32)


def test_regular_grid_3d_8():
    ar = np.zeros((3, 20, 40))
    g = regular_grid(ar.shape, 8, return_type='tuple')
    assert_equal(g, (slice(1.0, None, 3.0), slice(5.0, None, 10.0),
                     slice(5.0, None, 10.0)))
    ar[g] = 1
    assert_equal(ar.sum(), 8)

def test_regular_grid_warnings():
    ar = np.zeros((3, 20, 40))
    with expected_warnings(['slicing introduced in numpy 1.15']):
        g = regular_grid(ar.shape, 8)
        assert_equal(g.__class__, list)
        assert_equal(g, [slice(1.0, None, 3.0), slice(5.0, None, 10.0),
                         slice(5.0, None, 10.0)])
    with expected_warnings(['``return_type`` parameter will be removed']):
        g = regular_grid(ar.shape, 8, return_type='list')
        assert_equal(g.__class__, list)
        assert_equal(g, [slice(1.0, None, 3.0), slice(5.0, None, 10.0),
                         slice(5.0, None, 10.0)])
    with expected_warnings([None]):
        g = regular_grid(ar.shape, 8, return_type='tuple')
        assert_equal(g.__class__, tuple)
