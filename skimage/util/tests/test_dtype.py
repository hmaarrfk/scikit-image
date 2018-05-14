import numpy as np
import itertools
from skimage import (img_as_int, img_as_float,
                     img_as_uint, img_as_ubyte)
from skimage.util.dtype import convert
from skimage.util.dtype import _check_precision_loss as check_precision_loss

from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize


dtype_range = {np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1.0, 1.0),
               np.float64: (-1.0, 1.0)}


img_funcs = (img_as_int, img_as_float, img_as_uint, img_as_ubyte)
dtypes_for_img_funcs = (np.int16, np.float64, np.uint16, np.ubyte)
img_funcs_and_types = zip(img_funcs, dtypes_for_img_funcs)


def _verify_range(msg, x, vmin, vmax, dtype):
    assert_equal(x[0], vmin)
    assert_equal(x[-1], vmax)
    assert x.dtype == dtype


@parametrize("dtype, f_and_dt",
             itertools.product(dtype_range, img_funcs_and_types))
def test_range(dtype, f_and_dt):
    imin, imax = dtype_range[dtype]
    x = np.linspace(imin, imax, 10).astype(dtype)

    f, dt = f_and_dt

    with expected_warnings(['precision loss|sign loss|\A\Z']):
        y = f(x)

    omin, omax = dtype_range[dt]

    if imin == 0 or omin == 0:
        omin = 0
        imin = 0

    _verify_range("From %s to %s" % (np.dtype(dtype), np.dtype(dt)),
                  y, omin, omax, np.dtype(dt))


# Add non-standard data types that are allowed by the `convert` function.
dtype_range_extra = dtype_range.copy()
dtype_range_extra.update({np.int32: (-2147483648, 2147483647),
                          np.uint32: (0, 4294967295)})

dtype_pairs = [(np.uint8, np.uint32),
               (np.int8, np.uint32),
               (np.int8, np.int32),
               (np.int32, np.int8),
               (np.float64, np.float32),
               (np.int32, np.float32)]


@parametrize("dtype_in, dt", dtype_pairs)
def test_range_extra_dtypes(dtype_in, dt):
    """Test code paths that are not skipped by `test_range`"""

    imin, imax = dtype_range_extra[dtype_in]
    x = np.linspace(imin, imax, 10).astype(dtype_in)

    with expected_warnings(['precision loss|sign loss|\A\Z']):
        y = convert(x, dt)

    omin, omax = dtype_range_extra[dt]
    _verify_range("From %s to %s" % (np.dtype(dtype_in), np.dtype(dt)),
                  y, omin, omax, np.dtype(dt))


def test_downcast():
    x = np.arange(10).astype(np.uint64)
    with expected_warnings('Downcasting'):
        y = img_as_int(x)
    assert np.allclose(y, x.astype(np.int16))
    assert y.dtype == np.int16, y.dtype


def test_float_out_of_range():
    too_high = np.array([2], dtype=np.float32)
    with testing.raises(ValueError):
        img_as_int(too_high)
    too_low = np.array([-2], dtype=np.float32)
    with testing.raises(ValueError):
        img_as_int(too_low)


def test_copy():
    x = np.array([1], dtype=np.float64)
    y = img_as_float(x)
    z = img_as_float(x, force_copy=True)

    assert y is x
    assert z is not x


def test_bool():
    img_ = np.zeros((10, 10), np.bool_)
    img8 = np.zeros((10, 10), np.bool8)
    img_[1, 1] = True
    img8[1, 1] = True
    for (func, dt) in [(img_as_int, np.int16),
                       (img_as_float, np.float64),
                       (img_as_uint, np.uint16),
                       (img_as_ubyte, np.ubyte)]:
        converted_ = func(img_)
        assert np.sum(converted_) == dtype_range[dt][1]
        converted8 = func(img8)
        assert np.sum(converted8) == dtype_range[dt][1]


def test_check_precision_loss():
    dtype_pairs_conversion = [(np.uint8, np.uint8, False),
                              (np.uint8, np.int8, True),
                              (np.uint8, np.uint16, False),
                              (np.uint8, np.int16, False),
                              (np.uint8, np.float32, False),
                              (np.uint8, np.float64, False),

                              (np.int8, np.uint8, False),
                              (np.int8, np.int8, False),
                              (np.int8, np.uint16, False),
                              (np.int8, np.int16, False),
                              (np.int8, np.float32, False),
                              (np.int8, np.float64, False),

                              (np.int16, np.uint8, True),
                              (np.int16, np.int8, True),
                              (np.int16, np.uint16, False),
                              (np.int16, np.int16, False),
                              (np.int16, np.float32, False),
                              (np.int16, np.float64, False),

                              (np.uint16, np.uint8, True),
                              (np.uint16, np.int8, True),
                              (np.uint16, np.uint16, False),
                              (np.uint16, np.int16, True),
                              (np.uint16, np.float32, False),
                              (np.uint16, np.float64, False),

                              (np.float32, np.uint8, True),
                              (np.float32, np.int8, True),
                              (np.float32, np.uint16, True),
                              (np.float32, np.int16, True),
                              (np.float32, np.float32, False),
                              (np.float32, np.float64, False),

                              (np.float64, np.uint8, True),
                              (np.float64, np.int8, True),
                              (np.float64, np.uint16, True),
                              (np.float64, np.int16, True),
                              (np.float64, np.float32, True),
                              (np.float64, np.float64, False),

                              (np.bool, np.uint8, False),
                              (np.bool, np.int8, False),
                              (np.bool, np.uint16, False),
                              (np.bool, np.int16, False),
                              (np.bool, np.float32, False),
                              (np.bool, np.float64, False),

                              (np.uint8, np.bool, True),
                              (np.int8, np.bool, True),
                              (np.uint16, np.bool, True),
                              (np.int16, np.bool, True),
                              (np.float32, np.bool, True),
                              (np.float64, np.bool, True),
                              ]

    for (dtype_in, dtype_out, expected_value) in dtype_pairs_conversion:
        assert check_precision_loss(
            dtype_in, dtype_out, issue_warnings=False) == expected_value

    assert check_precision_loss(np.uint8, np.uint8,
                                issue_warnings=False, int_same_size_lossy=True)
