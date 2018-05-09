from __future__ import division
import numpy as np
from warnings import warn

__all__ = ['img_as_float32', 'img_as_float64', 'img_as_float',
           'img_as_int', 'img_as_uint', 'img_as_ubyte',
           'img_as_bool', 'dtype_limits']

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.uint32: (0, 2**32 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int32: (-2**31, 2**31 - 1),
               np.int64: (-2**63, 2**63 - 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

_supported_types = (np.bool_, np.bool8,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64,
                    np.float16, np.float32, np.float64)


def _check_precision_loss(dtypeobj_in, dtypeobj_out,
                         issue_warnings=False, int_same_size_lossy=False):
    """Check if conversion between images will incur loss of precision.

    Generally speaking, the output object needs to have more significant bits
    than the input kind.

    Details:
        float -> signed/unsigned ints will always return true.
        signed/unsigned -> float of same number of bytes will return true.

    Parameters
    ----------
    dtypeobj_in: np.dtype
        dtype of the input image.
    dtypeobj_out: np.dtype
        dtype of the output image.
    output_warning: bool, optional
        Outputs a warning using `warn` if operation will incur loss of
        precision.
    int_same_size_ok: bool, optional
        Should conversion between integers of the same same (and signedness)
        be considered lossy.

    """
    dtypeobj_in = np.dtype(dtypeobj_in)
    dtypeobj_out = np.dtype(dtypeobj_out)
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    has_loss = False

    if kind_in != 'b' and kind_out == 'b':
        has_loss = True
    elif kind_in == 'f' and kind_out == 'f':
        # float->float
        if itemsize_out < itemsize_in:
            has_loss = True
    elif kind_in == 'f':
        # float -> other (integer or bool)
        has_loss = True
    elif kind_out == 'f':
        # signed/unsigned int -> float
        if itemsize_in >= itemsize_out:
            has_loss = True
    elif ((kind_in == 'i' and kind_out == 'i') or
            (kind_in == 'u' and kind_out == 'u')):
        # signed/unsigned to same kind
        if itemsize_out < itemsize_in:
            has_loss = True
        elif int_same_size_lossy and itemsize_out == itemsize_in:
            has_loss = True
    elif kind_in == 'u' and kind_out == 'i':
        # usigned -> signed int
        if itemsize_in >= itemsize_out:
            has_loss = True
    elif kind_in == 'i' and kind_out == 'u':
        # signed int to unsigned
        if itemsize_out < itemsize_in:
            has_loss = True

    if issue_warnings and has_loss:
        warn("Possible precision loss when converting from {} to {}"
             .format(dtypeobj_in, dtypeobj_out))
    return has_loss


def dtype_limits(image, clip_negative=None):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
        The default behavior (None) is equivalent to True.

    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """
    if clip_negative is None:
        clip_negative = True
        warn('The default of `clip_negative` in `skimage.util.dtype_limits` '
             'will change to `False` in version 0.15.')
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def convert(image, dtype, force_copy=False, uniform=False,
            issue_warnings=True):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    References
    ----------
    .. [1] DirectX data conversion rules.
           http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.

    """
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    if dtype_in == dtype_out:
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError("Can not convert from {} to {}."
                         .format(dtypeobj_in, dtypeobj_out))

    def sign_loss(issue_warnings=True):
        if issue_warnings:
            warn("Possible sign loss when converting negative image of type "
                "{} to positive image of type {}."
                .format(dtypeobj_in, dtypeobj_out))

    def _dtype_itemsize(itemsize, *dtypes):
        # Return first of `dtypes` with itemsize greater than `itemsize`
        return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)

    def _dtype_bits(kind, bits, itemsize=1):
        # Return dtype of `kind` that can store a `bits` wide unsigned int
        def compare(x, y, kind='u'):
            if kind == 'u':
                return x <= y
            else:
                return x < y

        s = next(i for i in (itemsize, ) + (2, 4, 8) if compare(bits, i * 8,
                                                                kind=kind))
        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        """Scale an array of unsigned/positive integers from `n` to `m` bits.

        Numbers can be represented exactly only if `m` is a multiple of `n`.

        Parameters
        ----------
        a : ndarray
            Input image array.
        n : int
            Number of bits currently used to encode the values in `a`.
        m : int
            Desired number of bits to encode the values in `out`.
        copy : bool, optional
            If True, allocates and returns new array. Otherwise, modifies
            `a` in place.

        Returns
        -------
        out : array
            Output image array. Has the same kind as `a`.
        """
        kind = a.dtype.kind
        if n > m and a.max() < 2 ** m:
            mnew = int(np.ceil(m / 2) * 2)
            if mnew > m:
                dtype = "int{}".format(mnew)
            else:
                dtype = "uint{}".format(mnew)
            n = int(np.ceil(n / 2) * 2)
            warn("Downcasting {} to {} without scaling because max "
                 "value {} fits in {}".format(a.dtype, dtype, a.max(), dtype))
            return a.astype(_dtype_bits(kind, m))
        elif n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            prec_loss()
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                                casting='unsafe')
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of `n` bits
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of `n` bits,
            # then downscale with precision loss
            prec_loss()
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2**(o - m)
                return b
            else:
                a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2**(o - m)
                return a

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    _check_precision_loss(dtype_in, dtype_out, issue_warnings=issue_warnings)

    # any -> binary
    if kind_out == 'b':
        if kind_in in "fi":
            sign_loss()
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        if kind_out == 'f':
            # float -> float
            return image.astype(dtype_out)

        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(itemsize_out, dtype_in,
                                           np.float32, np.float64)
        if force_copy or computation_type != image.dtype:
            image_out = np.empty(shape=image.shape, dtype=computation_type)
        else:
            image_out = image

        if not uniform:
            if kind_out == 'u':
                np.multiply(image, imax_out,
                            out=image_out, dtype=computation_type)
            else:
                np.multiply(image, (imax_out - imin_out) / 2,
                            out=image_out, dtype=computation_type)
                image_out -= 1.0 / 2.
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == 'u':
            np.multiply(image, imax_out + 1,
                        out=image_out, dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            np.multiply(image, (imax_out - imin_out + 1.0) / 2.0,
                        out=image_out, dtype=computation_type)
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(itemsize_in, dtype_out,
                                           np.float32, np.float64)
        if kind_in == 'u':
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1. / imax_in,
                                dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image = np.multiply(image, 2. / (imax_in - imin_in),
                                dtype=computation_type)
            image += 1.0 / (imax_in - imin_in)
        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        sign_loss()
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def img_as_float32(image, force_copy=False, issue_warnings=True):
    """Convert an image to single-precision (32-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of float32
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return convert(image, np.float32, force_copy)


def img_as_float64(image, force_copy=False, issue_warnings=True):
    """Convert an image to double-precision (64-bit) floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return convert(image, np.float64, force_copy,
                   issue_warnings=issue_warnings)


def img_as_float(image, force_copy=False, default_dtype=np.float64,
                 issue_warnings=True):
    """Ensure that an image is of floating point type.

    First checks if the image is a floating point (any precision). If not
    it will convert it to the desired precision (by default, 64-bit).

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    default_dtype: np.dtype
        Desired for the image if it is not already a float.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of desired precision
        Output image.
    """
    if np.dtype(default_dtype).kind != 'f':
        raise ValueError('Provided default_dtype should be a float.')

    if image.dtype.kind == 'f':
        if force_copy:
            return image.copy()
        else:
            return image
    else:
        return convert(image, dtype=default_dtype, force_copy=force_copy,
                       issue_warnings=issue_warnings)


def img_as_uint(image, force_copy=False, issue_warnings=True):
    """Convert an image to 16-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 65535.

    """
    return convert(image, np.uint16, force_copy,
                   issue_warnings=issue_warnings)


def img_as_int(image, force_copy=False, issue_warnings=True):
    """Convert an image to 16-bit signed integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of uint16
        Output image.

    Notes
    -----
    The values are scaled between -32768 and 32767.
    If the input data-type is positive-only (e.g., uint8), then
    the output image will still only have positive values.

    """
    return convert(image, np.int16, force_copy,
                   issue_warnings=issue_warnings)


def img_as_ubyte(image, force_copy=False, issue_warnings=True):
    """Convert an image to 8-bit unsigned integer format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of ubyte (uint8)
        Output image.

    Notes
    -----
    Negative input values will be clipped.
    Positive values are scaled between 0 and 255.

    """
    return convert(image, np.uint8, force_copy,
                   issue_warnings=issue_warnings)


def img_as_bool(image, force_copy=False, issue_warnings=True):
    """Convert an image to boolean format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    issue_warnings: bool, optional
        Output warnings during convertion (typically for precision loss).

    Returns
    -------
    out : ndarray of bool (`bool_`)
        Output image.

    Notes
    -----
    The upper half of the input dtype's positive range is True, and the lower
    half is False. All negative values (if present) are False.

    """
    return convert(image, np.bool_, force_copy,
                   issue_warnings=issue_warnings)
