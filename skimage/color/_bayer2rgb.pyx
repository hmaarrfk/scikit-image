#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

"""These are unrolled implementations of bayer2rgb.

The goal is to have both a low memory order for the computation, and a low
computation order. The image is assumed to be of even size (N, M)

With the exception of the boundary conditions (which are O(M) or O(N)), the
computation is done in the following steps:

- Compute the red and blue components along the C direction.
    2 floating point operation
- Compute the rest
    - The green pixels are computed directly from the raw data.
    - The red and blue pixels are computed from intermediate data to avoid 2
      extra floating point additions.
"""

def _bayer2rgb_rggb(np.float64_t[:, ::1] raw_image):
    # Default bayer_pattern is 'rggb'
    cdef Py_ssize_t i, j
    cdef Py_ssize_t j_max = raw_image.shape[0]
    cdef Py_ssize_t i_max = raw_image.shape[1]

    # In this case, top_left refers to the red color
    # and br refers to the blue color
    cdef Py_ssize_t tl = 0
    cdef Py_ssize_t br = 2

    # i_max and j_max should be even
    color_image = np.ones(shape=(raw_image.shape[0], raw_image.shape[1], 3),
                           dtype=np.float64)

    # Horizontal boundary conditions
    # Measured exactly
    color_image[::2, 0, tl] = raw_image[::2, 0]
    color_image[1::2, i_max-1, br] = raw_image[1::2, i_max-1]

    # Assume a symmetric boundary condition. So the average of the two values
    # is original value copied over
    color_image[::2, i_max-1, 0] = raw_image[::2, i_max-2]
    color_image[1::2, 0, br] = raw_image[1::2, 1]


    # First compute the average along the x dimension
    # memory order O(i_max*j_max)
    for j in range(1, j_max - 1):
        for i in range(1, i_max - 1):
            if j % 2 == 0:
                if i % 2 == 1:
                    color_image[j, i, 0 ] = (
                        raw_image[j, i-1] + raw_image[j, i+1]) / 2
                else:
                    color_image[j, i, tl] = raw_image[j, i]
            else:
                if i % 2 == 0:
                    color_image[j, i , br] = (
                        raw_image[j, i-1] + raw_image[j, i+1]) / 2
                else:
                    color_image[j, i, br] = raw_image[j, i]

    # Red top boundary condition
    # use color_image during the computation to not fetch raw_image again
    color_image[0, ::2, tl] = raw_image[0, ::2]
    color_image[0, i_max-1, tl] = color_image[0, i_max-2, tl]
    if i_max > 2:
        color_image[0, 1:i_max-1:2, tl] = (
            color_image[0, :i_max-2:2, tl] + color_image[0, 2::2, tl]) / 2

    # blue bottom boundary condition
    color_image[j_max-1, 1::2, br] = raw_image[j_max-1, 1::br]
    color_image[j_max-1, 0, br] = color_image[j_max-1, 1, br]
    if i_max > 2:
        color_image[j_max-1, 2::2, br] = (
            color_image[j_max-1, 1:i_max-2:2, br] +
            color_image[j_max-1, 3:i_max:2, br]) / 2


    # Unrolled for speed
    for j in range(1, j_max-1):
        # only half the of the R/B entries need to be written
        if j % 2 == 0:
            color_image[j, 0, br] = (
                color_image[j-1, 0, br] + color_image[j+1, 0, br]) / 2
        else:
            color_image[j, 0, tl] = (
                color_image[j-1, 0, tl] + color_image[j+1, 0, tl]) / 2

        for i in range(1, i_max-1):
            if (j+i) % 2 == 0:
                color_image[j, i, 1] = (
                    raw_image[j-1, i] + raw_image[j+1, i] +
                    raw_image[j, i-1] + raw_image[j, i+1]) / 4
            else:
                color_image[j, i, 1] = raw_image[j, i]

            # only half the of the R/B entries need to be written
            if j % 2 == 0:
                color_image[j, i, br] = (
                    color_image[j-1, i, br] + color_image[j+1, i, br]) / 2
            else:
                color_image[j, i, tl] = (
                    color_image[j-1, i, tl] + color_image[j+1, i, tl]) / 2
                    # only half the of the R/B entries need to be written

        if j % 2 == 0:
            color_image[j, i_max-1, br] = (
                color_image[j-1, i_max-1, br] + color_image[j+1, i_max-1, br]) / 2
        else:
            color_image[j, i_max-1, tl] = (
                color_image[j-1, i_max-1, tl] + color_image[j+1, i_max-1, tl]) / 2

    # Red bottom boundary condition
    color_image[j_max-1, :, tl] = color_image[j_max-2, :, tl]
    # Blue top boundary condition
    color_image[0, :, br] = color_image[1, :, br]

    # Green boundary conditions

    # for the green boundary condition
    color_image[0, 0, 1] = (raw_image[0, 1] + raw_image[1, 0]) / 2
    color_image[j_max-1, i_max-1, 1] = (raw_image[j_max-2, i_max-1] +
                                        raw_image[j_max-1, i_max-2]) / 2
    color_image[0, 1::2, 1] = raw_image[0, 1::2]
    color_image[j_max-1, 0::2, 1] = raw_image[j_max-1, 0::2]
    color_image[1::2, 0, 1] = raw_image[1::2, 0]
    color_image[0::2, i_max-1, 1] = raw_image[0::2, i_max-1]

    if i_max > 2:
        color_image[0, 2::2, 1] = (
            color_image[0, 1:i_max-2:2, 1] + color_image[0, 3::2, 1] +
            color_image[1, 2::2, 1] * 2) / 4
        color_image[j_max-1, 1:i_max-2:2, 1] = (
            color_image[j_max-1, 0:i_max-2:2, 1] + color_image[j_max-1, 2::2, 1] +
            color_image[j_max-2, 1:i_max-2:2, 1] * 2) / 4

    if j_max > 2:
        color_image[2::2, 0, 1] = (
            color_image[1:j_max-2:2, 0, 1] + color_image[3::2, 0, 1] +
            color_image[2::2, 1, 1] * 2) / 4
        color_image[1:j_max-2:2, i_max-1, 1] = (
            color_image[0:j_max-2:2, i_max-1, 1] + color_image[2::2, i_max-1, 1] +
            color_image[1:j_max-2:2, i_max-2, 1] * 2) / 4

    return color_image


def _bayer2rgb_grbg(np.float64_t[:, ::1] raw_image):
    # Default bayer_pattern is 'rggb'
    cdef Py_ssize_t i, j
    cdef Py_ssize_t j_max = raw_image.shape[0]
    cdef Py_ssize_t i_max = raw_image.shape[1]

    # In this case, top_left refers to the red color
    # and br refers to the blue color
    cdef Py_ssize_t tl = 0
    cdef Py_ssize_t br = 2

    # i_max and j_max should be even
    color_image = np.ones(shape=(raw_image.shape[0], raw_image.shape[1], 3),
                           dtype=np.float64)

    # Horizontal boundary conditions
    # Measured exactly
    color_image[1::2, 0, tl] = raw_image[::2, 0]
    color_image[::2, i_max-1, br] = raw_image[1::2, i_max-1]

    """
    # Assume a symmetric boundary condition. So the average of the two values
    # is original value copied over
    color_image[::2, i_max-1, 0] = raw_image[::2, i_max-2]
    color_image[1::2, 0, br] = raw_image[1::2, 1]


    # First compute the average along the x dimension
    # memory order O(i_max*j_max)
    for j in range(1, j_max - 1):
        for i in range(1, i_max - 1):
            if j % 2 == 0:
                if i % 2 == 1:
                    color_image[j, i, 0 ] = (
                        raw_image[j, i-1] + raw_image[j, i+1]) / 2
                else:
                    color_image[j, i, tl] = raw_image[j, i]
            else:
                if i % 2 == 0:
                    color_image[j, i , br] = (
                        raw_image[j, i-1] + raw_image[j, i+1]) / 2
                else:
                    color_image[j, i, br] = raw_image[j, i]

    # Red top boundary condition
    # use color_image during the computation to not fetch raw_image again
    color_image[0, ::2, tl] = raw_image[0, ::2]
    color_image[0, i_max-1, tl] = color_image[0, i_max-2, tl]
    if i_max > 2:
        color_image[0, 1:i_max-1:2, tl] = (
            color_image[0, :i_max-2:2, tl] + color_image[0, 2::2, tl]) / 2

    # blue bottom boundary condition
    color_image[j_max-1, 1::2, br] = raw_image[j_max-1, 1::br]
    color_image[j_max-1, 0, br] = color_image[j_max-1, 1, br]
    if i_max > 2:
        color_image[j_max-1, 2::2, br] = (
            color_image[j_max-1, 1:i_max-2:2, br] +
            color_image[j_max-1, 3:i_max:2, br]) / 2


    # Unrolled for speed
    for j in range(1, j_max-1):
        # only half the of the R/B entries need to be written
        if j % 2 == 0:
            color_image[j, 0, br] = (
                color_image[j-1, 0, br] + color_image[j+1, 0, br]) / 2
        else:
            color_image[j, 0, tl] = (
                color_image[j-1, 0, tl] + color_image[j+1, 0, tl]) / 2

        for i in range(1, i_max-1):
            if (j+i) % 2 == 0:
                color_image[j, i, 1] = (
                    raw_image[j-1, i] + raw_image[j+1, i] +
                    raw_image[j, i-1] + raw_image[j, i+1]) / 4
            else:
                color_image[j, i, 1] = raw_image[j, i]

            # only half the of the R/B entries need to be written
            if j % 2 == 0:
                color_image[j, i, br] = (
                    color_image[j-1, i, br] + color_image[j+1, i, br]) / 2
            else:
                color_image[j, i, tl] = (
                    color_image[j-1, i, tl] + color_image[j+1, i, tl]) / 2
                    # only half the of the R/B entries need to be written

        if j % 2 == 0:
            color_image[j, i_max-1, br] = (
                color_image[j-1, i_max-1, br] + color_image[j+1, i_max-1, br]) / 2
        else:
            color_image[j, i_max-1, tl] = (
                color_image[j-1, i_max-1, tl] + color_image[j+1, i_max-1, tl]) / 2

    # Red bottom boundary condition
    color_image[j_max-1, :, tl] = color_image[j_max-2, :, tl]
    # Blue top boundary condition
    color_image[0, :, br] = color_image[1, :, br]

    # Green boundary conditions

    # for the green boundary condition
    color_image[0, 0, 1] = (raw_image[0, 1] + raw_image[1, 0]) / 2
    color_image[j_max-1, i_max-1, 1] = (raw_image[j_max-2, i_max-1] +
                                        raw_image[j_max-1, i_max-2]) / 2
    color_image[0, 1::2, 1] = raw_image[0, 1::2]
    color_image[j_max-1, 0::2, 1] = raw_image[j_max-1, 0::2]
    color_image[1::2, 0, 1] = raw_image[1::2, 0]
    color_image[0::2, i_max-1, 1] = raw_image[0::2, i_max-1]

    if i_max > 2:
        color_image[0, 2::2, 1] = (
            color_image[0, 1:i_max-2:2, 1] + color_image[0, 3::2, 1] +
            color_image[1, 2::2, 1] * 2) / 4
        color_image[j_max-1, 1:i_max-2:2, 1] = (
            color_image[j_max-1, 0:i_max-2:2, 1] + color_image[j_max-1, 2::2, 1] +
            color_image[j_max-2, 1:i_max-2:2, 1] * 2) / 4

    if j_max > 2:
        color_image[2::2, 0, 1] = (
            color_image[1:j_max-2:2, 0, 1] + color_image[3::2, 0, 1] +
            color_image[2::2, 1, 1] * 2) / 4
        color_image[1:j_max-2:2, i_max-1, 1] = (
            color_image[0:j_max-2:2, i_max-1, 1] + color_image[2::2, i_max-1, 1] +
            color_image[1:j_max-2:2, i_max-2, 1] * 2) / 4
    """
    return color_image
