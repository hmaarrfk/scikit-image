import numpy as np

from skimage import data
from skimage.feature import Cascade


def test_detector_astronaut():

    # Load the trained file from the module root.
    trained_file = data.fetch("lbpcascade_frontalface_opencv.xml")

    # Initialize the detector cascade.
    detector = Cascade(trained_file)

    img = data.astronaut()

    detected = detector.detect_multi_scale(img=img,
                                           scale_factor=1.2,
                                           step_ratio=1,
                                           min_size=(60, 60),
                                           max_size=(123, 123))

    assert len(detected) == 1, 'One face should be detected.'
