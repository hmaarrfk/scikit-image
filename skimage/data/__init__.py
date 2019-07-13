"""Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""

import os as _os
import numpy as np

from distutils.version import LooseVersion as Version
import pooch

from ..io import imread
from .._shared._warnings import expected_warnings, warn
from ..util.dtype import img_as_bool
from ._binary_blobs import binary_blobs
from ._detect import lbp_frontal_face_cascade_filename

from .. import __version__

import os.path as osp
data_dir = osp.abspath(osp.dirname(__file__))

__all__ = ['data_dir',
           'load',
           'download',
           'astronaut',
           'binary_blobs',
           'brick',
           'camera',
           'checkerboard',
           'chelsea',
           'clock',
           'coffee',
           'coins',
           'colorwheel',
           'grass',
           'gravel',
           'horse',
           'hubble_deep_field',
           'immunohistochemistry',
           'lbp_frontal_face_cascade_filename',
           'lfw_subset',
           'logo',
           'microaneurysms',
           'moon',
           'page',
           'text',
           'retina',
           'rocket',
           'rough_wall',
           'shepp_logan_phantom',
           'stereo_motorcycle']


# Pooch expects a `+` to exist in development versions.
# Since scikit-image doesn't follow that convetion, we have to manually provide
# it with the URL and set the version to None
if 'dev' in Version(__version__).version:
    # even dev versions should use the online repo since users might be
    # using the nightly builds
    base_url = "https://github.com/scikit-image/scikit-image/raw/master/skimage/"
    version = None
else:
    base_url = "https://github.com/scikit-image/scikit-image/raw/{version}/skimage/"
    version = __version__

# Create a new friend to manage your sample data storage
image_fetcher = pooch.create(
    # Pooch uses appdirs to select an appropriate directory for the cache on
    # each platform.
    # https://github.com/ActiveState/appdirs
    # On linux this converges to
    # '$HOME/.cache/scikit-image'
    path=pooch.os_cache("scikit-image"),
    base_url=base_url,
    version=version,
    env="SKIMAGE_DATADIR",
    # To generate the SHA256 hash, use the command
    # openssl sha256 filename
    registry={
        "data/astronaut.png" : "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
        "data/brick.png": "71f1912c840a4fb1576e2b7a1c44c542dadcd9c87dff1bbd342c3fa373dee0ba",
        "data/camera.png": "361a6d56d22ee52289cd308d5461d090e06a56cb36007d8dfc3226cbe8aaa5db",
        "data/cells_qpi.npz": "a9c5212894bd4de8fddebd679500aff67f04b4e25e41c3f347f3e876ce648252",
        "data/chessboard_GRAY.png": "3e51870774515af4d07d820bd8827364c70839bf9b573c746e485095e893df90",
        "data/chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
        "data/clock_motion.png": "f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955",
        "data/coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
        "data/coins.png": "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba",
        "data/color.png": "7d2df993de2b4fa2a78e04e5df8050f49a9c511aa75e59ab3bd56ac9c98aef7e",
        "data/horse.png": "c7fb60789fe394c485f842291ea3b21e50d140f39d6dcb5fb9917cc178225455",
        "data/grass.png": "ced49494bf777157e75a733d978dd3e54a01251687828eefced312ca7f62ad8e",
        "data/hubble_deep_field.jpg": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
        "data/ihc.png": "f8dd1aa387ddd1f49d8ad13b50921b237df8e9b262606d258770687b0ef93cef",
        "data/logo.png": "f2c57fe8af089f08b5ba523d95573c26e62904ac5967f4c8851b27d033690168",
        "data/lfw_subset.npy": "9560ec2f5edfac01973f63a8a99d00053fecd11e21877e18038fbe500f8e872c",
        "data/microaneurysms.png": "a1e1be59aa447f8ce082f7fa809997ab369a2b137cb6c4202abc647c7ccf6456",
        "data/moon.png": "78739619d11f7eb9c165bb5d2efd4772cee557812ec847532dbb1d92ef71f577",
        "data/motorcycle_left.png": "db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179",
        "data/motorcycle_right.png": "5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797",
        "data/motorcycle_disp.npz": "2e49c8cebff3fa20359a0cc6880c82e1c03bbb106da81a177218281bc2f113d7",
        "data/page.png": "341a6f0a61557662b02734a9b6e56ec33a915b2c41886b97509dedf2a43b47a3",
        "data/phantom.png": "552ff698167aa402cceb17981130607a228a0a0aa7c519299eaa4d5f301ba36c",
        "data/retina.jpg": "38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6",
        "data/rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
        "data/rough-wall.png": "59c641fabbb70ba50b47660ce385fffb2da2a7f0b9e5e4184e882c3094bb4207",
        "data/text.png": "bd84aa3a6e3c9887850d45d606c96b2e59433fbef50338570b63c319e668e6d1",
        "data/tests/chessboard_GRAY_U16.tif": "9fd3392c5b6cbc5f686d8ff83eb57ef91d038ee0852ac26817e5ac99df4c7f45",
        "data/tests/chessboard_GRAY_U16B.tif": "b0a9270751f0fc340c90b8b615b62b88187b9ab5995942717566735d523cddb2",
        "data/tests/chessboard_GRAY_U8.npy": "71f394694b721e8a33760a355b3666c9b7d7fc1188ff96b3cd23c2a1d73a38d8",
        "data/lbpcascade_frontalface_opencv.xml": "8cd81c5fccdbcca6b623a5f157e71b27e91907e667626a0e07da279745e12d19",
        "data/tests/astronaut_GRAY_hog_L1.npy": "5d8ab22b166d1dd49c12caeff9d178ed28132efea3852b952e9d75f7f7f94954",
        "data/tests/astronaut_GRAY_hog_L2-Hys.npy": "c4dd6e50d1129aada358311cf8880ce8c775f31e0e550fc322c16e43a96d56fe",
        "data/tests/rank_filter_tests.npz": "efaf5699630f4a53255e91681dc72a965acd4a8aa1f84671c686fb93e7df046d",
        "data/tests/multi.fits": "5c71a83436762a52b1925f2f0d83881af7765ed50aede155af2800e54bbd5040",
        "data/tests/simple.fits": "cd36087fdbb909b6ba506bbff6bcd4c5f4da3a41862608fbac5e8555ef53d40f",
        "data/tests/palette_color.png": "c4e817035fb9f7730fe95cff1da3866dea01728efc72b6e703d78f7ab9717bdd",
        "data/tests/palette_gray.png": "bace7f73783bf3ab3b7fdaf701707e4fa09f0dbd0ea72cf5b12ddc73d50b02a9",
        "data/tests/green_palette.png": "42d49d94be8f9bc76e50639d3701ed0484258721f6b0bd7f50bb1b9274a010f0",
        "data/tests/truncated.jpg": "4c226038acc78012d335efba29c6119a24444a886842182b7e18db378f4a557d",
        "data/tests/multipage.tif": "4da0ad0d3df4807a9847247d1b5e565b50d46481f643afb5c37c14802c78130f",
        "data/tests/multipage_rgb.tif": "1d23b844fd38dce0e2d06f30432817cdb85e52070d8f5460a2ba58aebf34a0de",
        "data/tests/no_time_for_that_tiny.gif": "20abe94ba9e45f18de416c5fbef8d1f57a499600be40f9a200fae246010eefce",
        "data/tests/foo3x5x4indexed.png": "48a64c25c6da000ffdb5fcc34ebafe9ba3b1c9b61d7984ea7ca6dc54f9312dfa",
        "data/tests/mssim_matlab_output.npz": "cc11a14bfa040c75b02db32282439f2e2e3e96779196c171498afaa70528ed7a",
        "data/tests/gray_morph_output.npz": "3012eb994e864e1dca1f66fada6b4375f84eac63658d049886b710488c2394d1",
        "data/tests/disk-matlab-output.npz": "8a39d5c866f6216d6a9c9166312aa4bbf4d18fab3d0dcd963c024985bde5856b",
        "data/tests/diamond-matlab-output.npz": "02fca68907e2b252b501dfe977eef71ae39fadaaa3702ebdc855195422ae1cc2",
        "data/tests/bw_text.png": "308c2b09f8975a69b212e103b18520e8cbb7a4eccfce0f757836cd371f1b9094",
        "data/tests/bw_text_skeleton.npy": "9ff4fc23c6a01497d7987f14e3a97cbcc39cce54b2b3b7ee33b84c1b661d0ae1",
        "data/tests/_blobs_3d_fiji_skeleton.tif": "5182a2a94f240528985b8d15ec2aebbd5ca3c6b9214beff1eb6099c431e12b7b",
        "data/tests/checker_bilevel.png": "2e207e486545874a2a3e69ba653b28fdef923157be9017559540e65d1bcb8e28",
        "restoration/tests/camera_rl.npy": "d219834415dc7094580abd975abb28bc7a6fb5ab83366e92c61ccffa66ca54fd",
        "restoration/tests/camera_unsup.npy": "6d911fd0028ee78add8c416553097f15c6c4e59723ea32bd828f71269b6ea240",
        "restoration/tests/camera_unsup2.npy": "30e81718f3cac0fc00d84518ca75a3c0fb9b246bb7748a9e20ec0b44da33734d",
        "restoration/tests/camera_wiener.npy": "71e7cab739d6d145a288ec85dd235a62ff34442ccd1488b08139bc809850772b",
        "feature/tests/data/OriginalX-130Y130.png": "bf24a06d99ae131c97e582ef5e1cd0c648a8dad0caab31281f3564045492811f",
        "feature/tests/data/OriginalX130Y130.png": "7fdd4c06d504fec35ee0703bd7ed2c08830b075a74c8506bae4a70d682f5a2db",
        "feature/tests/data/OriginalX75Y75.png": "c5cd58893c93140df02896df80b13ecf432f5c86eeaaf8fb311aec52a65c7016",
        "feature/tests/data/TransformedX-130Y130.png": "1cda90ed69c921eb7605b73b76d141cf4ea03fb8ce3336445ca08080e40d7375",
        "feature/tests/data/TransformedX130Y130.png": "bb10c6ae3f91a313b0ac543efdb7ca69c4b95e55674c65a88472a6c4f4692a25",
        "feature/tests/data/TransformedX75Y75.png": "a1e9ead5f8e4a0f604271e1f9c50e89baf53f068f1d19fab2876af4938e695ea",
    }
)

fetch = image_fetcher.fetch

data_dir = image_fetcher.abspath

# TODO: Load the requested readme by default

def download_all(directory=None):
    """Download all datasets for use with scikit-image offline.

    Scikit-image datasets are no longer shipped with the library by default.
    This allows us to use higher quality datasets, while keeping the
    library download size small.

    Call this function to download all sample images making them available
    offline on your machine.

    Parameters
    ----------
    directory: path-like, optional
        The directory where the dataset should be stored.

    Notes
    -----
    scikit-image will only search for images stored in the default directory.
    Only specify the directory if you wish to download the images to your own
    folder for a particular reason. You may access the location of the data
    directory by inspecting the variable `skimage.data.data_dir`.
    """

    # Consider moving this kind of logic to Pooch
    old_dir = image_fetcher.path
    try:
        if directory is not None:
            image_fetcher.path = directory

        for filename in image_fetcher.registry:
            fetch(filename)
    finally:
        image_fetcher.path = old_dir


def lbp_frontal_face_cascade_filename():
    """
    Returns the path to the XML file containing information about the weak
    classifiers of a cascade classifier trained using LBP features. It is part
    of the OpenCV repository [1]_.

    References
    ----------
    .. [1] OpenCV lbpcascade trained files
           https://github.com/Itseez/opencv/tree/master/data/lbpcascades
    """

    return fetch('data/lbpcascade_frontalface_opencv.xml')

def load(f, as_gray=False):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.

    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.
    """
    return imread(fetch(f), plugin='pil', as_gray=as_gray)


def camera():
    """Gray-level "camera" image.

    Often used for segmentation and denoising examples.

    Returns
    -------
    camera : (512, 512) uint8 ndarray
        Camera image.
    """
    return load("data/camera.png")


def astronaut():
    """Color image of the astronaut Eileen Collins.

    Photograph of Eileen Collins, an American astronaut. She was selected
    as an astronaut in 1992 and first piloted the space shuttle STS-63 in
    1995. She retired in 2006 after spending a total of 38 days, 8 hours
    and 10 minutes in outer space.

    This image was downloaded from the NASA Great Images database
    <https://flic.kr/p/r9qvLn>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    astronaut : (512, 512, 3) uint8 ndarray
        Astronaut image.
    """

    return load("data/astronaut.png")


def brick():
    """Brick wall.

    Returns
    -------
    brick: (512, 512) uint8 image
        A small section of a brick wall.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Bricks25>`_ and licensed
    under the Creative Commons CC0 License.

    A perspective transform was then applied to the image, prior to
    rotating it by 90 degrees, cropping and scaling it to obtain the final
    image.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Bricks25/%5B2K%5DBricks25.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Bricks25.zip', 'bw') as f:
    ...     f.write(r.content)
    >>> with zipfile.ZipFile('[2K]Bricks25.zip') as z:
    ... z.extract('Bricks25_col.jpg')

    >>> from numpy.linalg import inv
    >>> from skimage.transform import rescale, warp, rotate
    >>> from skimage.color import rgb2gray
    >>> from imageio import imread, imwrite
    >>> from skimage import img_as_ubyte
    >>> import numpy as np


    >>> # Obtained playing around with GIMP 2.10 with their perspective tool
    >>> H = inv(np.asarray([[ 0.54764, -0.00219, 0],
    ...                     [-0.12822,  0.54688, 0],
    ...                     [-0.00022,        0, 1]]))


    >>> brick_orig = imread('Bricks25_col.jpg')
    >>> brick = warp(brick_orig, H)
    >>> brick = rescale(brick[:1024, :1024], (0.5, 0.5, 1))
    >>> brick = rotate(brick, -90)
    >>> imwrite('brick.png', img_as_ubyte(rgb2gray(brick)))
    """
    return load("data/brick.png", as_gray=True)


def grass():
    """Grass.

    Returns
    -------
    grass: (512, 512) uint8 image
        Some grass.

    Notes
    -----
    The original image was downloaded from
    `DeviantArt <https://www.deviantart.com/linolafett/art/Grass-01-434853879>`__
    and licensed underthe Creative Commons CC0 License.

    The downloaded image was cropped to include a region of ``(512, 512)``
    pixels around the top left corner, converted to grayscale, then to uint8
    prior to saving the result in PNG format.

    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a407467e-4ff0-49f1-923f-c9e388e84612/d76wfef-2878b78d-5dce-43f9-be36->>     26ec9bc0df3b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2E0MDc0NjdlLTRmZjAtNDlmMS05MjNmLWM5ZTM4OGU4NDYxMlwvZDc2d2ZlZi0yODc4Yjc4ZC01ZGNlLTQzZjktYmUzNi0yNmVjOWJjMGRmM2IuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.98hIcOTCqXWQ67Ec5bM5eovKEn2p91mWB3uedH61ynI'
    >>> r = requests.get(url)
    >>> with open('grass_orig.jpg', 'bw') as f:
    ...     f.write(r.content)
    >>> grass_orig = imageio.imread('grass_orig.jpg')
    >>> grass = skimage.img_as_ubyte(skimage.color.rgb2gray(grass_orig[:512, :512]))
    >>> imageio.imwrite('grass.png', grass)
    """
    return load("data/grass.png", as_gray=True)


def rough_wall():
    """Rough wall.

    Returns
    -------
    rough_wall: (512, 512) uint8 image
        Some rough wall.

    """
    from warnings import warn
    warn("The rough_wall dataset has been removed due to licensing concerns."
         "It has been replaced with the gravel dataset. This warning message"
         "will be replaced with an error in scikit-image 0.17.", stacklevel=2)
    return gravel()


def gravel():
    """Gravel

    Returns
    -------
    gravel: (512, 512) uint8 image
        Grayscale gravel sample.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Gravel04>`__ and
    licensed under the Creative Commons CC0 License.

    The downloaded image was then rescaled to ``(1024, 1024)``, then the
    top left ``(512, 512)`` pixel region  was cropped prior to converting the
    image to grayscale and uint8 data type. The result was saved using the
    PNG format.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile

    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Gravel04/%5B2K%5DGravel04.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Gravel04.zip', 'bw') as f:
    ...     f.write(r.content)

    >>> with zipfile.ZipFile('[2K]Gravel04.zip') as z:
    ...     z.extract('Gravel04_col.jpg')

    >>> from skimage.transform import resize
    >>> gravel_orig = imageio.imread('Gravel04_col.jpg')
    >>> gravel = resize(gravel_orig, (1024, 1024))
    >>> gravel = skimage.img_as_ubyte(skimage.color.rgb2gray(gravel[:512, :512]))
    >>> imageio.imwrite('gravel.png', gravel)
    """
    return load("data/gravel.png", as_gray=True)


def text():
    """Gray-level "text" image used for corner detection.

    Notes
    -----
    This image was downloaded from Wikipedia
    <https://en.wikipedia.org/wiki/File:Corner.png>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    text : (172, 448) uint8 ndarray
        Text image.
    """

    return load("data/text.png")


def checkerboard():
    """Checkerboard image.

    Checkerboards are often used in image calibration, since the
    corner-points are easy to locate.  Because of the many parallel
    edges, they also visualise distortions particularly well.

    Returns
    -------
    checkerboard : (200, 200) uint8 ndarray
        Checkerboard image.
    """
    return load("data/chessboard_GRAY.png")


def coins():
    """Greek coins from Pompeii.

    This image shows several coins outlined against a gray background.
    It is especially useful in, e.g. segmentation tests, where
    individual objects need to be identified against a background.
    The background shares enough grey levels with the coins that a
    simple segmentation is not sufficient.

    Notes
    -----
    This image was downloaded from the
    `Brooklyn Museum Collection
    <https://www.brooklynmuseum.org/opencollection/archives/image/51611>`__.

    No known copyright restrictions.

    Returns
    -------
    coins : (303, 384) uint8 ndarray
        Coins image.
    """
    return load("data/coins.png")


def logo():
    """Scikit-image logo, a RGBA image.

    Returns
    -------
    logo : (500, 500, 4) uint8 ndarray
        Logo image.
    """
    return load("data/logo.png")


def microaneurysms():
    """Gray-level "microaneurysms" image.

    Detail from an image of the retina (green channel).
    The image is a crop of image 07_dr.JPG from the
    High-Resolution Fundus (HRF) Image Database:
    https://www5.cs.fau.de/research/data/fundus-images/

    Notes
    -----
    No copyright restrictions. CC0 given by owner (Andreas Maier).

    Returns
    -------
    microaneurysms : (102, 102) uint8 ndarray
        Retina image with lesions.

    References
    ----------
    .. [1] Budai, A., Bock, R, Maier, A., Hornegger, J.,
           Michelson, G. (2013).  Robust Vessel Segmentation in Fundus
           Images. International Journal of Biomedical Imaging, vol. 2013,
           2013.
           :DOI:`10.1155/2013/154860`
    """
    return load("data/microaneurysms.png")


def moon():
    """Surface of the moon.

    This low-contrast image of the surface of the moon is useful for
    illustrating histogram equalization and contrast stretching.

    Returns
    -------
    moon : (512, 512) uint8 ndarray
        Moon image.
    """
    return load("data/moon.png")


def page():
    """Scanned page.

    This image of printed text is useful for demonstrations requiring uneven
    background illumination.

    Returns
    -------
    page : (191, 384) uint8 ndarray
        Page image.
    """
    return load("data/page.png")


def horse():
    """Black and white silhouette of a horse.

    This image was downloaded from
    `openclipart <http://openclipart.org/detail/158377/horse-by-marauder>`

    No copyright restrictions. CC0 given by owner (Andreas Preuss (marauder)).

    Returns
    -------
    horse : (328, 400) bool ndarray
        Horse image.
    """
    return img_as_bool(load("data/horse.png", as_gray=True))


def clock():
    """Motion blurred clock.

    This photograph of a wall clock was taken while moving the camera in an
    aproximately horizontal direction.  It may be used to illustrate
    inverse filters and deconvolution.

    Released into the public domain by the photographer (Stefan van der Walt).

    Returns
    -------
    clock : (300, 400) uint8 ndarray
        Clock image.
    """
    return load("data/clock_motion.png")


def immunohistochemistry():
    """Immunohistochemical (IHC) staining with hematoxylin counterstaining.

    This picture shows colonic glands where the IHC expression of FHL2 protein
    is revealed with DAB. Hematoxylin counterstaining is applied to enhance the
    negative parts of the tissue.

    This image was acquired at the Center for Microscopy And Molecular Imaging
    (CMMI).

    No known copyright restrictions.

    Returns
    -------
    immunohistochemistry : (512, 512, 3) uint8 ndarray
        Immunohistochemistry image.
    """
    return load("data/ihc.png")


def chelsea():
    """Chelsea the cat.

    An example with texture, prominent edges in horizontal and diagonal
    directions, as well as features of differing scales.

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Stefan van der Walt).

    Returns
    -------
    chelsea : (300, 451, 3) uint8 ndarray
        Chelsea image.
    """
    return load("data/chelsea.png")


def coffee():
    """Coffee cup.

    This photograph is courtesy of Pikolo Espresso Bar.
    It contains several elliptical shapes as well as varying texture (smooth
    porcelain to course wood grain).

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Rachel Michetti).

    Returns
    -------
    coffee : (400, 600, 3) uint8 ndarray
        Coffee image.
    """
    return load("data/coffee.png")


def hubble_deep_field():
    """Hubble eXtreme Deep Field.

    This photograph contains the Hubble Telescope's farthest ever view of
    the universe. It can be useful as an example for multi-scale
    detection.

    Notes
    -----
    This image was downloaded from
    `HubbleSite
    <http://hubblesite.org/newscenter/archive/releases/2012/37/image/a/>`__.

    The image was captured by NASA and `may be freely used in the public domain
    <http://www.nasa.gov/audience/formedia/features/MP_Photo_Guidelines.html>`_.

    Returns
    -------
    hubble_deep_field : (872, 1000, 3) uint8 ndarray
        Hubble deep field image.
    """
    return load("data/hubble_deep_field.jpg")


def retina():
    """Human retina.

    This image of a retina is useful for demonstrations requiring circular
    images.

    Notes
    -----
    This image was downloaded from
    `wikimedia <https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_left_eye.jpg>`.
    This file is made available under the Creative Commons CC0 1.0 Universal
    Public Domain Dedication.

    References
    ----------
    .. [1] Häggström, Mikael (2014). "Medical gallery of Mikael Häggström 2014".
           WikiJournal of Medicine 1 (2). :DOI:`10.15347/wjm/2014.008`.
           ISSN 2002-4436. Public Domain

    Returns
    -------
    retina : (1411, 1411, 3) uint8 ndarray
        Retina image in RGB.
    """
    return load("data/retina.jpg")


def shepp_logan_phantom():
    """Shepp Logan Phantom.

    References
    ----------
    .. [1] L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head
           section," in IEEE Transactions on Nuclear Science, vol. 21,
           no. 3, pp. 21-43, June 1974. :DOI:`10.1109/TNS.1974.6499235`

    Returns
    -------
    phantom: (400, 400) float64 image
        Image of the Shepp-Logan phantom in grayscale.
    """
    return load("data/phantom.png", as_gray=True)


def colorwheel():
    """Color Wheel.

    Returns
    -------
    colorwheel: (370, 371, 3) uint8 image
        A colorwheel.
    """
    return load("data/color.png")


def rocket():
    """Launch photo of DSCOVR on Falcon 9 by SpaceX.

    This is the launch photo of Falcon 9 carrying DSCOVR lifted off from
    SpaceX's Launch Complex 40 at Cape Canaveral Air Force Station, FL.

    Notes
    -----
    This image was downloaded from
    `SpaceX Photos
    <https://www.flickr.com/photos/spacexphotos/16511594820/in/photostream/>`__.

    The image was captured by SpaceX and `released in the public domain
    <http://arstechnica.com/tech-policy/2015/03/elon-musk-puts-spacex-photos-into-the-public-domain/>`_.

    Returns
    -------
    rocket : (427, 640, 3) uint8 ndarray
        Rocket image.
    """
    return load("data/rocket.jpg")


def stereo_motorcycle():
    """Rectified stereo image pair with ground-truth disparities.

    The two images are rectified such that every pixel in the left image has
    its corresponding pixel on the same scanline in the right image. That means
    that both images are warped such that they have the same orientation but a
    horizontal spatial offset (baseline). The ground-truth pixel offset in
    column direction is specified by the included disparity map.

    The two images are part of the Middlebury 2014 stereo benchmark. The
    dataset was created by Nera Nesic, Porter Westling, Xi Wang, York Kitajima,
    Greg Krathwohl, and Daniel Scharstein at Middlebury College. A detailed
    description of the acquisition process can be found in [1]_.

    The images included here are down-sampled versions of the default exposure
    images in the benchmark. The images are down-sampled by a factor of 4 using
    the function `skimage.transform.downscale_local_mean`. The calibration data
    in the following and the included ground-truth disparity map are valid for
    the down-sampled images::

        Focal length:           994.978px
        Principal point x:      311.193px
        Principal point y:      254.877px
        Principal point dx:      31.086px
        Baseline:               193.001mm

    Returns
    -------
    img_left : (500, 741, 3) uint8 ndarray
        Left stereo image.
    img_right : (500, 741, 3) uint8 ndarray
        Right stereo image.
    disp : (500, 741, 3) float ndarray
        Ground-truth disparity map, where each value describes the offset in
        column direction between corresponding pixels in the left and the right
        stereo images. E.g. the corresponding pixel of
        ``img_left[10, 10 + disp[10, 10]]`` is ``img_right[10, 10]``.
        NaNs denote pixels in the left image that do not have ground-truth.

    Notes
    -----
    The original resolution images, images with different exposure and
    lighting, and ground-truth depth maps can be found at the Middlebury
    website [2]_.

    References
    ----------
    .. [1] D. Scharstein, H. Hirschmueller, Y. Kitajima, G. Krathwohl, N.
           Nesic, X. Wang, and P. Westling. High-resolution stereo datasets
           with subpixel-accurate ground truth. In German Conference on Pattern
           Recognition (GCPR 2014), Muenster, Germany, September 2014.
    .. [2] http://vision.middlebury.edu/stereo/data/scenes2014/

    """
    filename = fetch("data/motorcycle_disp.npz")
    disp = np.load(filename)['arr_0']
    return (load("data/motorcycle_left.png"),
            load("data/motorcycle_right.png"),
            disp)


def lfw_subset():
    """Subset of data from the LFW dataset.

    This database is a subset of the LFW database containing:

    * 100 faces
    * 100 non-faces

    The full dataset is available at [2]_.

    Returns
    -------
    images : (200, 25, 25) uint8 ndarray
        100 first images are faces and subsequent 100 are non-faces.

    Notes
    -----
    The faces were randomly selected from the LFW dataset and the non-faces
    were extracted from the background of the same dataset. The cropped ROIs
    have been resized to a 25 x 25 pixels.

    References
    ----------
    .. [1] Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012).
           Learning to align from scratch. In Advances in Neural Information
           Processing Systems (pp. 764-772).
    .. [2] http://vis-www.cs.umass.edu/lfw/

    """
    return np.load(fetch('data/lfw_subset.npy'))
