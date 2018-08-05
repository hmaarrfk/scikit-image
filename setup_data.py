from distutils.version import LooseVersion
import setuptools
from setup import (VERSION, DISTNAME, DESCRIPTION, LONG_DESCRIPTION,
                   MAINTAINER, MAINTAINER_EMAIL, URL, LICENSE, DOWNLOAD_URL,
                   classifiers)

if 'dev' not in VERSION:
    data_VERSION_REQ = 'skimage (=={})'.format(VERSION)
else:
    data_VERSION_REQ = 'skimage (>='
    for v in LooseVersion(VERSION).version:
        if 'dev' == v:
            break
        data_VERSION_REQ += str(v) + '.'
    # erase the trailing dot ``.``
    # close the parenthesis
    data_VERSION_REQ = data_VERSION_REQ[:-1] + ')'

setuptools.setup(
    name=(DISTNAME+'-data'),
    description='Data files to run examples and tests for scikit-image',
    long_description='Data files for the ' + LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    license=LICENSE,
    download_url=DOWNLOAD_URL,
    version=VERSION,
    requires=[data_VERSION_REQ],
    python_requires='>3.5',
    packages=setuptools.find_packages(
        include=['skimage.data', "*.tests", "*.tests.*", "tests.*"]
        ),
    include_package_data=True,
    zip_safe=False  # the package can run out of an .egg file
    )
