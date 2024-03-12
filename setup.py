from __future__ import print_function
import os.path
import sys
from numpy.distutils.core import setup


try:
    import numpy
except ImportError:
    print("numpy is required during installation")
    sys.exit(1)


DISTNAME = "Soft-DTW"
DESCRIPTION = "Soft-DTW loss function for Keras/TensorFlow"
LONG_DESCRIPTION = ""
MAINTAINER = "AISViz"
MAINTAINER_EMAIL = "aisviz@dal.ca"
URL = "https://github.com/aisviz/Soft-DTW"
LICENSE = "https://github.com/aisviz/Soft-DTW/LICENSE"
DOWNLOAD_URL = "https://github.com/JayKumarr/Soft-DTW"
VERSION = "1.0.0"


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage("softdtwkeras")

    return config


if __name__ == "__main__":
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              "Intended Audience :: Developers",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
              "Operating System :: MacOS" "Operating System :: POSIX :: Linux",
              "Operating System :: Microsoft :: Windows",
              "Operating System :: Unix",
              "Programming Language :: C",
              "Programming Language :: Python :: 3.10",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "Topic :: Scientific/Engineering :: Information Analysis",
              "Topic :: Software Development",
              "Topic :: Utilities",
        ]
    )
