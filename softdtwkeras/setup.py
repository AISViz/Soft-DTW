import os.path

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('softdtwkeras', parent_package, top_path)

    config.add_extension('soft_dtw_fast', sources=['soft_dtw_fast.c'],
                         include_dirs=[numpy.get_include()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
