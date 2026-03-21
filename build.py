# build.py

import sys

import numpy

from Cython.Build import cythonize
from setuptools import Extension


def build(setup_kwargs):
    """
    This function is mandatory to build the extensions with Poetry.
    """

    # Platform-conditional compile args
    extra_compile_args = ['-O3']
    if sys.platform != 'win32':
        extra_compile_args.append('-std=c++11')

    # Define the extension module
    extensions = [
        Extension(
            'bitser.genetic_texture_analysis',
            ['bitser/genetic_texture_analysis.pyx'],
            include_dirs=[numpy.get_include()],
            language='c++',
            extra_compile_args=extra_compile_args,
        )
    ]

    # Update build parameters
    setup_kwargs.update(
        {
            'ext_modules': cythonize(
                extensions,
                compiler_directives={
                    'language_level': '3',
                    'boundscheck': False,
                    'wraparound': False,
                    'initializedcheck': False,
                    'nonecheck': False,
                    'cdivision': True,
                },
                annotate=True,
            ),
            'include_dirs': [numpy.get_include()],
            'zip_safe': False,
        }
    )

    return setup_kwargs