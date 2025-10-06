# build.py
import os
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension


def build(setup_kwargs):
    """
    This function is mandatory to build the extensions with Poetry.
    """
    # Define the extension module
    extensions = [
        Extension(
            'bitser.genetic_texture_analysis',  # Update with appropriate module path
            ['bitser/genetic_texture_analysis.pyx'],  # Update path if needed
            include_dirs=[numpy.get_include()],
            language='c++',
            extra_compile_args=['-std=c++11', '-O3', '-march=native'],
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
