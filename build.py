import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def get_extensions():
    """
    Returns the list of Cython extension modules.
    Shared between the Poetry build hook and the direct setup() call.
    """
    extra_compile_args = ['-O3']
    if sys.platform != 'win32':
        extra_compile_args.append('-std=c++11')

    return [
        Extension(
            'bitser.genetic_texture_analysis',
            ['bitser/genetic_texture_analysis.pyx'],
            include_dirs=[numpy.get_include()],
            language='c++',
            extra_compile_args=extra_compile_args,
        )
    ]


def get_cythonized(extensions):
    """
    Runs cythonize on the extensions with shared compiler directives.
    """
    return cythonize(
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
    )


def build(setup_kwargs):
    """
    Poetry build hook — called by `poetry build` and `poetry install`.
    """
    setup_kwargs.update(
        {
            'ext_modules': get_cythonized(get_extensions()),
            'include_dirs': [numpy.get_include()],
            'zip_safe': False,
        }
    )
    return setup_kwargs


if __name__ == '__main__' and len(sys.argv) > 1:
    """
    Direct invocation — called by `python build.py build_ext --inplace`.
    packages=['bitser'] prevents setuptools from auto-discovering all the
    local data folders in your project root as packages.
    """
    setup(
        packages=['bitser'],
        ext_modules=get_cythonized(get_extensions()),
        include_dirs=[numpy.get_include()],
    )
