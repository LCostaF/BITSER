import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extra_compile_args = ['-O3']
if sys.platform != 'win32':
    extra_compile_args.append('-std=c++11')

extensions = [
    Extension(
        'bitser.genetic_texture_analysis',
        ['bitser/genetic_texture_analysis.pyx'],
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='bitser',
    packages=['bitser'],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
        },
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    package_data={'bitser': ['*.pyd', '*.so']},
    exclude_package_data={'bitser': ['*.pyx', '*.cpp']},
)
