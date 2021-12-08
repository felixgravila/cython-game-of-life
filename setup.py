from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(
        "gol_worker",
        sources=["gol_worker.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[],
        language="c++",
    )
]

setup(
    name="gol_worker",
    ext_modules=cythonize(extensions),
)
