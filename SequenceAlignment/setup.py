from distutils.core import setup, Extension
import numpy

c_ext = Extension("_SequenceAlignment", ["_SequenceAlignment.c", "swalignimp.c"], include_dirs=[numpy.get_include()])

setup(
    ext_modules=[c_ext],
    include_dirs=[numpy.get_include()],
)
