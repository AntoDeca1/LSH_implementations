from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Fetch the NumPy include directory.
numpy_include_dir = numpy.get_include()

# Define the extension module with the include directories
extensions = [
    Extension("cosine_similarity_fast", ["cosine_similarity_fast.pyx"],
              include_dirs=[numpy_include_dir])
]

setup(
    name='Cosine Similarity',
    ext_modules=cythonize(extensions)
)
