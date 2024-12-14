from distutils.core import setup, Extension
import numpy
import os

library_dirs = ["lib/" + os.environ["OS"] + "/" + os.environ["ARCH"]]

extension = Extension(
    "utils",
    ["utils.c", "pdf.c"],
    libraries=["pdfium"],
    library_dirs=library_dirs,
    include_dirs=["include", numpy.get_include()],
)
setup(name="utils", version="0.1.7", ext_modules=[extension])
