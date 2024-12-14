## 
## Python interface for djvu files
##
## Author : Sergey Mikhno
## Created: 2024-09-07
##

from cpython.bytes cimport PyBytes_FromStringAndSize
import numpy as np

cdef extern from "wrapper.h":
    ctypedef struct djvu_document:
        int width
        int height
        char* pixels

    cdef int get_djvu_number_of_pages(char* native_path)
    cdef djvu_document get_djvu_page(int pageno, char* filepath)


cpdef get_page_bytes(int pageno, char* filepath):
    doc = get_djvu_page(pageno, filepath)
    return PyBytes_FromStringAndSize(doc.pixels, doc.width * doc.height), doc.width, doc.height


def get_page(pageno, filepath):
    bpath = bytes(filepath, 'utf-8')
    return get_page_bytes(pageno, bpath)


def get_number_of_pages(native_path):
    bpath = bytes(native_path, "utf-8")
    n = get_djvu_number_of_pages(bpath)
    return n


def get_image_as_arrray(pageno, filepath):
    bs, w, h = get_page(pageno, filepath)
    return np.frombuffer(bs, dtype=np.uint8).reshape((h,w))
