/*Programmer: Chris Tralie
*Purpose: A wrapper for Python around fast Smith Waterman code in C
*Helps from following links:
*http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
*https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html
*Special thanks especially to http://dan.iel.fm/posts/python-c-extensions/*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "swalignimp.h"

/* Docstrings */
static char module_docstring[] =
    "This module provides an implementation of implicit Smith Waterman with and without diagonal constraints";
static char swalignimp_docstring[] =
    "Perform Smith Waterman on a binary matrix";
static char swalignimpconstrained_docstring[] =
    "Perform Smith Waterman with diagonal constraints on a binary matrix";

/* Available functions */
static PyObject *SequenceAlignment_swalignimp(PyObject *self, PyObject *args);
static PyObject *SequenceAlignment_swalignimpconstrained(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"swalignimp", SequenceAlignment_swalignimp, METH_VARARGS, swalignimp_docstring},
    {"swalignimpconstrained", SequenceAlignment_swalignimpconstrained, METH_VARARGS, swalignimpconstrained_docstring},
    {NULL, NULL, 0, NULL}
};



#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef Alignments =
{
    PyModuleDef_HEAD_INIT,
    "SequenceAlignment", /* name of module */
    module_docstring, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

/* Initialize the module */
PyMODINIT_FUNC PyInit__SequenceAlignment(void)
{
    /* Load `numpy` functionality. */
    import_array();
    return PyModule_Create(&Alignments);
}
#else
/* Initialize the module */
PyMODINIT_FUNC init_SequenceAlignment(void)
{
    PyObject *m = Py_InitModule3("_SequenceAlignment", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}
#endif


static PyObject *SequenceAlignment_swalignimp(PyObject *self, PyObject *args)
{
    PyObject *S_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &S_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int N = (int)PyArray_DIM(S_array, 0);
    int M = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform Smith Waterman */
    double score = swalignimp(S, N, M);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}

static PyObject *SequenceAlignment_swalignimpconstrained(PyObject *self, PyObject *args)
{
    PyObject *S_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &S_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int N = (int)PyArray_DIM(S_array, 0);
    int M = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform Smith Waterman */
    double score = swalignimpconstrained(S, N, M);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}
