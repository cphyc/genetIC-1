// Legacy genetIC streams: GSL ranlxs2 and gsl_ran_gaussian_ziggurat.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <algorithm>
#include <cmath>
#include <complex>

constexpr const char *tag = "genetic.ranlxs2";
static gsl_rng *state(PyObject *capsule) {
    return static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, tag));
}
static void destroy(PyObject *capsule) { gsl_rng_free(state(capsule)); }
static PyObject *seed(PyObject *, PyObject *args) {
    unsigned long value;
    if (!PyArg_ParseTuple(args, "k", &value)) return nullptr;
    auto *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
    if (!rng) return PyErr_NoMemory();
    gsl_rng_set(rng, value);
    auto *result = PyCapsule_New(rng, tag, destroy);
    if (!result) gsl_rng_free(rng);
    return result;
}
static PyObject *normal(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "On", &capsule, &count)) return nullptr;
    auto *rng = state(capsule);
    if (!rng) return nullptr;
    if (count < 0 || count > PY_SSIZE_T_MAX / sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Invalid random draw count"); return nullptr;
    }
    auto *result = PyBytes_FromStringAndSize(nullptr, count * sizeof(double));
    if (!result) return nullptr;
    auto *values = reinterpret_cast<double *>(PyBytes_AS_STRING(result));
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < count; ++i) values[i] = gsl_ran_gaussian_ziggurat(rng, 1.);
    Py_END_ALLOW_THREADS
    return result;
}
static PyObject *fourier(PyObject *, PyObject *args) {
    PyObject *capsule;
    unsigned long base_seed;
    int n, parallel, reverse;
    if (!PyArg_ParseTuple(args, "Okipp", &capsule, &base_seed, &n, &parallel, &reverse)) return nullptr;
    auto *rng = state(capsule);
    if (!rng) return nullptr;
    if (n < 2 || n > PY_SSIZE_T_MAX / n / (n / 2 + 1) / sizeof(std::complex<double>)) {
        PyErr_SetString(PyExc_ValueError, "Invalid Fourier grid size"); return nullptr;
    }
    const Py_ssize_t count = Py_ssize_t(n) * n * (n / 2 + 1);
    auto *result = PyBytes_FromStringAndSize(nullptr, count * sizeof(std::complex<double>));
    if (!result) return nullptr;
    auto *values = reinterpret_cast<std::complex<double> *>(PyBytes_AS_STRING(result));
    Py_BEGIN_ALLOW_THREADS
    std::fill(values, values + count, std::complex<double>(0));
    #pragma omp parallel for if(parallel) num_threads(6)
    for (int shell = 1; shell < n / 2; ++shell) {
        auto *local = parallel ? gsl_rng_alloc(gsl_rng_ranlxs2) : rng;
        if (parallel) gsl_rng_set(local, base_seed + shell);
        auto write = [&](int x, int y, int z, std::complex<double> value) {
            auto wrap = [n](int i) { return (i % n + n) % n; };
            x = wrap(x); y = wrap(y); z = wrap(z);
            if (z > n / 2) { x = wrap(-x); y = wrap(-y); z = n - z; value = std::conj(value); }
            values[(Py_ssize_t(x) * n + y) * (n / 2 + 1) + z] = value;
        };
        auto draw = [&](int x, int y, int z) {
            const double a = gsl_ran_gaussian_ziggurat(local, 1.) / std::sqrt(2.);
            const double b = gsl_ran_gaussian_ziggurat(local, 1.) / std::sqrt(2.);
            const std::complex<double> value = reverse ? std::complex<double>(b, a) : std::complex<double>(a, b);
            write(x, y, z, value);
            if (x == 0 || y == 0 || z == 0) write(-x, -y, -z, std::conj(value));
        };
        for (int a = -shell; a < shell; ++a)
            for (int b = -shell; b < shell; ++b) {
                draw(shell, a, b); draw(a, shell, b); draw(a, b, shell);
            }
        if (parallel) gsl_rng_free(local);
    }
    Py_END_ALLOW_THREADS
    return result;
}
static PyMethodDef methods[] = {
    {"seed", seed, METH_VARARGS, nullptr}, {"normal", normal, METH_VARARGS, nullptr},
    {"fourier", fourier, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_random", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__random() { return PyModule_Create(&module); }
