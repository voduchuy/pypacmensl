// Credits: https://stackoverflow.com/users/4657412/davidw
#include <Python.h>
#include "armadillo"
#include "_pacmensl_callbacks.h"

class PyConstrWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyConstrWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyConstrWrapper(const PyConstrWrapper& rhs): PyConstrWrapper(rhs.held) { // C++11 onwards only
    }

    PyConstrWrapper(PyConstrWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyConstrWrapper(): PyConstrWrapper(nullptr) {
    }

    ~PyConstrWrapper() {
        Py_XDECREF(held);
    }

    PyConstrWrapper& operator=(const PyConstrWrapper& rhs) {
        PyConstrWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyConstrWrapper& operator=(PyConstrWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    int operator()(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args) {
        if (held) { // nullptr check
            return call_py_constr_obj(held,num_species, num_constr, n_states, states, outputs, args); // note, no way of checking for errors until you return to Python
        }
        else{
            return -1;
        }
    }

private:
    PyObject* held;
};


class PyPropWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyPropWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyPropWrapper(const PyPropWrapper& rhs): PyPropWrapper(rhs.held) { // C++11 onwards only
    }

    PyPropWrapper(PyPropWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyPropWrapper(): PyPropWrapper(nullptr) {
    }

    ~PyPropWrapper() {
        Py_XDECREF(held);
    }

    PyPropWrapper& operator=(const PyPropWrapper& rhs) {
        PyPropWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyPropWrapper& operator=(PyPropWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    int operator()(const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args) {
        if (held) { // nullptr check
            return call_py_prop_obj(held, reaction, num_species, num_states, states, outputs, args);
        }
        else{
            throw std::runtime_error("Propensity wrapper called without a real Python object.");
        }
    }

private:
    PyObject* held;
};

class PyTFunWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyTFunWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyTFunWrapper(const PyTFunWrapper& rhs): PyTFunWrapper(rhs.held) { // C++11 onwards only
    }

    PyTFunWrapper(PyTFunWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyTFunWrapper(): PyTFunWrapper(nullptr) {
    }

    ~PyTFunWrapper() {
        Py_XDECREF(held);
    }

    PyTFunWrapper& operator=(const PyTFunWrapper& rhs) {
        PyTFunWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyTFunWrapper& operator=(PyTFunWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    int operator()(double t, int num_coefs, double* outputs, void* args) {
        if (held) { // nullptr check
            return call_py_t_fun_obj(held, t, num_coefs, outputs, args);
        }
        else{
            return -1;
        }
    }

private:
    PyObject* held;
};