/* Brush

brushgp.cc : Python interface to brush classes and functions, using pybind11

copyright 2020 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>

#include "program.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

PYBIND11_MODULE(brushgp, m) {
    py::class_<Brush::Program<float> >(m, "Program")
        .def(py::init(&Brush::Program<float>::create_py));
        // .def("fit", &Brush::Program<float>::fit_py)
        // .def("predict", &Brush::Program<float>::predict_py)
        // .def("loss", &Brush::Program<float>::loss_py)
        // .def("d_loss", &Brush::Program<float>::d_loss_py)
}