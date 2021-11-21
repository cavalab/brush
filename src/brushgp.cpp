/* Brush

brushgp.cc : Python interface to Brush classes and functions, using pybind11

copyright 2021 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "program.h"
#include "search_space.h"
#include "data/data.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


PYBIND11_MODULE(brushgp, m) {
    m.doc() = "Python interface for Brush";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<Brush::data::Data>(m, "Data")
        .def(py::init<ArrayXXf, ArrayXf>());

    // Notice: We change the interface for SearchSpace a little bit by 
    // constructing it with a Data object, rather than initializing it as an
    // empty struct and then calling init() with the Data object.
    py::class_<Brush::SearchSpace>(m, "SearchSpace")
        .def(py::init([](Brush::data::Data data) {
            SearchSpace SS;
            SS.init(data);
            return SS;
        }));

    py::class_<Brush::Program<ArrayXf> >(m, "Program")
        .def(py::init<SearchSpace&, int, int, int>())
        .def("fit", &Brush::Program<ArrayXf>::fit);
}