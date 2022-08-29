/* Brush

brushgp.cc : Python interface to Brush classes and functions, using pybind11

copyright 2021 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "program.h"
#include "search_space.h"
#include "data/data.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


// Test data as numpy arrays:
// X = np.array([[1.1, 2.0, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0],
//               [2.0, 1.2, 6.0, 4.0, 5.0, 8.0, 7.0, 5.0, 9.0, 10.0]])
//
// y = np.array( [1.0, 0.0, 1.4, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0])

PYBIND11_MODULE(brushgp, m) {
    m.doc() = R"pbdoc(
        Python interface for Brush
        --------------------------

        .. currentmodule:: brushgp

        .. autosummary::
           :toctree: _generate

           Data
           SearchSpace
           Program
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<Brush::data::Data>(m, "Data")
        .def(py::init<Ref<const ArrayXXf>&, Ref<const ArrayXf>& >())
        // .def("get_features", &Brush::data::Data::features, py::return_value_policy::reference_internal)
        // .def("get_y", &Brush::data::Data::get_y, py::return_value_policy::reference_internal)        
        ;

    // Notice: We change the interface for SearchSpace a little bit by 
    // constructing it with a Data object, rather than initializing it as an
    // empty struct and then calling init() with the Data object.
    py::class_<Brush::SearchSpace>(m, "SearchSpace")
        .def(py::init([](Brush::data::Data data) {
            SearchSpace SS;
            SS.init(data);
            return SS;
        }))
        .def("make_program", &Brush::SearchSpace::make_program<ArrayXf>)
        ;

    py::class_<Brush::Program<ArrayXf> >(m, "Program")
        .def(py::init<>())
        .def("fit", &Brush::Program<ArrayXf>::fit)
        .def("predict", &Brush::Program<ArrayXf>::predict)
        .def("get_model", &Brush::Program<ArrayXf>::get_model, 
                py::arg("type")="compact", py::arg("pretty")=false)
        ;
}
