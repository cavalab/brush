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
namespace br = Brush;
// using br::data;


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

    py::class_<br::data::Data>(m, "Data")
        .def(py::init<ArrayXXf, ArrayXf>())
        // .def("get_X", &br::data::Data::get_X, py::return_value_policy::reference_internal)
        ;

    // Notice: We change the interface for SearchSpace a little bit by 
    // constructing it with a Data object, rather than initializing it as an
    // empty struct and then calling init() with the Data object.
    py::class_<br::SearchSpace>(m, "SearchSpace")
        .def(py::init([](br::data::Data data) {
            SearchSpace SS;
            SS.init(data);
            return SS;
        }))
        .def("make_program", &br::SearchSpace::make_program<ArrayXf>)
        ;
    using Prg = br::Program<ArrayXf>;

    py::class_<Prg>(m, "Program")
        .def(py::init<>())
        .def("fit", static_cast<ArrayXf(br::Program<ArrayXf>::*)(const Data& d)>(&br::Program<ArrayXf>::fit), "fit from Data object")
        .def("fit", static_cast<ArrayXf(br::Program<ArrayXf>::*)(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)>(&br::Program<ArrayXf>::fit), 
            "fit from X,y data")
        .def("predict", static_cast<ArrayXf(br::Program<ArrayXf>::*)(const Data& d)>(&br::Program<ArrayXf>::fit), "fit from Data object")
        .def("predict", static_cast<ArrayXf(br::Program<ArrayXf>::*)(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)>(&br::Program<ArrayXf>::fit), 
            "fit from X,y data")
        .def("get_model", &br::Program<ArrayXf>::get_model, 
                py::arg("type")="compact", py::arg("pretty")=false)
        ;
}
