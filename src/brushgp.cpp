/* Brush

brushgp.cc : Python interface to Brush classes and functions, using pybind11

copyright 2021 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "program/program.h"
#include "search_space.h"
#include "data/data.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace br = Brush;
// using br::Data;


// Test data as numpy arrays:
// X = np.array([[1.1, 2.0, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0],
//               [2.0, 1.2, 6.0, 4.0, 5.0, 8.0, 7.0, 5.0, 9.0, 10.0]])
//
// y = np.array( [1.0, 0.0, 1.4, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0])

PYBIND11_MODULE(brush, m) {
    m.doc() = R"pbdoc(
        Python interface for Brush
        --------------------------

        .. currentmodule:: brush.core

        .. autosummary::
           :toctree: _generate

           Dataset
           SearchSpace
           Regressor
           Classifier
           MulticlassClassifier
           Representer
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<Brush::Data::Dataset>(m, "Dataset")
        .def(py::init<Ref<const ArrayXXf>&, Ref<const ArrayXf>& >())
        ;

    // Notice: We change the interface for SearchSpace a little bit by 
    // constructing it with a Dataset object, rather than initializing it as an
    // empty struct and then calling init() with the Dataset object.
    py::class_<br::SearchSpace>(m, "SearchSpace")
        .def(py::init([](br::Data::Dataset data) {
            SearchSpace SS;
            SS.init(data);
            return SS;
        }))
        .def("make_regressor", &br::SearchSpace::make_regressor)
        .def("make_classifier", &br::SearchSpace::make_classifier)
        .def("make_multiclass_classifier", &br::SearchSpace::make_multiclass_classifier)
        .def("make_representer", &br::SearchSpace::make_representer)
        ;

    using Reg = br::Program<ArrayXf>;

    py::class_<Reg>(m, "Regressor")
        .def(py::init<>())
        .def("fit",
             static_cast<Reg &(Reg::*)(const Dataset &d)>(&Reg::fit),
             "fit from Dataset object")
        .def("fit",
             static_cast<Reg &(Reg::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>
                (&Reg::fit),
             "fit from X,y data")
        .def("predict",
             static_cast<ArrayXf (Reg::*)(const Dataset &d)>(&Reg::predict),
             "predict from Dataset object")
        .def("predict",
             static_cast<ArrayXf (Reg::*)(const Ref<const ArrayXXf> &X)>(&Reg::predict),
             "fit from X,y data")
        .def("get_model",
             &Reg::get_model,
             py::arg("type") = "compact",
             py::arg("pretty") = false);

    using Cls = br::Program<ArrayXb>;
    py::class_<Cls>(m, "Classifier")
        .def(py::init<>())
        .def("fit",
             static_cast<Cls &(Cls::*)(const Dataset &d)>(&Cls::fit),
             "fit from Dataset object")
        .def("fit",
             static_cast<Cls &(Cls::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>
                (&Cls::fit),
             "fit from X,y data")
        .def("predict",
             static_cast<ArrayXb (Cls::*)(const Dataset &d)>(&Cls::predict),
             "predict from Dataset object")
        .def("predict",
             static_cast<ArrayXb (Cls::*)(const Ref<const ArrayXXf> &X)>(&Cls::predict),
             "fit from X,y data")
        .def("predict_proba",
             static_cast<ArrayXf (Cls::*)(const Dataset &d)>(&Cls::predict_proba),
             "predict from Dataset object")
        .def("predict_proba",
             static_cast<ArrayXf (Cls::*)(const Ref<const ArrayXXf> &X)>(&Cls::predict_proba),
             "fit from X,y data")
        .def("get_model",
             &Cls::get_model,
             py::arg("type") = "compact",
             py::arg("pretty") = false);

    using MCls = br::Program<ArrayXi>;
    py::class_<MCls>(m, "MulticlassClassifer")
        .def(py::init<>())
        .def("fit",
             static_cast<MCls &(MCls::*)(const Dataset &d)>(&MCls::fit),
             "fit from Dataset object")
        .def("fit",
             static_cast<MCls &(MCls::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>
                (&MCls::fit),
             "fit from X,y data")
        .def("predict",
             static_cast<ArrayXi (MCls::*)(const Dataset &d)>(&MCls::predict),
             "predict from Dataset object")
        .def("predict",
             static_cast<ArrayXi (MCls::*)(const Ref<const ArrayXXf> &X)>(&MCls::predict),
             "fit from X,y data")
        .def("get_model",
             &MCls::get_model,
             py::arg("type") = "compact",
             py::arg("pretty") = false);

    using Rep = br::Program<ArrayXXf>;
    py::class_<Rep>(m, "Representer")
        .def(py::init<>())
        .def("fit",
             static_cast<Rep &(Rep::*)(const Dataset &d)>(&Rep::fit),
             "fit from Dataset object")
        .def("fit",
             static_cast<Rep &(Rep::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>
                (&Rep::fit),
             "fit from X,y data")
        .def("predict",
             static_cast<ArrayXXf (Rep::*)(const Dataset &d)>(&Rep::predict),
             "predict from Dataset object")
        .def("predict",
             static_cast<ArrayXXf (Rep::*)(const Ref<const ArrayXXf> &X)>(&Rep::predict),
             "fit from X,y data")
        .def("get_model",
             &Rep::get_model,
             py::arg("type") = "compact",
             py::arg("pretty") = false);

}
