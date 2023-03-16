/* Brush

module.cpp : Python interface to Brush classes and functions, using pybind11

copyright 2021 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "module.h"

namespace py = pybind11;

// forward declarations
void bind_dataset(py::module &);
void bind_search_space(py::module &);
void bind_programs(py::module &);
void bind_params(py::module &);

PYBIND11_MODULE(_brush, m) {
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
      
     bind_params(m);
     bind_dataset(m);
     bind_search_space(m);
     py::module_ m2 = m.def_submodule("program", "Contains Program classes.");
     bind_programs(m2);

}
