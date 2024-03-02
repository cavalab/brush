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
void bind_params(py::module &);
void bind_dataset(py::module &);
void bind_search_space(py::module &);
void bind_programs(py::module &);
void bind_variations(py::module &);
void bind_selections(py::module &);
void bind_individuals(py::module &);
void bind_populations(py::module &);
void bind_estimators(py::module &);
void bind_evaluators(py::module &);

PYBIND11_MODULE(_brush, m) {

#ifdef VERSION_INFO
     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
     m.attr("__version__") = "dev";
#endif
     // main algorithm
     // bind_cbrush(m);

     // data structures to store solutions
     bind_params(m);
     bind_dataset(m);
     bind_search_space(m);

     // should these 4 below be exposed?
     bind_variations(m);
     bind_selections(m);
     bind_evaluators(m);
     // bind_populations(m);

     // solutions
     py::module_ m2 = m.def_submodule("program", "Contains Program classes.");
     bind_programs(m2);

     py::module_ m3 = m.def_submodule("individual", "Contains Individual classes.");
     bind_individuals(m3);

     py::module_ m4 = m.def_submodule("engine", "Learning engines (used inside the python estimators)."); 
     bind_estimators(m4);
}
