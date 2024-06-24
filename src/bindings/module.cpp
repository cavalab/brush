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

// forward declarations ------------------

// non-templated bindings
void bind_params(py::module &);
void bind_dataset(py::module &);
void bind_search_space(py::module &);
void bind_fitness(py::module &);

// templated bindings
void bind_programs(py::module &);
void bind_evaluators(py::module &);
void bind_individuals(py::module &);
void bind_variations(py::module &);
void bind_selections(py::module &);
void bind_engines(py::module &);

PYBIND11_MODULE(_brush, m) {

#ifdef VERSION_INFO
     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
     m.attr("__version__") = "dev"; // TODO: uve version file
#endif
     // data structures
     bind_params(m);
     bind_dataset(m);
     bind_search_space(m);
     bind_fitness(m);

     // TODO: create a submodule for them
     bind_evaluators(m);
     bind_selections(m);
     bind_variations(m);

     // solutions
     py::module_ m2 = m.def_submodule("program", "Contains Program classes.");
     bind_programs(m2);

     py::module_ m3 = m.def_submodule("individual", "Contains Individual classes.");
     bind_individuals(m3);

     py::module_ m4 = m.def_submodule("engine", "Learning engines."); 
     bind_engines(m4);
}
