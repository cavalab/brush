#include "module.h"
#include "../params.h"
#include "../util/rnd.h"

namespace br = Brush;

void bind_params(py::module& m)
{
    // py::object params = Brush::PARAMS;
    // m.attr("PARAMS") = params;

    //  py::class_<br::Params>(m, "Params", py::dynamic_attr())
    //       .def(py::init<>())

    m.def("set_params", &Brush::set_params); // TODO: delete this. use parameters class

    m.def("get_params", &br::get_params);
    m.def("set_random_state", [](unsigned int seed)
                                { br::Util::r = *br::Util::Rnd::initRand(); 
                                  br::Util::r.set_seed(seed); });
    m.def("rnd_flt", [](){ return br::Util::r.rnd_flt(); });

    py::class_<Brush::Parameters>(m, "Parameters")
        .def(py::init([]()
                      { Brush::Parameters p; return p; }))
        // TODO: define getters and setters, and create the bindings here. Make the Brush bindings use these here
        .def_property("pop_size", &Brush::Parameters::get_pop_size, &Brush::Parameters::set_pop_size)
        .def_property("mutation_probs", &Brush::Parameters::get_mutation_probs, &Brush::Parameters::set_mutation_probs)
        ;    
}