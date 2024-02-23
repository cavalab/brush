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
        .def_property("pop_size", &Brush::Parameters::get_pop_size, &Brush::Parameters::set_pop_size)
        .def_property("gens", &Brush::Parameters::get_gens, &Brush::Parameters::set_gens)
        .def_property("current_gen", &Brush::Parameters::get_current_gen, &Brush::Parameters::set_current_gen)
        .def_property("num_islands", &Brush::Parameters::get_num_islands, &Brush::Parameters::set_num_islands)
        .def_property("max_depth", &Brush::Parameters::get_max_depth, &Brush::Parameters::set_max_depth)
        .def_property("max_size", &Brush::Parameters::get_max_size, &Brush::Parameters::set_max_size)
        .def_property("objectives", &Brush::Parameters::get_objectives, &Brush::Parameters::set_objectives)
        .def_property("sel", &Brush::Parameters::get_sel, &Brush::Parameters::set_sel)
        .def_property("surv", &Brush::Parameters::get_surv, &Brush::Parameters::set_surv)
        .def_property("cx_prob", &Brush::Parameters::get_cx_prob, &Brush::Parameters::set_cx_prob)
        .def_property("mig_prob", &Brush::Parameters::get_mig_prob, &Brush::Parameters::set_mig_prob)
        .def_property("functions", &Brush::Parameters::get_functions, &Brush::Parameters::set_functions)
        .def_property("mutation_probs", &Brush::Parameters::get_mutation_probs, &Brush::Parameters::set_mutation_probs)
        
        ;    
}