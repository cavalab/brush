#include "module.h"
#include "../params.h"
#include "../util/rnd.h"

namespace br = Brush;

void bind_params(py::module& m)
{
    m.def("set_random_state", [](unsigned int seed)
                                { br::Util::r = *br::Util::Rnd::initRand(); 
                                  br::Util::r.set_seed(seed); });
    m.def("rnd_flt", [](){ return br::Util::r.rnd_flt(); });

    py::class_<Brush::Parameters>(m, "Parameters")
        .def(py::init([](){ Brush::Parameters p; return p; }))
        .def_property("verbosity", &Brush::Parameters::get_verbosity, &Brush::Parameters::set_verbosity)
        .def_property("pop_size", &Brush::Parameters::get_pop_size, &Brush::Parameters::set_pop_size)
        .def_property("max_gens", &Brush::Parameters::get_max_gens, &Brush::Parameters::set_max_gens)
        .def_property("max_stall", &Brush::Parameters::get_max_stall, &Brush::Parameters::set_max_stall)
        .def_property("max_time", &Brush::Parameters::get_max_time, &Brush::Parameters::set_max_time)
        .def_property("current_gen", &Brush::Parameters::get_current_gen, &Brush::Parameters::set_current_gen)
        .def_property("scorer", &Brush::Parameters::get_scorer, &Brush::Parameters::set_scorer)
        .def_property("random_state", &Brush::Parameters::get_random_state, &Brush::Parameters::set_random_state)
        .def_property("load_population", &Brush::Parameters::get_load_population, &Brush::Parameters::set_load_population)
        .def_property("save_population", &Brush::Parameters::get_save_population, &Brush::Parameters::set_save_population)
        .def_property("logfile", &Brush::Parameters::get_logfile, &Brush::Parameters::set_logfile)
        .def_property("num_islands", &Brush::Parameters::get_num_islands, &Brush::Parameters::set_num_islands)
        .def_property("constants_simplification", &Brush::Parameters::get_constants_simplification, &Brush::Parameters::set_constants_simplification)
        .def_property("inexact_simplification", &Brush::Parameters::get_inexact_simplification, &Brush::Parameters::set_inexact_simplification)
        .def_property("use_arch", &Brush::Parameters::get_use_arch, &Brush::Parameters::set_use_arch)
        .def_property("val_from_arch", &Brush::Parameters::get_val_from_arch, &Brush::Parameters::set_val_from_arch)
        .def("set_n_classes", &Brush::Parameters::set_n_classes)
        .def("set_class_weights", &Brush::Parameters::set_class_weights)
        .def("set_sample_weights", &Brush::Parameters::set_sample_weights)
        .def_property_readonly("n_classes", &br::Parameters::get_n_classes)
        .def_property_readonly("class_weights", &br::Parameters::get_class_weights)
        .def_property_readonly("sample_weights", &br::Parameters::get_sample_weights)
        .def_property("n_jobs", &Brush::Parameters::get_n_jobs, &Brush::Parameters::set_n_jobs)
        .def_property("weights_init", &Brush::Parameters::get_weights_init, &Brush::Parameters::set_weights_init)
        .def_property("classification", &Brush::Parameters::get_classification, &Brush::Parameters::set_classification)
        .def_property("shuffle_split", &Brush::Parameters::get_shuffle_split, &Brush::Parameters::set_shuffle_split)
        .def_property("validation_size", &Brush::Parameters::get_validation_size, &Brush::Parameters::set_validation_size)
        .def_property("feature_names", &Brush::Parameters::get_feature_names, &Brush::Parameters::set_feature_names)
        .def_property("batch_size", &Brush::Parameters::get_batch_size, &Brush::Parameters::set_batch_size)
        .def_property("max_depth", &Brush::Parameters::get_max_depth, &Brush::Parameters::set_max_depth)
        .def_property("max_size", &Brush::Parameters::get_max_size, &Brush::Parameters::set_max_size)
        .def_property("objectives", &Brush::Parameters::get_objectives, &Brush::Parameters::set_objectives)
        .def_property("sel", &Brush::Parameters::get_sel, &Brush::Parameters::set_sel)
        .def_property("surv", &Brush::Parameters::get_surv, &Brush::Parameters::set_surv)
        .def_property("cx_prob", &Brush::Parameters::get_cx_prob, &Brush::Parameters::set_cx_prob)
        .def_property("mig_prob", &Brush::Parameters::get_mig_prob, &Brush::Parameters::set_mig_prob)
        .def_property("bandit", &Brush::Parameters::get_bandit, &Brush::Parameters::set_bandit)
        .def_property("functions", &Brush::Parameters::get_functions, &Brush::Parameters::set_functions)
        .def_property("mutation_probs", &Brush::Parameters::get_mutation_probs, &Brush::Parameters::set_mutation_probs)
        .def(py::pickle(
          [](const Brush::Parameters &p) { // __getstate__
              /* Return a tuple that fully encodes the state of the object */
              // return py::make_tuple(p.value(), p.extra());
              nl::json j = p;
              return j;
          },
          [](nl::json j) { // __setstate__
              Brush::Parameters p = j;
              return p;
          })
        )
        ;    
}