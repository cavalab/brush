#include "module.h"
#include "bind_individuals.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;


void bind_individuals(py::module& m)
{
     // fitness is used to prototype with deap API. TODO: replace deapIndividual with brush individual (once it gets implemented)
     py::class_<br::Fitness>(m, "Fitness", py::dynamic_attr())
          .def(py::init<>())
        .def(py::init<const std::vector<float>&>(), "Constructor with weights")
        .def_property("values", &br::Fitness::get_values, &br::Fitness::set_values)
          .def_property_readonly("weights", &br::Fitness::get_weights)
          .def_property_readonly("wvalues", &br::Fitness::get_wvalues)
        .def("dominates", &Fitness::dominates)
        .def("clearValues", &Fitness::clearValues, "Clear the weighted values vector")
        .def_property("rank", &Fitness::get_rank, &Fitness::set_rank)
        .def_property("loss", &Fitness::get_loss, &Fitness::set_loss)
        .def_property("loss_v", &Fitness::get_loss_v, &Fitness::set_loss_v)
        .def_property("crowding_dist", &Fitness::get_crowding_dist, &Fitness::set_crowding_dist)
    
        .def("valid", &Fitness::valid, "Check if the fitness is valid")
        .def("__hash__", &Fitness::hash, py::is_operator())
        .def("__eq__", &Fitness::operator==, py::is_operator())
        .def("__ne__", &Fitness::operator!=, py::is_operator())
        .def("__lt__", &Fitness::operator<, py::is_operator())
        .def("__gt__", &Fitness::operator>, py::is_operator())
        .def("__le__", &Fitness::operator<=, py::is_operator())
        .def("__ge__", &Fitness::operator>=, py::is_operator())
     //    .def("__str__", &Fitness::toString, "String representation of the Fitness object")
     //    .def("__repr__", &Fitness::repr, "Representation for debugging the Fitness object")
          .def(py::pickle(
               [](const br::Fitness &f) { // __getstate__
                    /* Return a tuple that fully encodes the state of the object */
                    // return py::make_tuple(p.value(), p.extra());
                    nl::json j = f;
                    return j;
               },
               [](nl::json j) { // __setstate__
                    br::Fitness f = j;
                    return f;
               }
               )
          )
          ;

     bind_individual<br::ProgramType::Regressor>(m, "RegressorIndividual");
     bind_individual<br::ProgramType::BinaryClassifier>(m, "ClassifierIndividual");
     bind_individual<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierIndividual");
     // bind_individual<br::ProgramType::Representer>(m, "RepresenterIndividual");
}