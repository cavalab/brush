#include "module.h"

#include "../ind/fitness.h"

namespace nl = nlohmann;
namespace br = Brush;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

void bind_fitness(py::module& m)
{
    py::class_<br::Fitness>(m, "Fitness", py::dynamic_attr())
          .def(py::init<>())
        .def(py::init<const std::vector<float>&>(), "Constructor with weights")
        .def_property("values", &br::Fitness::get_values, &br::Fitness::set_values)
        .def_property_readonly("weights", &br::Fitness::get_weights)
        .def_property_readonly("wvalues", &br::Fitness::get_wvalues)
        .def("dominates", &br::Fitness::dominates)
        .def("clearValues", &br::Fitness::clearValues, "Clear the weighted values vector")
        .def_property("rank", &br::Fitness::get_rank, &br::Fitness::set_rank)
        .def_property("crowding_dist", &br::Fitness::get_crowding_dist, &br::Fitness::set_crowding_dist)

        .def_property("loss", &br::Fitness::get_loss, &br::Fitness::set_loss)
        .def_property("loss_v", &br::Fitness::get_loss_v, &br::Fitness::set_loss_v)
        .def_property("size", &br::Fitness::get_size, &br::Fitness::set_size)
        .def_property("complexity", &br::Fitness::get_complexity, &br::Fitness::set_complexity)        
        .def_property("linear_complexity", &br::Fitness::get_linear_complexity, &br::Fitness::set_linear_complexity)        
        .def_property("depth", &br::Fitness::get_depth, &br::Fitness::set_depth)

        .def_property_readonly("prev_loss", &br::Fitness::get_loss)
        .def_property_readonly("prev_loss_v", &br::Fitness::get_loss_v)
        .def_property_readonly("prev_size", &br::Fitness::get_size)
        .def_property_readonly("prev_complexity", &br::Fitness::get_complexity)
        .def_property_readonly("prev_linear_complexity", &br::Fitness::get_linear_complexity)
        .def_property_readonly("prev_depth", &br::Fitness::get_depth)
        
        .def("valid", &br::Fitness::valid, "Check if the fitness is valid")
        .def("__hash__", &br::Fitness::hash, py::is_operator())
        .def("__eq__", &br::Fitness::operator==, py::is_operator())
        .def("__ne__", &br::Fitness::operator!=, py::is_operator())
        .def("__lt__", &br::Fitness::operator<, py::is_operator())
        .def("__gt__", &br::Fitness::operator>, py::is_operator())
        .def("__le__", &br::Fitness::operator<=, py::is_operator())
        .def("__ge__", &br::Fitness::operator>=, py::is_operator())
        .def("__str__", &br::Fitness::toString, "String representation of the Fitness object")
        .def("__repr__", &br::Fitness::repr, "Representation for debugging the Fitness object")
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

}