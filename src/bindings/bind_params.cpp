#include "module.h"
#include "../params.h"

void bind_params(py::module& m)
{
    // py::object params = Brush::PARAMS;
    // m.attr("PARAMS") = params;

    //  py::class_<br::Params>(m, "Params", py::dynamic_attr())
    //       .def(py::init<>())

    m.def("set_params", &Brush::set_params);
    m.def("get_params", &Brush::get_params);
}