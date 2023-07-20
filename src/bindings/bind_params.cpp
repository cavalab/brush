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

    m.def("set_params", &br::set_params);
    m.def("get_params", &br::get_params);
    m.def("set_random_state", [](unsigned int seed)
                                { br::Util::r = *br::Util::Rnd::initRand(); 
                                  br::Util::r.set_seed(seed); });
}