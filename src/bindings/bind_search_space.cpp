#include "module.h"
#include "../search_space.h"
#include "../program/program.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_search_space(py::module &m)
{
    // Notice: We change the interface for SearchSpace a little bit by
    // constructing it with a Dataset object, rather than initializing it as an
    // empty struct and then calling init() with the Dataset object.
    py::class_<br::SearchSpace>(m, "SearchSpace")
        .def(py::init([](br::Data::Dataset data)
                    {
                SearchSpace SS;
                SS.init(data);
                return SS; }))
        .def("make_regressor", &br::SearchSpace::make_regressor)
        .def("make_classifier", &br::SearchSpace::make_classifier)
        .def("make_multiclass_classifier", &br::SearchSpace::make_multiclass_classifier)
        .def("make_representer", &br::SearchSpace::make_representer)
        .def("print", &br::SearchSpace::print)
    ;
}