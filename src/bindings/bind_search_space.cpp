#include "module.h"
#include "../search_space.h"
#include "../program/program.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

void bind_search_space(py::module &m)
{
    // Notice: We change the interface for SearchSpace a little bit by
    // constructing it with a Dataset object, rather than initializing it as an
    // empty struct and then calling init() with the Dataset object.
    py::class_<br::SearchSpace>(m, "SearchSpace")
        .def(py::init([](br::Data::Dataset data, bool weights_init=true){
                SearchSpace SS;
                SS.init(data, {}, weights_init);
                return SS;
            }),
            py::arg("data"),
            py::arg("weights_init") = true
        )
        .def(py::init<const Dataset&, const unordered_map<string,float>&, 
            bool>(),
            py::arg("data"),
            py::arg("user_ops"),
            py::arg("weights_init") = true
        )
        .def("make_regressor", &br::SearchSpace::make_regressor)
        .def("make_classifier", &br::SearchSpace::make_classifier)
        .def("make_multiclass_classifier", &br::SearchSpace::make_multiclass_classifier)
        .def("make_representer", &br::SearchSpace::make_representer)
        .def("print", 
            &br::SearchSpace::print, 
            stream_redirect()
        )
    ;
}