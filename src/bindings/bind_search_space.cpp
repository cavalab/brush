#include "module.h"
#include "../vary/search_space.h"
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
            py::arg("weights_init") = true )
        .def(py::init<const Dataset&, const unordered_map<string,float>&, 
            bool>(),
            py::arg("data"),
            py::arg("user_ops"),
            py::arg("weights_init") = true )
        .def("__repr__", &br::SearchSpace::repr, "Representation for debugging the SearchSpace object")
        .def("make_regressor", &br::SearchSpace::make_regressor,
             py::arg("max_d") = 0,
             py::arg("max_size") = 0,
             py::arg("params") = Brush::Parameters() )
        .def("make_classifier", &br::SearchSpace::make_classifier,
             py::arg("max_d") = 0,
             py::arg("max_size") = 0,
             py::arg("params") = Brush::Parameters() )
        .def("make_multiclass_classifier",
             &br::SearchSpace::make_multiclass_classifier,
             py::arg("max_d") = 0,
             py::arg("max_size") = 0,
             py::arg("params") = Brush::Parameters() )
        .def("make_representer", &br::SearchSpace::make_representer,
             py::arg("max_d") = 0,
             py::arg("max_size") = 0,
             py::arg("params") = Brush::Parameters() )
        .def("print", 
            &br::SearchSpace::print, 
            stream_redirect()
        )
    ;
}