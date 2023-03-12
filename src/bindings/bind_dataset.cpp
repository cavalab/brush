#include "module.h"
#include "../data/data.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_dataset(py::module & m)
{
    py::class_<br::Data::Dataset>(m, "Dataset")
        .def(py::init<Ref<const ArrayXXf> &, Ref<const ArrayXf> &>())
        .def_readwrite("y", &br::Data::Dataset::y)
        .def("get_n_samples", &br::Data::Dataset::get_n_samples)
        .def("get_n_features", &br::Data::Dataset::get_n_features)
    //     .def_readwrite("features", &br::Data::Dataset::features)
        ;

}