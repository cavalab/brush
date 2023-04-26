#include "module.h"
#include "../data/data.h"
#include "../types.h"
#include "../data/io.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_dataset(py::module & m)
{
    py::class_<br::Data::Dataset>(m, "Dataset")
        // construct from X 
        .def(py::init<Ref<const ArrayXXf> &>())
        // construct from X, feature names 
        .def(py::init<
                const Ref<const ArrayXXf>&, 
                const vector<string>&
            >()
        )
        // construct from X,y arrays
        .def(py::init<Ref<const ArrayXXf> &, Ref<const ArrayXf> &>())
        // construct from X, y, feature names 
        .def(py::init<
                const Ref<const ArrayXXf>&, 
                const Ref<const ArrayXf>&,
                const vector<string>&
            >()
        )
        .def_readwrite("y", &br::Data::Dataset::y)
    //     .def_readwrite("features", &br::Data::Dataset::features)
        .def("get_n_samples", &br::Data::Dataset::get_n_samples)
        .def("get_n_features", &br::Data::Dataset::get_n_features)
        .def("print", &br::Data::Dataset::print)
        .def("get_batch", &br::Data::Dataset::get_batch)
        .def("get_X", &br::Data::Dataset::get_X)
        ;

    m.def("read_csv", &br::Data::read_csv, py::arg("path"), py::arg("target"), py::arg("sep")=',');
}