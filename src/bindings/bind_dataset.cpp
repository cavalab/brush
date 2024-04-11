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
        // construct from X, feature names (and optional validation and batch sizes) with constructor 3.
        .def(py::init([](const Ref<const ArrayXXf>& X, 
                         const vector<string>& feature_names=vector<string>(),
                         const bool c=false,
                         const float validation_size=0.0,
                         const float batch_size=1.0){
                return br::Data::Dataset(
                    X, feature_names, c, validation_size, batch_size);
            }), 
            py::arg("X"),
            py::arg("feature_names") = vector<string>(),
            py::arg("c") = false,
            py::arg("validation_size") = 0.0,
            py::arg("batch_size") = 1.0
        )
        // construct from X, y, feature names (and optional validation and batch sizes) with constructor 2.
        .def(py::init([](const Ref<const ArrayXXf>& X, 
                         const Ref<const ArrayXf>& y,
                         const vector<string>& feature_names=vector<string>(),
                         const bool c=false,
                         const float validation_size=0.0,
                         const float batch_size=1.0){
                return br::Data::Dataset(
                    X, y, feature_names, {}, c, validation_size, batch_size);
            }), 
            py::arg("X"),
            py::arg("y"),
            py::arg("feature_names") = vector<string>(),
            py::arg("c") = false,
            py::arg("validation_size") = 0.0,
            py::arg("batch_size") = 1.0
        )
        // construct from X, feature names, but copying the feature types from a
        // reference dataset with constructor 4. Useful for predicting (specially
        // because the user can provide a single element matrix, or an array with
        // no feature names).
        .def(py::init([](const Ref<const ArrayXXf>& X, 
                         const br::Data::Dataset& ref_dataset,
                         const vector<string>& feature_names,
                         const bool c=false){
                return br::Data::Dataset(X, ref_dataset, feature_names, c);
            }), 
            py::arg("X"),
            py::arg("ref_dataset"),
            py::arg("feature_names"),
            py::arg("c") = false
        )
        
        .def_readwrite("y", &br::Data::Dataset::y)
        // .def_readwrite("features", &br::Data::Dataset::features)
        .def("get_n_samples", &br::Data::Dataset::get_n_samples)
        .def("get_n_features", &br::Data::Dataset::get_n_features)
        .def("print", &br::Data::Dataset::print)
        .def("get_batch", &br::Data::Dataset::get_batch)
        .def("get_training_data", &br::Data::Dataset::get_training_data)
        .def("get_validation_data", &br::Data::Dataset::get_validation_data)
        .def("get_batch_size", &br::Data::Dataset::get_batch_size)
        .def("set_batch_size", &br::Data::Dataset::set_batch_size)
        .def("split", &br::Data::Dataset::split)
        .def("get_X", &br::Data::Dataset::get_X)
        ;

    m.def("read_csv", &br::Data::read_csv, py::arg("path"), py::arg("target"), py::arg("sep")=',');
}