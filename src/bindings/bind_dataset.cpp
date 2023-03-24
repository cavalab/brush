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
        // construct from X array
        .def(py::init<Ref<const ArrayXXf> &>())
        // construct from X,y arrays
        .def(py::init<Ref<const ArrayXXf> &, Ref<const ArrayXf> &>())
        // construct from X, feature names, y 
        .def(py::init<
                const Ref<const ArrayXXf>&, 
                const Ref<const ArrayXf>&,
                const vector<string>&
            >()
        )
        // construct from X, feature names 
        .def(py::init<
                const Ref<const ArrayXXf>&, 
                const vector<string>&
            >()
        )
            // ,
            // py::arg("y_")=ArrayXf(), 
            // py::arg("Z")={},
            // py::arg("vn")={},
            // py::arg("c")=false 
        // )
        // .def(py::init<
        //         std::map<string, br::Data::State>& 
        //     >()
        //     // ,
        //     // py::arg("y_")=ArrayXf(), 
        //     // py::arg("c")=false 
        // )
        // .def(py::init<
        //         std::map<string, br::Data::State>&, 
        //         const Ref<const ArrayXf>&, 
        //         bool
        //     >()
        //     // ,
        //     // py::arg("y_")=ArrayXf(), 
        //     // py::arg("c")=false 
        // )
        // .def(py::init<Ref<const ArrayXXf> &, Ref<const ArrayXf> &>())
        .def_readwrite("y", &br::Data::Dataset::y)
        .def("get_n_samples", &br::Data::Dataset::get_n_samples)
        .def("get_n_features", &br::Data::Dataset::get_n_features)
        .def("print", &br::Data::Dataset::print)
    //     .def_readwrite("features", &br::Data::Dataset::features)
        ;

    m.def("read_csv", &br::Data::read_csv, py::arg("path"), py::arg("target"), py::arg("sep")=',');
}