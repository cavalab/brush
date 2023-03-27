#include "module.h"
#include "bind_programs.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using Reg = br::Program<ArrayXf>;
using Cls = br::Program<ArrayXb>;
using Rep = br::Program<ArrayXXf>;
using MCls = br::Program<ArrayXi>;

void bind_programs(py::module& m)
{
     py::class_<br::Fitness>(m, "Fitness", py::dynamic_attr())
          .def(py::init<>())
          .def_readwrite("values", &br::Fitness::values)
          .def_readwrite("valid", &br::Fitness::valid)
          // .def("__del__", [](const C&) -> void { 
          //      std::cout << "deleting C" << std::endl;
          //      }
          // )
          ;

     bind_program<Reg>(m, "Regressor");
     bind_program<Cls>(m, "Classifier");
     bind_program<MCls>(m, "MultiClassifier");
     bind_program<Rep>(m, "Representer");

     // using Reg = br::Program<ArrayXf>;
     // py::class_<Reg>(m2, "Regressor")
     //      .def(py::init<>())
     //      // .def("fitness",&Reg::fitness,"fitness struct")
     //      // .def_readwrite("fitness.values",&Reg::fitness::values)
     //      // .def_readwrite("fitness.valid",&Reg::fitness::valid)
     //      .def("fit",
     //           static_cast<Reg &(Reg::*)(const Dataset &d)>(&Reg::fit),
     //           "fit from Dataset object")
     //      .def("fit",
     //           static_cast<Reg &(Reg::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&Reg::fit),
     //           "fit from X,y data")
     //      .def("predict",
     //           static_cast<ArrayXf (Reg::*)(const Dataset &d)>(&Reg::predict),
     //           "predict from Dataset object")
     //      .def("predict",
     //           static_cast<ArrayXf (Reg::*)(const Ref<const ArrayXXf> &X)>(&Reg::predict),
     //           "fit from X,y data")
     //      .def("get_model",
     //           &Reg::get_model,
     //           py::arg("type") = "compact",
     //           py::arg("pretty") = false)
     //      .def("size", &Reg::size)
     //      ;


     // using Cls = br::Program<ArrayXb>;
     // py::class_<Cls>(m2, "Classifier", py::dynamic_attr() )
     //      .def(py::init<>())
     //      .def_readwrite("fitness", &Cls::fitness)
     //      // .def_readwrite("fitness.values", &Cls::fitness_values)
     //      // .def_readwrite("fitness.valid", &Cls::fitness_valid)
     //      .def("fit",
     //           static_cast<Cls &(Cls::*)(const Dataset &d)>(&Cls::fit),
     //           "fit from Dataset object")
     //      .def("fit",
     //           static_cast<Cls &(Cls::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&Cls::fit),
     //           "fit from X,y data")
     //      .def("predict",
     //           static_cast<ArrayXb (Cls::*)(const Dataset &d)>(&Cls::predict),
     //           "predict from Dataset object")
     //      .def("predict",
     //           static_cast<ArrayXb (Cls::*)(const Ref<const ArrayXXf> &X)>(&Cls::predict),
     //           "fit from X,y data")
     //      .def("predict_proba",
     //           static_cast<ArrayXf (Cls::*)(const Dataset &d)>(&Cls::predict_proba),
     //           "predict from Dataset object")
     //      .def("predict_proba",
     //           static_cast<ArrayXf (Cls::*)(const Ref<const ArrayXXf> &X)>(&Cls::predict_proba),
     //           "fit from X,y data")
     //      .def("get_model",
     //           &Cls::get_model,
     //           py::arg("type") = "compact",
     //           py::arg("pretty") = false)
     //      .def("size", &Cls::size)
     //      // .def("cross", static_cast<Cls &(Cls::*)(const Cls&)>(&Cls::cross))
     //      .def("cross", &Cls::cross)
     //      .def("mutate", &Cls::mutate) // static_cast<Cls &(Cls::*)()>(&Cls::mutate))
     //      .def("set_search_space", &Cls::set_search_space)
     //      .def(py::pickle(
     //           [](const Cls &p) { // __getstate__
     //                /* Return a tuple that fully encodes the state of the object */
     //                // return py::make_tuple(p.value(), p.extra());
     //                nl::json j = p;
     //                return j;
     //           },
     //           [](nl::json j) { // __setstate__
     //                Cls p = j;

     //                return p;
     //           }
     //           )
     //      )
     //      ;

     // using MCls = br::Program<ArrayXi>;
     // py::class_<MCls>(m2, "MulticlassClassifer")
     //      .def(py::init<>())
     //      .def("fit",
     //           static_cast<MCls &(MCls::*)(const Dataset &d)>(&MCls::fit),
     //           "fit from Dataset object")
     //      .def("fit",
     //           static_cast<MCls &(MCls::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&MCls::fit),
     //           "fit from X,y data")
     //      .def("predict",
     //           static_cast<ArrayXi (MCls::*)(const Dataset &d)>(&MCls::predict),
     //           "predict from Dataset object")
     //      .def("predict",
     //           static_cast<ArrayXi (MCls::*)(const Ref<const ArrayXXf> &X)>(&MCls::predict),
     //           "fit from X,y data")
     //      .def("get_model",
     //           &MCls::get_model,
     //           py::arg("type") = "compact",
     //           py::arg("pretty") = false)
     //      .def("size", &MCls::size)
     //      ;

     // using Rep = br::Program<ArrayXXf>;
     // py::class_<Rep>(m2, "Representer")
     //      .def(py::init<>())
     //      .def("fit",
     //           static_cast<Rep &(Rep::*)(const Dataset &d)>(&Rep::fit),
     //           "fit from Dataset object")
     //      .def("fit",
     //           static_cast<Rep &(Rep::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&Rep::fit),
     //           "fit from X,y data")
     //      .def("predict",
     //           static_cast<ArrayXXf (Rep::*)(const Dataset &d)>(&Rep::predict),
     //           "predict from Dataset object")
     //      .def("predict",
     //           static_cast<ArrayXXf (Rep::*)(const Ref<const ArrayXXf> &X)>(&Rep::predict),
     //           "fit from X,y data")
     //      .def("get_model",
     //           &Rep::get_model,
     //           py::arg("type") = "compact",
     //           py::arg("pretty") = false)
     //      .def("size", &Rep::size)
     //      .def(py::pickle(
     //           [](const Rep &p) { // __getstate__
     //                /* Return a tuple that fully encodes the state of the object */
     //                // return py::make_tuple(p.value(), p.extra());
     //                nl::json j = p;
     //                return j;
     //           },
     //           [](nl::json j) { // __setstate__
     //                Rep p = j;

     //                return p;
     //           }
     //           )
     //      )
     //      ;
}