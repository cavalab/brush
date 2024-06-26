#include "module.h"
#include "../program/program.h"

using Reg = Brush::RegressorProgram;
using Cls = Brush::ClassifierProgram;
using Rep = Brush::RepresenterProgram;
using MCls = Brush::MulticlassClassifierProgram;

namespace nl = nlohmann;
namespace br = Brush;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template<typename T>
void bind_program(py::module& m, string name)
{
    using RetType = std::conditional_t<
            std::is_same_v<T,Reg>, ArrayXf, 
            std::conditional_t<std::is_same_v<T,Cls>, ArrayXb, 
            std::conditional_t<std::is_same_v<T,MCls>, ArrayXi, ArrayXXf>>>;

    py::class_<T> prog(m, name.data() ); 
    prog.def(py::init<>())
        .def(py::init(
            [](const json& j){ T p = j; return p; })
        )
        .def("fit",
            static_cast<T &(T::*)(const Dataset &d)>(&T::fit),
            "fit from Dataset object")
        .def("fit",
            static_cast<T &(T::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&T::fit),
            "fit from X,y data")
        .def("predict",
            static_cast<RetType (T::*)(const Dataset &d)>(&T::predict),
            "predict from Dataset object")
        .def("predict",
            static_cast<RetType (T::*)(const Ref<const ArrayXXf> &X)>(&T::predict),
            "predict from X data")
        .def("get_model",
            &T::get_model,
            py::arg("type") = "compact",
            py::arg("pretty") = false,
            stream_redirect()
            )
        .def("get_dot_model", &T::get_dot_model, py::arg("extras")="")
        .def("get_weights", &T::get_weights)
        .def("size", &T::size, py::arg("include_weight")=true)
        .def("complexity", &T::complexity)
        .def("linear_complexity", &T::linear_complexity)
        .def("depth", &T::depth)
        // .def("cross", &T::cross, py::return_value_policy::automatic,
        //      "Performs one attempt to stochastically swap subtrees between two programs and generate a child")
        // .def("mutate", &T::mutate, py::return_value_policy::automatic,
        //      "Performs one attempt to stochastically mutate the program and generate a child")
        .def("set_search_space", &T::set_search_space)
        //.def("copy", &T::copy<>, py::return_value_policy::copy)
        .def("copy", [](const T& self){ T clone(self); return clone; })
        .def(py::pickle(
            [](const T &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                // return py::make_tuple(p.value(), p.extra());
                nl::json j = p;
                return j;
            },
            [](nl::json j) { // __setstate__
                T p = j;

                return p;
            }
            )
        )
        ;
    if constexpr (std::is_same_v<T,Cls>)
    {
        prog.def("predict_proba",
                static_cast<ArrayXf (T::*)(const Dataset &d)>(&T::predict_proba),
                "predict from Dataset object")
           .def("predict_proba",
                static_cast<ArrayXf (T::*)(const Ref<const ArrayXXf> &X)>(&T::predict_proba),
                "predict from X data")
            ;
    }

}