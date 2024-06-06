#include "module.h"

#include "../ind/individual.h"

namespace nl = nlohmann;
namespace br = Brush;

using Reg = Brush::RegressorIndividual;
using Cls = Brush::ClassifierIndividual;
using MCls = Brush::MulticlassClassifierIndividual;
using Rep = Brush::RepresenterIndividual;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template <br::ProgramType PT>
void bind_individual(py::module& m, string name)
{
    using Class = br::Pop::Individual<PT>;
    
    using RetType = std::conditional_t<
            std::is_same_v<Class,Reg>, ArrayXf, 
            std::conditional_t<std::is_same_v<Class,Cls>, ArrayXb, 
            std::conditional_t<std::is_same_v<Class,MCls>, ArrayXi, ArrayXXf>>>;

    py::class_<Class> ind(m, name.data() ); 
    ind.def(py::init<>())
       .def(py::init([](br::Program<PT>& prg){ Class i(prg);
                                                    return i; })
       )
       .def(py::init([](const json& j){ br::Program<PT> prg = j;
                                        Class i(prg);
                                        return i; })
       )
       .def("init", &Class::init)
       .def_property("objectives", &Class::get_objectives, &Class::set_objectives)
       .def_property_readonly("program", &Class::get_program)
       .def_property_readonly("fitness", &Class::get_fitness)
       .def("get_model", &Class::get_model, 
            py::arg("fmt") = "compact",
            py::arg("pretty") = false)
       .def("get_dot_model", &Class::get_dot_model,
            py::arg("extras") = "")
       .def("fit",
            static_cast<Class &(Class::*)(const Dataset &d)>(&Class::fit),
            "fit from Dataset object")
        .def("fit",
            static_cast<Class &(Class::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&Class::fit),
            "fit from X,y data")
        .def("predict",
            static_cast<RetType (Class::*)(const Dataset &d)>(&Class::predict),
            "predict from Dataset object")
        .def("predict",
            static_cast<RetType (Class::*)(const Ref<const ArrayXXf> &X)>(&Class::predict),
            "predict from X data")
       .def(py::pickle(
            [](const Class &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                // return py::make_tuple(p.value(), p.extra());
                nl::json j = p;
                return j;
            },
            [](nl::json j) { // __setstate__
                Class p = j;
                return p;
            }
            )
       )
       ;

    if constexpr (std::is_same_v<Class,Cls>)
    {
        ind.def("predict_proba",
                static_cast<ArrayXf (Class::*)(const Dataset &d)>(&Class::predict_proba),
                "predict from Dataset object")
           .def("predict_proba",
                static_cast<ArrayXf (Class::*)(const Ref<const ArrayXXf> &X)>(&Class::predict_proba),
                "predict from X data")
            ;
    }

}