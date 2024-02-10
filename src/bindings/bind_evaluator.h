#include "module.h"
#include "../eval/evaluation.h"
#include "../eval/evaluation.cpp"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template <br::ProgramType T>
void bind_evaluator(py::module& m, string name)
{
    using Class = br::Eval::Evaluation<T>;
    
    py::class_<Class> eval(m, name.data() ); 
    eval.def(py::init<>())    
        .def("assign_fit", &Class::assign_fit)
        .def_property("scorer", &Class::get_scorer, &Class::set_scorer)
        ;
}