#include "module.h"
#include "../eval/evaluation.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template <br::ProgramType T>
void bind_evaluator(py::module& m, string name)
{
    using Class = br::Eval::Evaluation<T>;
    // TODO: will this part of c++ be exposed?
    py::class_<Class> eval(m, name.data() ); 
    eval.def(py::init<>())    
        // Keep the original four-argument Python API. The optional dataset is
        // used internally by batched Engine runs to score a fixed validation
        // partition and is not required by the DEAP interface.
        .def("assign_fit", &Class::assign_fit,
             py::arg("individual"), py::arg("data"), py::arg("params"),
             py::arg("validation") = false,
             py::arg("validation_data") = nullptr)
        .def_property("scorer", &Class::get_scorer, &Class::set_scorer)
        ;
}
