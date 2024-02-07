#include "module.h"
#include "../variation.h"
#include "../variation.cpp" // TODO: figure out why im having symbol errors (if i dont include the cpp here as well)
#include "../population.h"

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_variation(py::module& m, string name)
{
    using Class = br::Var::Variation<PT>;

    // TODO: make variation a non-templated class
    py::class_<Class> vary(m, name.data() );

    vary.def(py::init<>([](br::Parameters& p, br::SearchSpace& ss){
             Class variation(p, ss);
             return variation; }))
        .def("mutate", &Class::mutate, py::return_value_policy::automatic)
        .def("cross", &Class::cross, py::return_value_policy::automatic)
        ;
}