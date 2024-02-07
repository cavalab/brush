#include "module.h"
#include "../population.h"
#include "../population.cpp" // TODO: figure out why im having symbol errors (if i dont include the cpp here as well)

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_population(py::module& m, string name)
{
    using Class = br::Pop::Population<PT>;

    // TODO: make population a non-templated class
    py::class_<Class> pop(m, name.data() );

    // TODO: access individuals by index
    pop.def(py::init<>())
       ;
}