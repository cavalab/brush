#include "module.h"
#include "../selection/selection.h"
#include "../selection/selection.cpp" // TODO: figure out why im having symbol errors (if i dont include the cpp here as well)

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_selection(py::module& m, string name)
{
    using Class = br::Sel::Selection<PT>;

    // TODO: make selection a non-templated class
    py::class_<Class> sel(m, name.data() );

    sel.def(py::init<>())
       .def(py::init(
           [](string type, bool survival){ Class s(type, survival); return s; })
       )
       ;
}