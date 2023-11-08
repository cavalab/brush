#include "module.h"
#include "../cbrush.h"
#include "../types.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

// TODO: copy bind_programs.h to make the cbrush
void bind_cbrush(py::module& m)
{
    py::class_<br::CBrush<br::PT::Regressor>>(m, "BrushRegressor")
        .def(py::init([]()
                      { br::CBrush<br::PT::Regressor> est; return est; }))
        ;
}