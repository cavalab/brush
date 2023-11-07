#include "module.h"
#include "../cbrush.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_cbrush(py::module& m)
{
    py::class_<br::CBrush>(m, "CBrush")
        .def(py::init([]()
                      { br::CBrush est; return est; }))
        ;
}