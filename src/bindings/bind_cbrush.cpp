#include "module.h"
#include "../cbrush.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using namespace Brush;

void bind_cbrush(py::module& m)
{
    py::class_<CBrush>(m, "CBrush", py::dynamic_attr())
        .def(py::init([]()
                      { CBrush est; return est; }))
        ;
}