#include "module.h"
#include "bind_programs.h"
namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

// using Reg = br::Program<ArrayXf>;
// using Cls = br::Program<ArrayXb>;
// using Rep = br::Program<ArrayXXf>;
// using MCls = br::Program<ArrayXi>;

void bind_programs(py::module& m)
{
     bind_program<br::RegressorProgram>(m, "Regressor");
     bind_program<br::ClassifierProgram>(m, "Classifier");
     bind_program<br::MulticlassClassifierProgram>(m, "MultiClassifier");
     bind_program<br::RepresenterProgram>(m, "Representer");
}