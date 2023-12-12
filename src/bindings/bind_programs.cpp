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
     // fitness is used to prototype with deap API. TODO: replace deapIndividual with brush individual (once it gets implemented)
     py::class_<br::Fitness>(m, "Fitness", py::dynamic_attr())
          .def(py::init<>())
          .def_readwrite("values", &br::Fitness::values)
          .def_readwrite("valid", &br::Fitness::valid)
          ;

     bind_program<br::RegressorProgram>(m, "Regressor");
     bind_program<br::ClassifierProgram>(m, "Classifier");
     bind_program<br::MulticlassClassifierProgram>(m, "MultiClassifier");
     bind_program<br::RepresenterProgram>(m, "Representer");
}