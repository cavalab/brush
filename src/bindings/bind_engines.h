#include "module.h"
#include "../engine.h"
#include "../engine.cpp"

// TODO: figure out why do I need to include the whole thing (otherwise it gives me symbol errors)
#include "../selection/selection.h"
#include "../selection/selection.cpp"
#include "../selection/selection_operator.h"
#include "../selection/selection_operator.cpp"
#include "../selection/nsga2.h"
#include "../selection/nsga2.cpp"
#include "../selection/lexicase.h"
#include "../selection/lexicase.cpp"

#include "../eval/evaluation.h"
#include "../eval/evaluation.cpp"

#include "../population.cpp"
#include "../population.h"

using Reg = Brush::RegressorEngine;
using Cls = Brush::ClassifierEngine;
using Rep = Brush::RepresenterEngine;
using MCls = Brush::MulticlassClassifierEngine;

namespace nl = nlohmann;
namespace br = Brush;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template<typename T>
void bind_engine(py::module& m, string name)
{    
    using RetType = std::conditional_t<
            std::is_same_v<T,Reg>, ArrayXf, 
            std::conditional_t<std::is_same_v<T,Cls>, ArrayXb, 
            std::conditional_t<std::is_same_v<T,MCls>, ArrayXi, ArrayXXf>>>;

    py::class_<T> engine(m, name.data() ); 
    engine.def(py::init<>())
             .def(py::init([](br::Parameters& p){ T e(p);
                                                  return e; })
             )
             .def_property("params", &T::get_params, &T::set_params)
             .def_property_readonly("is_fitted", &T::get_is_fitted)
             .def_property_readonly("best_ind", &T::get_best_ind)
             .def("run", &T::run, "run from brush dataset")
             ;

    // specialization for subclasses
    if constexpr (std::is_same_v<T,Cls>)
    {

    }
}