#include "module.h"
#include "../engine.h"
#include "../engine.cpp"

// TODO: figure out why do I need to include the whole thing (otherwise it gives me symbol errors)
#include "../bandit/bandit.h"
#include "../bandit/bandit.cpp"
#include "../bandit/bandit_operator.h"
#include "../bandit/bandit_operator.cpp"
#include "../bandit/dummy.h"
#include "../bandit/dummy.cpp"
#include "../bandit/thompson.h"
#include "../bandit/thompson.cpp"

#include "../ind/individual.h"
#include "../ind/individual.cpp"
#include "../vary/variation.h"
#include "../vary/variation.cpp"

#include "../eval/evaluation.h"
#include "../eval/evaluation.cpp"

#include "../pop/population.cpp"
#include "../pop/population.h"

#include "../selection/selection.h"
#include "../selection/selection.cpp"
#include "../selection/selection_operator.h"
#include "../selection/selection_operator.cpp"
#include "../selection/nsga2.h"
#include "../selection/nsga2.cpp"
#include "../selection/lexicase.h"
#include "../selection/lexicase.cpp"

#include "../pop/archive.cpp"
#include "../pop/archive.h"

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
             .def(py::init([](br::Parameters& p, br::SearchSpace& s){
                T e(p, s); return e; })
             )
             .def_property("params", &T::get_params, &T::set_params)
             .def_property("search_space", &T::get_search_space, &T::set_search_space)
             .def_property_readonly("is_fitted", &T::get_is_fitted)
             .def_property_readonly("best_ind", &T::get_best_ind)
             .def("fit",
                static_cast<T &(T::*)(Dataset &d)>(&T::fit),
                py::call_guard<py::gil_scoped_release>(), 
                "fit from Dataset object")
            .def("fit",
                static_cast<T &(T::*)(const Ref<const ArrayXXf> &X, const Ref<const ArrayXf> &y)>(&T::fit),
                py::call_guard<py::gil_scoped_release>(), 
                "fit from X,y data")
            .def("predict",
                static_cast<RetType (T::*)(const Dataset &d)>(&T::predict),
                "predict from Dataset object")
            .def("predict",
                static_cast<RetType (T::*)(const Ref<const ArrayXXf> &X)>(&T::predict),
                "predict from X data")
            .def("predict_archive",
                static_cast<RetType (T::*)(int id, const Dataset &d)>(&T::predict_archive),
                "predict from individual in archive")
            .def("predict_archive",
                static_cast<RetType (T::*)(int id, const Ref<const ArrayXXf> &X)>(&T::predict_archive),
                "predict from individual in archive")
            .def("get_archive", &T::get_archive, py::arg("front") = false)
            .def("get_population", &T::get_population)
            .def("set_population", &T::set_population)
            .def("lock_nodes",
                &T::lock_nodes,
                py::arg("end_depth") = 0,
                py::arg("skip_leaves") = true,
                stream_redirect()
            )
            .def("unlock_nodes",
                &T::unlock_nodes,
                py::arg("start_depth") = 0,
                stream_redirect()
            )
            .def(py::pickle(
                [](const T &p) { // __getstate__
                    /* Return a tuple that fully encodes the state of the object */
                    // return py::make_tuple(p.value(), p.extra());
                    nl::json j = p;
                    return j;
                },
                [](nl::json j) { // __setstate__
                    T p = j;
                    // TODO: do I need to get the data and ss reference, then call init for this new instance?
                    return p;
                })
             )
             ;

    // specialization for subclasses
    if constexpr (std::is_same_v<T,Cls>)
    {
        engine.def("predict_proba",
                static_cast<ArrayXf (T::*)(const Dataset &d)>(&T::predict_proba),
                "predict from Dataset object")
           .def("predict_proba",
                static_cast<ArrayXf (T::*)(const Ref<const ArrayXXf> &X)>(&T::predict_proba),
                "predict from X data")
            .def("predict_proba_archive",
                static_cast<ArrayXf (T::*)(int id, const Dataset &d)>(&T::predict_proba_archive),
                "predict from individual in archive")
            .def("predict_proba_archive",
                static_cast<ArrayXf (T::*)(int id, const Ref<const ArrayXXf> &X)>(&T::predict_proba_archive),
                "predict from individual in archive")
            
            ;
    }
}