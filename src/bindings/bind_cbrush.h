#include "module.h"
#include "../cbrush.h"

using Reg = Brush::RegressorEstimator;
using Cls = Brush::ClassifierEstimator;
using Rep = Brush::RepresenterEstimator;
using MCls = Brush::MulticlassClassifierEstimator;

namespace nl = nlohmann;
namespace br = Brush;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

template<typename T>
void bind_estimator(py::module& m, string name)
{
    // using RetType = std::conditional_t<
    //         std::is_same_v<T,Reg>, ArrayXf, 
    //         std::conditional_t<std::is_same_v<T,Cls>, ArrayXb, 
    //         std::conditional_t<std::is_same_v<T,MCls>, ArrayXi, ArrayXXf>>>;

    // py::class_<T> ind(m, name.data() ); 
    // ind.def(py::init<>())
    //    ;
    // if constexpr (std::is_same_v<T,Cls>)
    // {

    // }
}