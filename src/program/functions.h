/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Code below heavily inspired by heal-research/operon
*/
#ifndef OP_H
#define OP_H
#include <cmath>
#include <iterator>
#include <functional>
#include <numeric>
#include <type_traits>

#include "../init.h"
#include "nodetype.h"
#include "../util/utils.h"
#include "../data/data.h"
#include "node.h"
using namespace Brush::Util;

using namespace std;
// using namespace Brush;

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;
using Eigen::Array;
using Eigen::ArrayXi;
using Eigen::Dynamic;

using Brush::Data::TimeSeries;
using Brush::Data::TimeSeriesf;
using Brush::NodeType;

namespace Brush
{
/* Operators
- In order to work with ceres, operators should be templated to take jets.
- by returning auto, operators preserve expression templates in Eigen. that allows them to be evaluated once after the expression is constructed. 
- might need to extend eigen to handle the median case
https://eigen.tuxfamily.org/dox/TopicCustomizing_Plugins.html

- eigen reference of operators: 
https://eigen.tuxfamily.org/dox/group__QuickRefPage.html#arrayonly
*/
    template<Brush::NodeType N>
    struct Function 
    {
        template<typename T1, typename... Tn>
        inline auto operator()(const T1& t1, Tn... tn) { return t1; }
    };
    /* template<Brush::NodeType N = NodeType::Add> */
    template<>
    struct Function<NodeType::Add> 
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { 
            return t1 + t2; 
        }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1 - t2; }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1 * t2 ;}
    };

    template<>
    struct Function<NodeType::Div>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1 / t2 ; }
    };
    

    /* coefficient-wise minimum of two or more arguments. */
    template<>
    struct Function<NodeType::Min>
    {
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(const T1& t1, const T2& t2, Tn... tn) { return fmin(t1, t2, tn ...); }

        template<typename T>
        inline auto operator()(const T& t) { return t.rowwise().minCoeff(); }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.min(); } 
    };

    /* coefficient-wise maximum of two or more arguments. */
    template<>
    struct Function<NodeType::Max>
    {
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(const T1& t1, const T2& t2, Tn... tn) { return fmax(t1, t2, tn ...); }

        template<typename T>
        inline auto operator()(const T& t) { return t.rowwise().maxCoeff(); }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.max(); }
    };

    /* mean */
    template<>
    struct Function<NodeType::Mean>
    {

        template<typename T>
        inline auto operator()(const T& t) { return t.rowwise().mean(); }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.mean(); }
        
    };

    /* median 
    https://stackoverflow.com/questions/62696455/is-there-a-way-to-find-the-median-value-of-coefficients-of-an-eigen-matrix#62698308
    template<>
    */
    template<>
    struct Function<NodeType::Median>
    {
        template<typename Derived>
        typename Derived::Scalar median( Eigen::DenseBase<Derived>& d ){
            auto r { d.reshaped() };
            std::sort( r.begin(), r.end() );
            return r.size() % 2 == 0 ?
                r.segment( (r.size()-2)/2, 2 ).mean() :
                r( r.size()/2 );
        }

        template<typename Derived>
        typename Derived::Scalar median( const Eigen::DenseBase<Derived>& d ){
            typename Derived::PlainObject m { d.replicate(1,1) };
            return median(m);
        }
        
        template<typename Derived, typename T=Derived::Scalar>
        inline auto operator()(const Eigen::ArrayBase<Derived>& t) { 
            Eigen::Array<T,-1,1> tmp(t.rows());
            std::transform(t.rowwise().begin(), t.rowwise().end(), 
                           tmp.begin(),
                           [&](const auto& i){return this->median(i);}
            );
            return tmp;
        }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.median(); }
    };

    template<>
    struct Function<NodeType::Prod>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.rowwise().prod(); }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.prod(); } 
    };
    /* sum */
    template<>
    struct Function<NodeType::Sum>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.rowwise().sum(); }

        inline auto operator()(ArrayXXb t) { 
            return (t.rowwise().count().cast <float> ());
        }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.sum(); } 
    };

    template<>
    struct Function<NodeType::Count>
    {
        inline auto operator()(ArrayXXb t) { return t.rowwise().count(); }

        template<typename T>
        inline auto operator()(const TimeSeries<T>& t) { return t.count(); } 
    };

    /* coefficient-wise maximum of two or more arguments. */
    template<>
    struct Function<NodeType::ArgMax>
    {
        template<typename T>
        inline auto operator()(const T& t) { 
            ArrayXi idx(t.rows());
            for (int i = 0; i < t.rows(); ++i)
                t.row(i).maxCoeff(&idx(i));
            return idx;
        }
    };

    template<>
    struct Function<NodeType::Pow>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1.pow(t2); }
    };

    template<>
    struct Function<NodeType::Abs>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.abs(); }
    };

    template<>
    struct Function<NodeType::Logistic>
    {
        template<typename T>
        inline auto operator()(const T& t) { return float(1.0) / (float(1.0) + (-t).exp()) ; }
    };

    template<>
    struct Function<NodeType::Log>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.log(); }
    };

    template<>
    struct Function<NodeType::Logabs>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.abs().log(); }
    };

    template<>
    struct Function<NodeType::Log1p>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.log1p(); }
    };

    template<>
    struct Function<NodeType::Ceil>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.ceil(); }
    };

    template<>
    struct Function<NodeType::Floor>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.floor(); }
    };

    template<>
    struct Function<NodeType::Exp>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.exp(); }
    };

    template<>
    struct Function<NodeType::Sin>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.sin(); }
    };

    template<>
    struct Function<NodeType::Cos>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.cos(); }
    };

    template<>
    struct Function<NodeType::Tan>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.tan(); }
    };

    template<>
    struct Function<NodeType::Asin>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.asin(); }
    };

    template<>
    struct Function<NodeType::Acos>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.acos(); }
    };

    template<>
    struct Function<NodeType::Atan>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.atan(); }
    };

    template<>
    struct Function<NodeType::Sinh>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.sinh(); }
    };

    template<>
    struct Function<NodeType::Cosh>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.cosh(); }
    };

    template<>
    struct Function<NodeType::Tanh>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.tanh(); }
    };

    template<>
    struct Function<NodeType::Sqrt>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.sqrt(); }
    };

    template<>
    struct Function<NodeType::Sqrtabs>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.abs().sqrt(); }
    };

    template<>
    struct Function<NodeType::Square>
    {
        template<typename T>
        inline auto operator()(const T& t) { return t.square(); }
    };

    template<>
    struct Function<NodeType::Before>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1.before(t2); }
    };

    template<>
    struct Function<NodeType::After>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1.after(t2); }
    };

    template<>
    struct Function<NodeType::During>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2) { return t1.during(t2); }
    };

    template<>
    struct Function<NodeType::SplitOn>
    {
        template<typename T1, typename T2>
        inline auto operator()(const T1& t1, const T2& t2, const T2& t3) { return t2; }
    };
    
    /// @brief Stacks Eigen arrays into a 2d-array, where each array is a column.
    /// @tparam T : underlying type in array
    template<typename T> 
    constexpr auto Stack = [](auto m, auto... ms) {
        return ((Array<T,-1,-1>(m.rows(),1+sizeof...(ms))<<m),...,ms).finished();
    }; 

    template<>
    struct Function<NodeType::Softmax>
    {
       template <typename T>
       inline auto softmax(const ArrayBase<T> &t) const
       {
          auto tMinusMax = t.rowwise() - t.colwise().maxCoeff();
          return tMinusMax.rowwise() - tMinusMax.exp().colwise().sum().log();
       }

       template <typename T>
       inline auto operator()(const ArrayBase<T> &t)
       {
          return this->softmax(t);
       }

    //    template<typename T, typename ...Ts>
    //    inline auto operator()(const Array<T,-1,1>& first, const Ts& ... inputs) 
    //    { 
    //        auto output = Stack<T>(first, inputs...);
    //        return this->softmax(output);
    //    }
    };

} // Brush

#endif
