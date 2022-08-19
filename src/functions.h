/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Code below heavily inspired by heal-research/operon
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
#include "nodemap.h"
#include "util/utils.h"
#include "data/data.h"
#include "node.h"
using namespace Brush::Util;

using namespace std;
// using namespace Brush;

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;
using Eigen::Array;
using Eigen::ArrayXi;
using Eigen::Dynamic;

using Brush::data::State;
using Brush::data::TimeSeries;
using Brush::data::TimeSeriesf;
using Brush::NodeType;

namespace Brush
{
/* Operators
- In order to work with ceres, operators should be templated to take jets.
- by returning auto, operators preserve expression templates in Eigen. that allows them to be evaluated once after the expression is constructed. 
- might need to extend eigen to handle the median case
https://eigen.tuxfamily.org/dox/TopicCustomizing_Plugins.html
*/
    template<Brush::NodeType N>
    struct Function 
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        template<typename T1, typename... Tn>
        inline auto operator()(T1 t1, Tn... tn) { return t1; }
    };
    /* template<Brush::NodeType N = NodeType::Add> */
    template<>
    struct Function<NodeType::Add> 
    {
        template<typename T1, typename T2>
        inline auto operator()(T1 t1, T2 t2) { return t1 + t2; }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        template<typename T1, typename T2>
        inline auto operator()(T1 t1, T2 t2) { return t1 - t2; }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        template<typename T1, typename T2>
        inline auto operator()(T1 t1, T2 t2) { return t1 * t2 ;}
    };

    template<>
    struct Function<NodeType::Div>
    {
        template<typename T1, typename T2>
        inline auto operator()(T1 t1, T2 t2) { return t1 / t2 ; }
    };

    /* coefficient-wise minimum of two or more arguments. */
    template<>
    struct Function<NodeType::Min>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        
        // template<typename T, typename... Tn>
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return fmin(t1, t2, tn ...); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().minCoeff(); }

        // template<>
        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.min(); } 
    };

    /* coefficient-wise maximum of two or more arguments. */
    template<>
    struct Function<NodeType::Max>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return fmax(t1, t2, tn ...); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().maxCoeff(); }

        // template<>
        // inline auto operator()(TimeSeriesf t) { return t.apply(Eigen::maxCoeff()); }
        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.max(); }
    };

    /* mean */
    template<>
    struct Function<NodeType::Mean>
    {
        // template<typename T, typename T2>
        // inline auto operator()(T t) { return t.rowwise().mean(); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().mean(); }

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.mean(); }
        
    };

    /* median 
    https://stackoverflow.com/questions/62696455/is-there-a-way-to-find-the-median-value-of-coefficients-of-an-eigen-matrix#62698308
    */
    /* template<typename Derived> */
    /* typename Derived::Scalar median( Eigen::DenseBase<Derived>& d ){ */
    /*     auto r { d.reshaped() }; */
    /*     std::sort( r.begin(), r.end() ); */
    /*     return r.size() % 2 == 0 ? */
    /*         r.segment( (r.size()-2)/2, 2 ).mean() : */
    /*         r( r.size()/2 ); */
    /* } */

    /* template<typename Derived> */
    /* typename Derived::Scalar median( const Eigen::DenseBase<Derived>& d ){ */
    /*     typename Derived::PlainObject m { d.replicate(1,1) }; */
    /*     return median(m); */
    /* } */
    template<>
    struct Function<NodeType::Median>
    {
        /* template<typename T> */
        /* inline auto operator()(T t) { return median(t); } */
        
        template<typename T>
        inline auto operator()(Eigen::Array<T,-1,-1> t) { 
            Array<T,-1,1> tmp;
            std::transform(t.rowwise().begin(), t.rowwise().end(), 
                           tmp.begin(),
                           [](const auto& i){return Util::median(i);}
            );
            return tmp;
        }

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.median(); }
    };
    /* sum */
    template<>
    struct Function<NodeType::Sum>
    {
        template<typename T>
        inline auto operator()(T t) { return t.rowwise().sum(); }

        inline auto operator()(ArrayXXb t) { auto tmp = t.rowwise().count(); return (tmp.cast <float> ());}
        /* template<typename T> */
        /* requires (std::is_same_v<T,bool> || std::is_same_v<T,int>) */
        /* inline auto operator()(Array<T,-1,-1> t) { */ 
        /*     ArrayXf tmp = t.rowwise().sum().cast <float> (); */
        /*     return tmp; */ 
        /* } */

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.sum(); } 
    };
    template<>
    struct Function<NodeType::Count>
    {
        /* template<typename T> */
        /* inline auto operator()(T t) { return t.rowwise().size(); } */
        inline auto operator()(ArrayXXb t) { return t.rowwise().count(); }

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { return t.count(); } 
    };


     // bin... and unary functions
     /* template<> */
     /* struct Function<NodeType::Aq> */
     /* { */
     /*     template<typename T> */
     /*     inline auto operator()(T t1, T t2) { return t1 / (typename T::Scalar{1.0} + t2.square()).sqrt(); } */
     /* }; */

     template<>
     struct Function<NodeType::Pow>
     {
         template<typename T>
         inline auto operator()(T t1, T t2) { return t1.pow(t2); }
     };

     template<>
     struct Function<NodeType::Abs>
     {
         template<typename T>
         inline auto operator()(T t) { return t.abs(); }

         /* template<typename T> */
         /* inline auto operator()(TimeSeries<T> t) { return t.transform(std::fabs); } */
         /* inline auto operator()(TimeSeries<T> t) { return t.apply([](const auto& i){return i.abs();}); } */
     };

     template<>
     struct Function<NodeType::Log>
     {
         template<typename T>
         inline auto operator()(T t) { return t.log(); }
     };

     template<>
     struct Function<NodeType::Logabs>
     {
         template<typename T>
         inline auto operator()(T t) { return t.abs().log(); }
     };

     template<>
     struct Function<NodeType::Log1p>
     {
         template<typename T>
         inline auto operator()(T t) { return t.log1p(); }
     };

     template<>
     struct Function<NodeType::Ceil>
     {
         template<typename T>
         inline auto operator()(T t) { return t.ceil(); }
     };

     template<>
     struct Function<NodeType::Floor>
     {
         template<typename T>
         inline auto operator()(T t) { return t.floor(); }
     };

     template<>
     struct Function<NodeType::Exp>
     {
         template<typename T>
         inline auto operator()(T t) { return t.exp(); }
     };

     template<>
     struct Function<NodeType::Sin>
     {
         template<typename T>
         inline auto operator()(T t) { return t.sin(); }
     };

     template<>
     struct Function<NodeType::Cos>
     {
         template<typename T>
         inline auto operator()(T t) { return t.cos(); }
     };

     template<>
     struct Function<NodeType::Tan>
     {
         template<typename T>
         inline auto operator()(T t) { return t.tan(); }
     };

     template<>
     struct Function<NodeType::Asin>
     {
         template<typename T>
         inline auto operator()(T t) { return t.asin(); }
     };

     template<>
     struct Function<NodeType::Acos>
     {
         template<typename T>
         inline auto operator()(T t) { return t.acos(); }
     };

     template<>
     struct Function<NodeType::Atan>
     {
         template<typename T>
         inline auto operator()(T t) { return t.atan(); }
     };

     template<>
     struct Function<NodeType::Sinh>
     {
         template<typename T>
         inline auto operator()(T t) { return t.sinh(); }
     };

     template<>
     struct Function<NodeType::Cosh>
     {
         template<typename T>
         inline auto operator()(T t) { return t.cosh(); }
     };

     template<>
     struct Function<NodeType::Tanh>
     {
         template<typename T>
         inline auto operator()(T t) { return t.tanh(); }
     };

     template<>
     struct Function<NodeType::Sqrt>
     {
         template<typename T>
         inline auto operator()(T t) { return t.sqrt(); }
     };

     template<>
     struct Function<NodeType::Sqrtabs>
     {
         template<typename T>
         inline auto operator()(T t) { return t.abs().sqrt(); }
     };

     /* template<> */
     /* struct Function<NodeType::Cbrt> */
     /* { */
     /*     template<typename T> */
     /*     inline auto operator()(T t) { return t.unaryExpr([](typename T::Scalar const& v) { return std::cbrt(v); }); } */
     /* }; */

     template<>
     struct Function<NodeType::Square>
     {
         template<typename T>
         inline auto operator()(T t) { return t.square(); }
     };

     template<>
     struct Function<NodeType::Before>
     {
         template<typename T1, typename T2>
         inline auto operator()(T1 t1, T2 t2) { return t1.before(t2); }
     };

     template<>
     struct Function<NodeType::After>
     {
         template<typename T1, typename T2>
         inline auto operator()(T1 t1, T2 t2) { return t1.after(t2); }
     };

     template<>
     struct Function<NodeType::During>
     {
         template<typename T1, typename T2>
         inline auto operator()(T1 t1, T2 t2) { return t1.during(t2); }
     };

     template<>
     struct Function<NodeType::SplitOn>
     {
         template<typename T1, typename T2>
         inline auto operator()(T1 t1, T2 t2, T2 t3) { return t2; }
     };
} // Brush

#endif
