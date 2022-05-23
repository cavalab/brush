/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Code below heavily inspired by heal-research/operon
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
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
using Brush::data::TimeSeriesf;

namespace Brush
{
/* Operators
- In order to work with ceres, operators should be templated to take jets.
- by returning auto, operators preserve expression templates in Eigen. that allows them to be evaluated once after the expression is constructed. 
- might need to extend eigen to handle the median case
https://eigen.tuxfamily.org/dox/TopicCustomizing_Plugins.html
*/
    template<Brush::NodeType N = NodeType::Add>
    struct Function 
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return t1 + t2 + (tn + ...); }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        // template<typename T>
        // inline auto operator()(T t) { return -t; }

        // template<typename T, typename... Tn>
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T t1, T t2, Tn... tn) { return t1 - (t2 + (tn + ...)); }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }

        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return t1 * t2 * (tn * ...); }
    };

    template<>
    struct Function<NodeType::Div>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t.inverse(); }

        // template<typename T, typename... Tn>
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T t1, T t2, Tn... tn) { return t1 / (t2 * (tn * ...)); }
    };

    /* coefficient-wise minimum of two or more arguments. */
    template<>
    struct Function<NodeType::Min>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        
        // template<typename T, typename... Tn>
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return t1.min(t2.min(tn), ...)); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().minCoeff(); }

        // template<>
        // inline auto operator()(TimeSeriesf t) { return t.apply(Function<NodeType::Max>()); }
        template<typename T>
        inline auto operator()(TimeSeries<T> t) { 
            return t.apply([](const auto& i){return i.minCoeff();});
        }
    };

    /* coefficient-wise maximum of two or more arguments. */
    template<>
    struct Function<NodeType::Max>
    {
        // template<typename T>
        // inline auto operator()(T t) { return t; }
        
        template<typename T1, typename T2, typename... Tn>
        inline auto operator()(T1 t1, T2 t2, Tn... tn) { return t1.max(t2.max(tn), ...)); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().maxCoeff(); }

        // template<>
        // inline auto operator()(TimeSeriesf t) { return t.apply(Eigen::maxCoeff()); }
        template<typename T>
        inline auto operator()(TimeSeries<T> t) { 
            return t.apply([](const auto& i){return i.maxCoeff();});
        }
    };

    // // TODO: these need to be specialized for +timeseries :(
    // /* maximum coefficient*
    // template<>
    // struct Function<NodeType::MaxCoeff>
    // {
    //     template<typename T>
    //     inline auto operator()(T t) { return t.rowwise().maxCoeff(); }

    //     // template<>
    //     // inline auto operator()(TimeSeriesf t) { return t.apply(Eigen::maxCoeff()); }
    //     template<typename T>
    //     inline auto operator()(TimeSeries<T> t) { 
    //         return t.apply([](const auto& i){return i.maxCoeff();});
    //     }
        
    // };

    /* minimum coefficient*/
    // template<>
    // struct Function<NodeType::MinCoeff>
    // {
    //     template<typename T>
    //     inline auto operator()(T t) { return t.rowwise().minCoeff(); }

    //     // template<>
    //     // inline auto operator()(TimeSeriesf t) { return t.apply(Function<NodeType::Max>()); }
    //     template<typename T>
    //     inline auto operator()(TimeSeries<T> t) { 
    //         return t.apply([](const auto& i){return i.minCoeff();});
    //     }
        
    // };

    /* mean */
    template<>
    struct Function<NodeType::Mean>
    {
        // template<typename T, typename T2>
        // inline auto operator()(T t) { return t.rowwise().mean(); }

        template<typename T>
        inline auto operator()(T t) { return t.rowwise().mean(); }

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { 
            return t.apply([](const auto& i){return i.mean();});
        }
        
    };

    /* median 
    https://stackoverflow.com/questions/62696455/is-there-a-way-to-find-the-median-value-of-coefficients-of-an-eigen-matrix#62698308
    */
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
    template<>
    struct Function<NodeType::Median>
    {
        template<typename T>
        inline auto operator()(T t) { return median(t); }
        
    };
    /* sum */
    template<>
    struct Function<NodeType::Sum>
    {
        template<typename T>
        inline auto operator()(T t) { return t.rowwise().sum(); }

        template<typename T>
        inline auto operator()(TimeSeries<T> t) { 
            return t.apply([](const auto& i){return i.sum();});
        }
    };


    // // continuations for n-ary functions (add, sub, mul, div, fmin, fmax)
    // template<NodeType N = NodeType::Add>
    // struct ContinuedFunction
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r += t; }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r += t1 + (tn + ...); }
    // };

    // template<>
    // struct ContinuedFunction<NodeType::Sub>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r -= t; }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r -= t1 + (tn + ...); }
    // };

    // template<>
    // struct ContinuedFunction<NodeType::Mul>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r *= t; }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r *= t1 * (tn * ...); }
    // };

    // template<>
    // struct ContinuedFunction<NodeType::Div>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r /= t; }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r /= t1 * (tn * ...); }
    // };

    // template<>
    // struct ContinuedFunction<NodeType::Min>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = r.min(t); }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r = r.min((t1.min(tn), ...)); }
    // };

    // template<>
    // struct ContinuedFunction<NodeType::Max>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = r.max(t); }

    //     template<typename R, typename T, typename... Ts>
    //     inline auto operator()(R r, T t1, Ts... tn) { r = r.max((t1.max(tn), ...)); }
    // };

    // // bin... and unary functions
    // template<>
    // struct Function<NodeType::Aq>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t1, T t2) { r = t1 / (typename T::Scalar{1.0} + t2.square()).sqrt(); }
    // };

    // template<>
    // struct Function<NodeType::Pow>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t1, T t2) { r = t1.pow(t2); }
    // };

    // template<>
    // struct Function<NodeType::Abs>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.abs(); }
    // };

    // template<>
    // struct Function<NodeType::Log>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.log(); }
    // };

    // template<>
    // struct Function<NodeType::Logabs>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.abs().log(); }
    // };

    // template<>
    // struct Function<NodeType::Log1p>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.log1p(); }
    // };

    // template<>
    // struct Function<NodeType::Ceil>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.ceil(); }
    // };

    // template<>
    // struct Function<NodeType::Floor>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.floor(); }
    // };

    // template<>
    // struct Function<NodeType::Exp>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.exp(); }
    // };

    // template<>
    // struct Function<NodeType::Sin>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.sin(); }
    // };

    // template<>
    // struct Function<NodeType::Cos>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.cos(); }
    // };

    // template<>
    // struct Function<NodeType::Tan>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.tan(); }
    // };

    // template<>
    // struct Function<NodeType::Asin>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.asin(); }
    // };

    // template<>
    // struct Function<NodeType::Acos>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.acos(); }
    // };

    // template<>
    // struct Function<NodeType::Atan>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.atan(); }
    // };

    // template<>
    // struct Function<NodeType::Sinh>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.sinh(); }
    // };

    // template<>
    // struct Function<NodeType::Cosh>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.cosh(); }
    // };

    // template<>
    // struct Function<NodeType::Tanh>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.tanh(); }
    // };

    // template<>
    // struct Function<NodeType::Sqrt>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.sqrt(); }
    // };

    // template<>
    // struct Function<NodeType::Sqrtabs>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.abs().sqrt(); }
    // };

    // template<>
    // struct Function<NodeType::Cbrt>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.unaryExpr([](typename T::Scalar const& v) { return ceres::cbrt(v); }); }
    // };

    // template<>
    // struct Function<NodeType::Square>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R r, T t) { r = t.square(); }
    // };

    // template<>
    // struct Function<NodeType::Dynamic>
    // {
    //     template<typename R, typename T>
    //     inline auto operator()(R /*unused*/, T /*unused*/) { /* do nothing */ }
    // };
////////////////////////////////////////////////////////////////////////////////
/* Partial Derivative Functions 
 *
 * These fns calculate partial derivates of fns f(x1, ... xm).
 * In the case that df/dx1 = df/dx2, only one function is defined. 
 * For binary operators that aren't symmetric, d_*_rhs and d_*_lhs are defined. 
 * see also: 
 * https://eigen.tuxfamily.org/dox/group__CoeffwiseMathFunctions.html
 */

    

// auto begin = overloaded { 
//     [](auto arg) {return arg.begin(); };
//     [](Eigen::ArrayBase<bool,Dynamic,Dynamic>& arg) {return arg.rowwise(); };
//     [](Eigen::ArrayBase<int,Dynamic,Dynamic>& arg) {return arg.rowwise(); };
//     [](Eigen::ArrayBase<float,Dynamic,Dynamic>& arg) {return arg.rowwise(); };
// };
// auto end = overloaded {
//     [](auto arg) {return arg.end(); };
//     [](Eigen::ArrayBase<bool,Dynamic,Dynamic>& arg) {return arg.rowwise()+arg.rows()-1; };
//     [](Eigen::ArrayBase<int,Dynamic,Dynamic>& arg) {return arg.rowwise()+arg.rows()-1; };
//     [](Eigen::ArrayBase<float,Dynamic,Dynamic>& arg) {return arg.rowwise()+arg.rows()-1; };
// };
/// add
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// d_add(const T &lhs, const T &rhs) 
// {
//     return T(1);
// };

// /// add specialization for Eigen Arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// d_add(const T &lhs, const T &rhs) 
// {
//     return T::Ones(lhs.rows(),lhs.cols()); 
// };

// /// sub
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// d_sub_lhs(const T &lhs, const T &rhs)  
// {
//     return T(1);
// };
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// d_sub_rhs(const T &lhs, const T &rhs)  
// {
//     return T(-1);
// };
// /// sub specialization for Eigen Arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// d_sub_lhs(const T &lhs, const T &rhs)  
// {
//     return T::Ones(lhs.rows(),lhs.cols());
// };
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T>
// d_sub_rhs(const T &lhs, const T &rhs)  
// {
//     return -T::Ones(rhs.rows(),rhs.cols()); 
// };

// /// multiply
// template<typename T>
// T d_times(const T &lhs, const T &rhs)  
// {
//     return rhs;
// }

// /// divide
// template<typename T>
// T d_div_lhs(const T &lhs, const T &rhs) 
// {
//     return 1/rhs;
// };
// template<typename T>
// T d_div_rhs(const T &lhs, const T &rhs) 
// {
//     return -lhs/(pow(rhs,2));
// };

// /// log
// template<typename T>
// T d_safe_log(const T &x)
// {
//     return Brush::Util::limited(T(1/x));
// };

// /// relu
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// d_relu(const T &x) 
// {
//     return x > 0 ? 1 : 0.01;
// };

// /// relu specialization for Eigen Arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// d_relu(const T &x)  
// {
//     return (x > 0).select(T::Ones(x.rows(),x.cols()), 
//                            T::Zero(x.rows(),x.cols())+0.0001);
// }

// /// logit
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// d_logit(const T &x) 
// {
//     return exp(-x)/pow(1+exp(-x), float(2)); 
// };

// /// logit specialization for Eigen Arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// d_logit(const T &x)  
// {
//     return exp(-x)/(1+exp(-x)).pow(2);
// };

// template<typename T>
// T d_pow_lhs(const T& lhs, const T& rhs) 
// {
//     return rhs * pow(lhs, rhs-1); 
// };

// template<typename T>
// T d_pow_rhs(const T& lhs, const T& rhs) 
// {
//     return log(lhs) * pow(lhs, rhs); 
// };

// ////////////////////////////////////////////////////////////////////////////////

// /*! Operator definitions */

// /// safe log
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// safe_log(const T& x)
// {
//     if (fabs(x) > NEAR_ZERO)
//         return log(T(abs(x)));
//     else
//         return MIN_FLT;
// };

// /// safe log for Eigen arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// safe_log(const T& x)
// {
//     return (abs(x) > NEAR_ZERO).select(log(abs(x)),MIN_FLT);
// };

// /// relu
// struct relu{
//     inline ArrayXf operator()(const ArrayXf& x){
//         return (x > 0).select(x, ArrayXf::Zero(x.size()));
//         };
//     inline ArrayXXf operator()(const ArrayXXf& x){
//         // return (x > 0).select(x, ArrayXXf::Zero(x.rows(),x.cols()));
//         return (x > 0).select(x, 0.0);
//         };
//     // inline ArrayXf operator()(const ArrayXf& x){return max(x, 0.01);};
//     // inline ArrayXXf operator()(const ArrayXf& x){return max(x, 0.01);};
// };
// // template<typename T>
// // std::enable_if_t<std::is_scalar_v<T>, T> 
// // relu(const T& x)
// // {
// //     return max(x, T(0.01)); 
// // };

// // /// relu for Eigen arrays
// // template<typename T>
// // std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// // relu(const T& x)
// // {
// //     // return (x > 0).select(x, T::Zero(x.size())+T(0.01));
// //     return (x > 0).select(x, T::Zero(x.size()));
// // };

// // template<typename T, typename U, typename V> struct BinaryOp;

// template<typename T, typename U=T, typename V=T>
// struct BinaryOp
// {
//     std::string name;
//     int complexity;
//     const function<T(U,V)> f;
//     const vector<function<U(U,V)>> df;

//     inline BinaryOp(string n, 
//                           int c, 
//                           const function<T(U,V)>& f, 
//                           const vector<function<U(U,V)>>& df
//                          ): name(n),  complexity(c), f(f), df(df)  {}
// };

// template<typename T, typename U=T>
// struct UnaryOp
// {
//     std::string name;
//     int complexity;
//     const function<T(U)> f;
//     const vector<function<U(U)>> df;

//     inline UnaryOp(string n, 
//                           int c, 
//                           const function<T(U)>& f, 
//                           const vector<function<U(U)>>& df
//                          ): name(n),  complexity(c), f(f), df(df)  {}
// };

// /* Binary Operators */

// template<typename T> 
// struct Add : public BinaryOp<T>
// {
//     Add(): BinaryOp<T>("ADD", 2, std::plus<T>(), {d_add<T>,d_add<T>}) {}
// };

// template<typename T> 
// struct Sub : public BinaryOp<T>
// {
//     Sub(): BinaryOp<T>("SUB", 2, std::minus<T>(), {d_sub_lhs<T>, d_sub_rhs<T>}) {}
// };

// template<typename T> 
// struct Times : public BinaryOp<T>
// {
//     Times(): BinaryOp<T>("TIMES", 3, std::multiplies<T>(), {d_times<T>,d_times<T>}) {}
// };

// template<typename T> 
// struct Div : public BinaryOp<T>
// {
//     Div(): BinaryOp<T>("DIV", 4, std::divides<T>(), {d_div_lhs<T>,d_div_rhs<T>}) {}
// };


// template<typename T> 
// struct Pow : public BinaryOp<T>
// {
//     Pow(): BinaryOp<T>("POW", 7, 
//         [](const T& a, const T& b){return pow(a,b);}, 
//         {d_pow_lhs<T>, d_pow_rhs<T>}
//     ) {}
// };
// ////////////////////////////////////////////////////////////////////////////////
// /* Unary Operators */

// template<typename T> 
// struct Exp : public UnaryOp<T>
// {
//     Exp(): UnaryOp<T>("EXP", 7, 
//                             [](const T& x){ return exp(x); }, 
//                             {[](const T& x) -> T{ return {exp(x)}; }} 
//     ) {}
// };

// template<typename T> 
// struct SafeLog : public UnaryOp<T> 
// {
//     SafeLog(): UnaryOp<T>("LOG", 4, safe_log<T>, {d_safe_log<T>}) {}
// };

// template<typename T> 
// struct Sin : public UnaryOp<T>
// {
//     Sin(): UnaryOp<T>("SIN", 9,  
//                             [](const T& x){ return sin(x); }, 
//                             {[](const T& x) -> T{ return {-cos(x)}; }}
//     ) {}
// };

// template<typename T> 
// struct Cos : public UnaryOp<T>
// {
//     Cos(): UnaryOp<T>("COS", 9, 
//                             [](const T& x){ return cos(x); }, 
//                             {[](const T& x) -> T{ return {sin(x)}; }}
//     ) {}
// };

// template<typename T> 
// struct Tanh : public UnaryOp<T>
// {
//     Tanh(): UnaryOp<T>("TANH", 9,  
//                     [](const T& x){ return tanh(x); }, 
//                     {[](const T& x) -> T{return {1-pow(tanh(x), 2)}; }}
//     ) {} 
// };

// template<typename T> 
// struct Sqrt : public UnaryOp<T>
// {
//     Sqrt(): UnaryOp<T>("SQRT", 5, 
//                     [](const T& x){ return sqrt(x); }, 
//                     {[](const T& x) -> T{return {x/ (2*sqrt(abs(x)))}; } }
//     ){}
// };

// template<typename T> 
// struct Square : public UnaryOp<T>
// {
//     Square(): UnaryOp<T>("SQUARE", 4, 
//                                [](const T& x){ return pow(x, 2); },
//                                {[](const T& x) -> T{return {2*x}; }}
//     ){}
// };

// template<typename T> 
// struct Cube : public UnaryOp<T>
// {
//     Cube(): UnaryOp<T>("CUBE", 4, 
//         [](const T& x){ return pow(x, 3); },
//         {[](const T& x) -> T{return {3*pow(x, 2)}; }}
//     ){}
// };

// template<typename T> 
// struct Logit : public UnaryOp<T>
// {
//     Logit(): UnaryOp<T>("LOGIT", 3, 
//         [](const T& x){ return 1/(1+exp(-x)); },
//         { d_logit<T> }
//     ){}
// };

// template<typename T>
// struct Relu : public UnaryOp<T>
// {
//     Relu(): UnaryOp<T>("RELU", 3, relu(), {d_relu<T>}) {}  
// };

// template<typename T, typename U>
// inline T lt( const U& A, const U& B) { return T(A < B); };

// ////////////////////////////////////////////////////////////////////////////////
// /* Reductions */
// // https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

// // TODO: make these work with different sized data (longitudinal/ timeseries)

// /* At a point x where there is exactly one function fi such that fi(x) is the median, the derivative of the median is indeed the derivative of fi (this is not the median of the derivatives). 
// At a point where there are more than one functions equal to the median, the derivative of the median exists only if the derivatives of these functions are equal at that point. 
// Otherwise, the derivative of the median does not exist at that point.

// NOTE: for eigen functions, look at using https://stackoverflow.com/questions/21271728/how-to-pass-member-function-pointer-to-stdfunction for getting references to 
// member functions for different types

// */
// template<typename T>
// T d_median(const Array<T,-1,1>& v) 
// {
//     // instantiate a vector
//     vector<float> x(v.size());
//     x.assign(v.data(),v.data()+v.size());
//     // middle element
//     size_t n = x.size()/2;
//     // sort nth element of array
//     nth_element(x.begin(),x.begin()+n,x.end());
//     // if evenly sized, return average of middle two elements
//     if (x.size() % 2 == 0) {
//         nth_element(x.begin(),x.begin()+n-1,x.end());
//         return (x[n] + x[n-1]) / 2;
//     }
//     // otherwise return middle element
//     else
//         return x[n];
// }

// // struct sum {
// //     ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().sum();};
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return i.sum();}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_sum{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };

// // struct mean {
// //     ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().mean();};
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return i.mean();}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_mean{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };

// // struct var {
// //     float variance(ArrayXf v)
// //     {
// //         return pow((v - v.mean()), 2).mean();
// //     }

// //     ArrayXf operator()(const ArrayXXf& x)
// //     { 
// //         ArrayXf tmp(x.rows());
// //         std::transform(x.rowwise().cbegin(), x.rowwise().cend(), 
// //                        tmp.begin(),
// //                        this->variance
// //         );
// //         return tmp;
// //    };
// //    ArrayXf operator()(const TimeSeriesf& x)
// //    {
// //        ArrayXf tmp(x.value.rows());
// //        std::transform(x.value.cbegin(), x.value.cend(), 
// //                       tmp.begin(),
// //                       this->variance
// //        );
// //        return tmp;
// //    };
// // };

// struct d_var {
// //     float variance(ArrayXf v)
// //     {
// //         return pow((v - v.mean()), 2).mean();
// //     }

// //     ArrayXf operator()(const ArrayXXf& x)
// //     { 
// //         ArrayXf tmp(x.rows());
// //         std::transform(x.rowwise().cbegin(), x.rowwise().cend(), 
// //                        tmp.begin(),
// //                        this->variance
// //         );
// //         return tmp;
// //    };
// //    ArrayXf operator()(const TimeSeriesf& x)
// //    {
// //        ArrayXf tmp(x.value.size());
// //        std::transform(x.value.cbegin(), x.value.cend(), 
// //                       tmp.begin(),
// //                       this->variance
// //        );
// //        return tmp;
// //    };
// };
//         // [](const ArrayXXf& x){ 
//         //     return (x - x.rowwise().mean()).pow(2).rowwise().mean(); 
//         // },
//         // { 
//         //     [](const ArrayXXf& x){ 
//         //         return 2/x.cols()*(x - x.rowwise().mean()); 
//         //     } 
//         // }

// // struct min {
// //     ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().minCoeff();};
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return i.minCoeff();}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_min{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };

// // struct max {
// //     ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().maxCoeff();};
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return i.maxCoeff();}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_max{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };
// ////////////////////////////////////////////////////////////////////////////////
// /* Long Operator
//  * Operates on matrices and timeseries, returns array (for whole sample set)
//  */
// template<typename T> // T: float, int, bool
// struct LongOp {
//     using OpType=std::function<T(const Eigen::Ref<const Array<T,Dynamic,1>>)>;
    
//     OpType op;

//     LongOp(OpType o): op(o) {};

//     /* apply op to begin and end iterators */
//     template<typename Iter>                                    
//     Array<T,Dynamic,1> apply(Iter start, Iter end)
//     {
//         Array<T,Dynamic,1> dest(end-start);
//         std::transform(start, end, 
//                        dest.begin(),
//                        this->op
//         );
//         return dest;
//     };
//     Array<T,Dynamic,1> operator()(const Eigen::Ref<const ArrayXXf>& x)
//     {
//         // auto rows = x.rowwise();
//         return this->apply(x.rowwise().cbegin(), x.rowwise().cend());
//     };
//     Array<T,Dynamic,1> operator()(const TimeSeriesf& x)
//     {
//         return this->apply(x.value.cbegin(), x.value.cend());
//     };
// };

// // struct sum : LongOP<float>
// // {
// //     op = [](const auto& i){return i.sum();};
// // };

// // struct median : LongOP<float>
// // {
// //     op = [](const auto& i){return Util::median(i);};
// // };
// // auto median = LongOp<float>([](const auto& i){return Util::median(i);});
// // auto mean = LongOp<float>([](const auto& i){return i.mean();});
// // auto min = LongOp<float>([](const ArrayXf& i){return i.minCoeff();});
// // auto max = LongOp<float>([](const ArrayXf& i){return i.maxCoeff();});
// // auto var = LongOp<float>([](const ArrayXf& i){return pow((i - i.mean()), 2).mean();});
// // // auto sum = LongOp<float>([](const ArrayXf& i){return i.sum();});
// // auto count = LongOp<float>([](const ArrayXf& i){return i.size();});


// // template<typename T> // T: float, int, bool
// // struct DLongOp {
// //     using Dtype = Eigen::Ref<Array<T,Dynamic,1>>;
// //     using OpType=std::function<Dtype(const Dtype)>;
// //     OpType op;

// //     DLongOp(OpType o): op(o) {};

// //     /* apply op to begin and end iterators */
// //     template<typename Iter>                                    
// //     Array<T,Dynamic,1> apply(Iter start, Iter end)
// //     {
// //         Array<T,Dynamic,1> dest(end-start);
// //         std::transform(start, end, 
// //                        dest.begin(),
// //                        this->op
// //         );
// //         return dest;
// //     };
// //     ArrayXXf operator()(const ArrayXXf& x)
// //     {
// //         return this->apply(x.rowwise().begin(), x.rowwise().end());
// //     };
// //     TimeSeriesf operator()(const TimeSeriesf& x)
// //     {
// //         return TimeSeriesf(x.time,
// //                            this->apply(x.value.begin(), x.value.end()));
// //     };
// // };
// // struct d_mean{
// //     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
// //     TimeSeriesf operator()(const TimeSeriesf& x){ 
// //         TimeSeriesf::ValType ones;
// //         for (const auto& i : x.value)
// //             ones.push_back(ArrayXf::Ones(i.rows()));
// //         return TimeSeriesf(x.time, ones); 
// //     };
// // };
// // struct median {
// //     ArrayXf operator()(const ArrayXXf& x)
// //     { 
// //         ArrayXf tmp;
// //         std::transform(x.rowwise().cbegin(), x.rowwise().cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return Util::median(i);}
// //         );
// //         return tmp;
// //     };
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return Util::median(i);}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_median{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };

// // struct count {
// //     ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().count();};
// //     ArrayXf operator()(const TimeSeriesf& x)
// //     {
// //         ArrayXf tmp;
// //         std::transform(x.cbegin(), x.cend(), 
// //                         tmp.begin(),
// //                         [](const auto& i){return i.count();}
// //         );
// //         return tmp;
// //     };
// // };
// struct d_count{
//     ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
//     TimeSeriesf operator()(const TimeSeriesf& x){ 
//         TimeSeriesf::ValType ones;
//         for (const auto& i : x.value)
//             ones.push_back(ArrayXf::Ones(i.rows()));
//         return TimeSeriesf(x.time, ones); 
//     };
// };

// // stat operators that summarize matrices and timeseries 
// // template<typename Arg> struct Sum;
// // template<typename Arg> struct Mean;
// // template<typename Arg> struct Min;
// // template<typename Arg> struct Max;
// // template<typename Arg> struct Median;
// // template<typename Arg> struct Count;

// /** 
//  * Sum for matrices
//  */ 
// template<typename T> 
// struct Sum<T> : public UnaryOp<ArrayXf, T>
// {
//     Sum<T>(): UnaryOp<ArrayXf, T>(
//         "SUM", 1, 
//         LongOp<float>( [](const auto& i){return i.sum();} ), 
//         // sum(),
//         {d_sum()}
//     ){}
// };
// // template<>
// // struct Sum<ArrayXXf> : public UnaryOp<ArrayXf, ArrayXXf>
// // {
// //     Sum<ArrayXXf>(): UnaryOp<ArrayXf, ArrayXXf>(
// //         "SUM", 1, 
// //         LongOp<float>( [](const auto& i){return i.sum();} ), 
// //         // sum(),
// //         {d_sum()}
// //     ){}
// // };

// // /** 
// //  * Sum for Time series
// //  */ 
// // template<>
// // struct Sum<TimeSeriesf> : public UnaryOp<ArrayXf, TimeSeriesf>
// // {
// //     Sum<TimeSeriesf>(): UnaryOp<ArrayXf, TimeSeriesf>(
// //         "SUM", 1, 
// //         LongOp<float>( [](const auto& i){return i.sum();} ), 
// //         // sum(),
// //         {d_sum()}
// //     ){}
// // };

// // template<>
// // struct Mean<ArrayXXf> : public UnaryOp<ArrayXf, ArrayXXf>
// // {
// //     Mean<ArrayXXf>(): UnaryOp<ArrayXf, ArrayXXf>(
// //         "MEAN", 
// //         1, 
// //         mean(),
// //         {d_mean()}
// //         // [](const ArrayXXf& x){ return x.rowwise().mean(); },
// //         // { [](const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols())/x.cols(); } }
// //     ){}
// // };

// // template<>
// // struct Mean<TimeSeriesf> : public UnaryOp<ArrayXf, TimeSeriesf>
// // {
// //     Mean<TimeSeriesf>(): UnaryOp<ArrayXf, TimeSeriesf>(
// //         "MEAN", 
// //         1, 
// //         mean(),
// //         {d_mean()}
// //         // [](const TimeSeriesf& x){ 
// //         //     return x.rowwise().mean(); 
// //         // },
// //         // { [](const TimeSeriesf& x){ return ArrayXXf::Ones(x.rows(), x.cols())/x.cols(); } }
// //     ){}
// // };

// // template<>
// // struct Var<ArrayXXf> : public UnaryOp<ArrayXf, ArrayXXf>
// // {
// //     Var(): UnaryOp<ArrayXf, ArrayXXf>(
// //         "VAR", 
// //         2, 
// //         var(),
// //         {d_var()}
// //     ){}
// // };

// // struct Count : public UnaryOp<ArrayXf, ArrayXXf>
// // {
// //     Count(): UnaryOp<ArrayXf, ArrayXXf>(
// //         "COUNT", 
// //         2, 
// //         [](const ArrayXXf& x){ 
// //             return (x - x.rowwise().mean()).pow(2).rowwise().mean(); 
// //         },
// //         { 
// //             [](const ArrayXXf& x){ 
// //                 return 2/x.cols()*(x - x.rowwise().mean()); 
// //             } 
// //         }
// //     ){}
// // };

// // /// calculate variance when mean provided
// // float variance(const ArrayXf& v, float mean) 
// // {
// //     ArrayXf tmp = mean*ArrayXf::Ones(v.size());
// //     return pow((v - tmp), 2).mean();
// // }

// // /// calculate variance
// // float variance(const ArrayXf& v) 
// // {
// //     float mean = v.mean();
// //     return variance(v, mean);
// // }


// ////////////////////////////////////////////////////////////////////////////////
// /* Operator creation routines */

// template<typename T>
// auto make_op_map(map<string, std::function<shared_ptr<UnaryOp<T>>()>>& op_map)
// {
//     op_map = {
//             {"SIN",         make_shared<Sin<T>> },
//             {"COS",         make_shared<Cos<T>> },
//             {"EXP",         make_shared<Exp<T>> },
//             {"SAFELOG",     make_shared<SafeLog<T>> },
//             {"SQRT",        make_shared<Sqrt<T>> },
//             {"SQUARE",      make_shared<Square<T>> },
//             {"CUBE",        make_shared<Cube<T>> },
//             {"TANH",        make_shared<Tanh<T>> },
//             {"LOGIT",       make_shared<Logit<T>> },
//             {"RELU",        make_shared<Relu<T>> }
//     };

// }
// template<typename T>
// void make_op_map(map<string, std::function<shared_ptr<BinaryOp<T>>()>>& op_map)
// {
//     op_map = {
//             {"ADD",     make_shared<Add<T>> },
//             {"SUB",     make_shared<Sub<T>> },
//             {"TIMES",   make_shared<Times<T>> },
//             {"DIV",     make_shared<Div<T>> },
//             {"POW",     make_shared<Pow<T>> },
//     };

// }

// template<typename O>
// struct OpMaker
// {
//     typedef map<string, std::function<shared_ptr<O>()>> OpMapType;
//     OpMapType op_map;
//     // make_operators(OpMapType om): op_map(om) {}
//     OpMaker(){ make_op_map(this->op_map); };

//     vector<shared_ptr<O>> make(vector<string>& op_names)
//     {
//         vector<shared_ptr<O>> operators;

//         // if op names is empty, return all nodes
//         if (op_names.empty())
//         {
//             operators.resize(op_map.size());
//             transform(op_map.begin(), op_map.end(), 
//                       operators.begin(),
//                       [](const auto& bom){return bom.second();});
//             return operators;
//         }
//         // else, make the operators in op_names
//         for (const auto& op: op_names)
//         {
//             if (op_map.find(op) != op_map.end())
//                 operators.push_back(op_map[op]());
//         }
//         return operators;
//     };

// };

} // Brush

#endif
