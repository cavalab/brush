/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
#include "util/utils.h"
#include "data/data.h"
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
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
d_add(const T &lhs, const T &rhs) 
{
    return T(1);
};

/// add specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
d_add(const T &lhs, const T &rhs) 
{
    return T::Ones(lhs.rows(),lhs.cols()); 
};

/// sub
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
d_sub_lhs(const T &lhs, const T &rhs)  
{
    return T(1);
};
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
d_sub_rhs(const T &lhs, const T &rhs)  
{
    return T(-1);
};
/// sub specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
d_sub_lhs(const T &lhs, const T &rhs)  
{
    return T::Ones(lhs.rows(),lhs.cols());
};
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T>
d_sub_rhs(const T &lhs, const T &rhs)  
{
    return -T::Ones(rhs.rows(),rhs.cols()); 
};

/// multiply
template<typename T>
T d_times(const T &lhs, const T &rhs)  
{
    return rhs;
}

/// divide
template<typename T>
T d_div_lhs(const T &lhs, const T &rhs) 
{
    return 1/rhs;
};
template<typename T>
T d_div_rhs(const T &lhs, const T &rhs) 
{
    return -lhs/(pow(rhs,2));
};

/// log
template<typename T>
T d_safe_log(const T &x)
{
    return Brush::Util::limited(T(1/x));
};

/// relu
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
d_relu(const T &x) 
{
    return x > 0 ? 1 : 0.01;
};

/// relu specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
d_relu(const T &x)  
{
    return (x > 0).select(T::Ones(x.rows(),x.cols()), 
                           T::Zero(x.rows(),x.cols())+0.0001);
}

/// logit
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
d_logit(const T &x) 
{
    return exp(-x)/pow(1+exp(-x), float(2)); 
};

/// logit specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
d_logit(const T &x)  
{
    return exp(-x)/(1+exp(-x)).pow(2);
};

template<typename T>
T d_pow_lhs(const T& lhs, const T& rhs) 
{
    return rhs * pow(lhs, rhs-1); 
};

template<typename T>
T d_pow_rhs(const T& lhs, const T& rhs) 
{
    return log(lhs) * pow(lhs, rhs); 
};

////////////////////////////////////////////////////////////////////////////////

/*! Operator definitions */

/// safe log
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
safe_log(const T& x)
{
    if (fabs(x) > NEAR_ZERO)
        return log(T(abs(x)));
    else
        return MIN_FLT;
};

/// safe log for Eigen arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
safe_log(const T& x)
{
    return (abs(x) > NEAR_ZERO).select(log(abs(x)),MIN_FLT);
};

/// relu
struct relu{
    inline ArrayXf operator()(const ArrayXf& x){
        return (x > 0).select(x, ArrayXf::Zero(x.size()));
        };
    inline ArrayXXf operator()(const ArrayXXf& x){
        // return (x > 0).select(x, ArrayXXf::Zero(x.rows(),x.cols()));
        return (x > 0).select(x, 0.0);
        };
    // inline ArrayXf operator()(const ArrayXf& x){return max(x, 0.01);};
    // inline ArrayXXf operator()(const ArrayXf& x){return max(x, 0.01);};
};
// template<typename T>
// std::enable_if_t<std::is_scalar_v<T>, T> 
// relu(const T& x)
// {
//     return max(x, T(0.01)); 
// };

// /// relu for Eigen arrays
// template<typename T>
// std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
// relu(const T& x)
// {
//     // return (x > 0).select(x, T::Zero(x.size())+T(0.01));
//     return (x > 0).select(x, T::Zero(x.size()));
// };

// template<typename T, typename U, typename V> struct BinaryOp;

template<typename T, typename U=T, typename V=T>
struct BinaryOp
{
    std::string name;
    int complexity;
    const function<T(U,V)> f;
    const vector<function<U(U,V)>> df;

    inline BinaryOp(string n, 
                          int c, 
                          const function<T(U,V)>& f, 
                          const vector<function<U(U,V)>>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

template<typename T, typename U=T>
struct UnaryOp
{
    std::string name;
    int complexity;
    const function<T(U)> f;
    const vector<function<U(U)>> df;

    inline UnaryOp(string n, 
                          int c, 
                          const function<T(U)>& f, 
                          const vector<function<U(U)>>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

/* Binary Operators */

template<typename T> 
struct Add : public BinaryOp<T>
{
    Add(): BinaryOp<T>("ADD", 2, std::plus<T>(), {d_add<T>,d_add<T>}) {}
};

template<typename T> 
struct Sub : public BinaryOp<T>
{
    Sub(): BinaryOp<T>("SUB", 2, std::minus<T>(), {d_sub_lhs<T>, d_sub_rhs<T>}) {}
};

template<typename T> 
struct Times : public BinaryOp<T>
{
    Times(): BinaryOp<T>("TIMES", 3, std::multiplies<T>(), {d_times<T>,d_times<T>}) {}
};

template<typename T> 
struct Div : public BinaryOp<T>
{
    Div(): BinaryOp<T>("DIV", 4, std::divides<T>(), {d_div_lhs<T>,d_div_rhs<T>}) {}
};


template<typename T> 
struct Pow : public BinaryOp<T>
{
    Pow(): BinaryOp<T>("POW", 7, 
        [](const T& a, const T& b){return pow(a,b);}, 
        {d_pow_lhs<T>, d_pow_rhs<T>}
    ) {}
};
////////////////////////////////////////////////////////////////////////////////
/* Unary Operators */

template<typename T> 
struct Exp : public UnaryOp<T>
{
    Exp(): UnaryOp<T>("EXP", 7, 
                            [](const T& x){ return exp(x); }, 
                            {[](const T& x) -> T{ return {exp(x)}; }} 
    ) {}
};

template<typename T> 
struct SafeLog : public UnaryOp<T> 
{
    SafeLog(): UnaryOp<T>("LOG", 4, safe_log<T>, {d_safe_log<T>}) {}
};

template<typename T> 
struct Sin : public UnaryOp<T>
{
    Sin(): UnaryOp<T>("SIN", 9,  
                            [](const T& x){ return sin(x); }, 
                            {[](const T& x) -> T{ return {-cos(x)}; }}
    ) {}
};

template<typename T> 
struct Cos : public UnaryOp<T>
{
    Cos(): UnaryOp<T>("COS", 9, 
                            [](const T& x){ return cos(x); }, 
                            {[](const T& x) -> T{ return {sin(x)}; }}
    ) {}
};

template<typename T> 
struct Tanh : public UnaryOp<T>
{
    Tanh(): UnaryOp<T>("TANH", 9,  
                    [](const T& x){ return tanh(x); }, 
                    {[](const T& x) -> T{return {1-pow(tanh(x), 2)}; }}
    ) {} 
};

template<typename T> 
struct Sqrt : public UnaryOp<T>
{
    Sqrt(): UnaryOp<T>("SQRT", 5, 
                    [](const T& x){ return sqrt(x); }, 
                    {[](const T& x) -> T{return {x/ (2*sqrt(abs(x)))}; } }
    ){}
};

template<typename T> 
struct Square : public UnaryOp<T>
{
    Square(): UnaryOp<T>("SQUARE", 4, 
                               [](const T& x){ return pow(x, 2); },
                               {[](const T& x) -> T{return {2*x}; }}
    ){}
};

template<typename T> 
struct Cube : public UnaryOp<T>
{
    Cube(): UnaryOp<T>("CUBE", 4, 
        [](const T& x){ return pow(x, 3); },
        {[](const T& x) -> T{return {3*pow(x, 2)}; }}
    ){}
};

template<typename T> 
struct Logit : public UnaryOp<T>
{
    Logit(): UnaryOp<T>("LOGIT", 3, 
        [](const T& x){ return 1/(1+exp(-x)); },
        { d_logit<T> }
    ){}
};

template<typename T>
struct Relu : public UnaryOp<T>
{
    Relu(): UnaryOp<T>("RELU", 3, relu(), {d_relu<T>}) {}  
};

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };

////////////////////////////////////////////////////////////////////////////////
/* Reductions */
// https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

// TODO: make these work with different sized data (longitudinal/ timeseries)

/* At a point x where there is exactly one function fi such that fi(x) is the median, the derivative of the median is indeed the derivative of fi (this is not the median of the derivatives). 
At a point where there are more than one functions equal to the median, the derivative of the median exists only if the derivatives of these functions are equal at that point. 
Otherwise, the derivative of the median does not exist at that point.

NOTE: for eigen functions, look at using https://stackoverflow.com/questions/21271728/how-to-pass-member-function-pointer-to-stdfunction for getting references to 
member functions for different types

*/
template<typename T>
T d_median(const Array<T,-1,1>& v) 
{
    // instantiate a vector
    vector<float> x(v.size());
    x.assign(v.data(),v.data()+v.size());
    // middle element
    size_t n = x.size()/2;
    // sort nth element of array
    nth_element(x.begin(),x.begin()+n,x.end());
    // if evenly sized, return average of middle two elements
    if (x.size() % 2 == 0) {
        nth_element(x.begin(),x.begin()+n-1,x.end());
        return (x[n] + x[n-1]) / 2;
    }
    // otherwise return middle element
    else
        return x[n];
}

struct sum {
    ArrayXf operator()(const ArrayXXf& x){ return x.rowwise().sum();};
    ArrayXf operator()(const TimeSeriesf& x)
    {
        ArrayXf tmp;
        std::transform(x.cbegin(), x.cend(), 
                        tmp.begin(),
                        [](const auto& i){return i.sum();}
        );
        return tmp;
    };
};
struct d_sum{
    ArrayXXf operator()(const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols()); };
    TimeSeriesf operator()(const TimeSeriesf& x){ 
        TimeSeriesf::ValType ones;
        for (const auto& i : x.value)
            ones.push_back(ArrayXf::Ones(i.rows()));
        return TimeSeriesf(x.time, ones); 
    };
};
template<typename Arg> struct Sum;

template<>
struct Sum<ArrayXXf> : public UnaryOp<ArrayXf, ArrayXXf>
{
    Sum<ArrayXXf>(): UnaryOp<ArrayXf, ArrayXXf>("SUM", 1, sum(), {d_sum()}){}
};

template<>
struct Sum<TimeSeriesf> : public UnaryOp<ArrayXf, TimeSeriesf>
{
    Sum<TimeSeriesf>(): UnaryOp<ArrayXf, TimeSeriesf>( "SUM", 1, sum(), {d_sum()}
        // [](const TimeSeriesf& x){ 
        //     vector<float> tmp;
        //     std::transform(x.cbegin(), x.cend(), 
        //                    tmp.begin(),
        //                    [](const auto& i){return i.sum();}
        //     );
        //     return ArrayXf::Map(tmp.data(), tmp.size());

        // },
        // { 
        //     [](const TimeSeriesf& x){ 
        //         TimeSeriesf::ValType ones;
        //         for (const auto& i : x.value)
        //             ones.push_back(ArrayXf::Ones(i.rows()));
        //         return TimeSeriesf(x.time, ones); 
        //     } 
        // }
    ){}
};

struct Mean : public UnaryOp<ArrayXf, ArrayXXf>
{
    Mean(): UnaryOp<ArrayXf, ArrayXXf>(
        "MEAN", 
        1, 
        [](const ArrayXXf& x){ return x.rowwise().mean(); },
        { [](const ArrayXXf& x){ return ArrayXXf::Ones(x.rows(), x.cols())/x.cols(); } }
    ){}
};

struct Var : public UnaryOp<ArrayXf, ArrayXXf>
{
    Var(): UnaryOp<ArrayXf, ArrayXXf>(
        "VAR", 
        2, 
        [](const ArrayXXf& x){ 
            return (x - x.rowwise().mean()).pow(2).rowwise().mean(); 
        },
        { 
            [](const ArrayXXf& x){ 
                return 2/x.cols()*(x - x.rowwise().mean()); 
            } 
        }
    ){}
};

// struct Count : public UnaryOp<ArrayXf, ArrayXXf>
// {
//     Count(): UnaryOp<ArrayXf, ArrayXXf>(
//         "COUNT", 
//         2, 
//         [](const ArrayXXf& x){ 
//             return (x - x.rowwise().mean()).pow(2).rowwise().mean(); 
//         },
//         { 
//             [](const ArrayXXf& x){ 
//                 return 2/x.cols()*(x - x.rowwise().mean()); 
//             } 
//         }
//     ){}
// };

// /// calculate variance when mean provided
// float variance(const ArrayXf& v, float mean) 
// {
//     ArrayXf tmp = mean*ArrayXf::Ones(v.size());
//     return pow((v - tmp), 2).mean();
// }

// /// calculate variance
// float variance(const ArrayXf& v) 
// {
//     float mean = v.mean();
//     return variance(v, mean);
// }


////////////////////////////////////////////////////////////////////////////////
/* Operator creation routines */

template<typename T>
void make_op_map(map<string, std::function<shared_ptr<UnaryOp<T>>()>>& op_map)
{
    op_map = {
            {"SIN",         make_shared<Sin<T>> },
            {"COS",         make_shared<Cos<T>> },
            {"EXP",         make_shared<Exp<T>> },
            {"SAFELOG",     make_shared<SafeLog<T>> },
            {"SQRT",        make_shared<Sqrt<T>> },
            {"SQUARE",      make_shared<Square<T>> },
            {"CUBE",        make_shared<Cube<T>> },
            {"TANH",        make_shared<Tanh<T>> },
            {"LOGIT",       make_shared<Logit<T>> },
            {"RELU",        make_shared<Relu<T>> }
    };

}
template<typename T>
void make_op_map(map<string, std::function<shared_ptr<BinaryOp<T>>()>>& op_map)
{
    op_map = {
            {"ADD",     make_shared<Add<T>> },
            {"SUB",     make_shared<Sub<T>> },
            {"TIMES",   make_shared<Times<T>> },
            {"DIV",     make_shared<Div<T>> },
            {"POW",     make_shared<Pow<T>> },
    };

}

template<typename O>
struct OpMaker
{
    typedef map<string, std::function<shared_ptr<O>()>> OpMapType;
    OpMapType op_map;
    // make_operators(OpMapType om): op_map(om) {}
    OpMaker(){ make_op_map(this->op_map); };

    vector<shared_ptr<O>> make(vector<string>& op_names)
    {
        vector<shared_ptr<O>> operators;

        // if op names is empty, return all nodes
        if (op_names.empty())
        {
            operators.resize(op_map.size());
            transform(op_map.begin(), op_map.end(), 
                      operators.begin(),
                      [](const auto& bom){return bom.second();});
            return operators;
        }
        // else, make the operators in op_names
        for (const auto& op: op_names)
        {
            if (op_map.find(op) != op_map.end())
                operators.push_back(op_map[op]());
        }
        return operators;
    };

};

} // Brush

#endif
