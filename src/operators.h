/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
#include "util/utils.h"

using namespace std;
// using namespace Brush;

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;
using Eigen::Array;
using Eigen::ArrayXi;


namespace Brush
{
/* Partial Derivative Functions 
 *
 * These fns take partial derivates of fns f(x1, ... xm).
 * Each returns an array of partial derivatives of the form 
 * df/dx1, ..., df/dxm 
 */

/// add
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,2>> 
d_add(const T &lhs, const T &rhs) 
{
    return {1, 1};
};

/// add specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,2>> 
d_add(const T &lhs, const T &rhs) 
{
    return {T::Ones(lhs.size()), T::Ones(rhs.size())}; 
};

/// sub
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,2>> 
d_sub(const T &lhs, const T &rhs)  
{
    return {1, -1};
};
/// sub specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,2>> 
d_sub(const T &lhs, const T &rhs)  
{
    return {T::Ones(lhs.size()), -T::Ones(rhs.size())}; 
};

/// multiply
template<typename T>
array<T,2> d_times(const T &lhs, const T &rhs)  
{
    return {rhs, lhs};
}

/// divide
template<typename T>
array<T,2> d_div(const T &lhs, const T &rhs) 
{
    return {1/rhs, -lhs/(pow(rhs,2))};
};

/// log
template<typename T>
array<T,1> d_safe_log(const T &x)
{
    return {Brush::Util::limited(T(1/x))};
};

/// relu
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,1>> 
d_relu(const T &x) 
{
    return {x > 0 ? 1 : 0.01};
};

/// relu specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,1>> 
d_relu(const T &x)  
{
    return {(x > 0).select(ArrayXf::Ones(x.size()), 
                            ArrayXf::Zero(x.size())+0.01)};
}

template<typename T>
array<T,2> d_pow(const T& lhs, const T& rhs) 
{
    return {rhs * pow(lhs, rhs-1), log(lhs) * pow(lhs, rhs)}; 
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
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
relu(const T& x)
{
    return max(x, T(0.01)); 
};

/// relu for Eigen arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
relu(const T& x)
{
    return (x > 0).select(x, T::Zero(x.size())+0.01);
};
// struct OperatorBase
// {
//     static inline const std::string name = "UNDEFINED";
//     static inline const int complexity = -1;
//     // string name;
//     // int complexity;
//     std::string get_name(){return this->name;};
// };
 
// TODO: make this generic, multitype? 
template<typename T, typename U, typename V> struct BinaryOperator;
// template<typename T, typename U> struct UnaryOperator;

template<typename T, typename U=T, typename V=T>
struct BinaryOperator
{
    std::string name;
    int complexity;
    const function<T(U,V)> f;
    const function<array<T,2>(U,V)> df;

    inline BinaryOperator(string n, 
                          int c, 
                          const function<T(U,V)>& f, 
                          const function<array<T,2>(U,V)>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

template<typename T, typename U=T>
struct UnaryOperator
{
    std::string name;
    int complexity;
    const function<T(U)> f;
    const function<array<T,1>(U)> df;

    inline UnaryOperator(string n, 
                          int c, 
                          const function<T(U)>& f, 
                          const function<array<T,1>(U)>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

/* Binary Operators */

template<typename T> 
struct Add : public BinaryOperator<T>
{
    Add(): BinaryOperator<T>("ADD", 2, std::plus<T>(), d_add<T>) {}
};

template<typename T> 
struct Sub : public BinaryOperator<T>
{
    Sub(): BinaryOperator<T>("SUB", 2, std::minus<T>(), d_sub<T>) {}
};

template<typename T> 
struct Times : public BinaryOperator<T>
{
    Times(): BinaryOperator<T>("TIMES", 3, std::multiplies<T>(), d_times<T>) {}
};

template<typename T> 
struct Div : public BinaryOperator<T>
{
    Div(): BinaryOperator<T>("DIV", 4, std::divides<T>(), d_div<T>) {}
};


template<typename T> 
struct Pow : public BinaryOperator<T>
{
    Pow(): BinaryOperator<T>("POW", 7, 
        [](const T& a, const T& b){return pow(a,b);}, 
        d_pow<T>
    ) {}
};

////////////////////////////////////////////////////////////////////////////////
/* Unary Operators */

template<typename T> 
struct Exp : public UnaryOperator<T>
{
    Exp(): UnaryOperator<T>("EXP", 7, 
                            [](const T& x){ return exp(x); }, 
                            [](const T& x) -> array<T,1>{ return {exp(x)}; } 
    ) {}
};

template<typename T> 
struct SafeLog : public UnaryOperator<T> 
{
    SafeLog(): UnaryOperator<T>("LOG", 4, safe_log<T>, d_safe_log<T>) {}
};

template<typename T> 
struct Sin : public UnaryOperator<T>
{
    Sin(): UnaryOperator<T>("SIN", 9,  
                            [](const T& x){ return sin(x); }, 
                            [](const T& x) -> array<T,1>{ return {-cos(x)}; }
    ) {}
};

template<typename T> 
struct Cos : public UnaryOperator<T>
{
    Cos(): UnaryOperator<T>("COS", 9, 
                            [](const T& x){ return cos(x); }, 
                            [](const T& x) -> array<T,1>{ return {sin(x)}; }
    ) {}
};

template<typename T> 
struct Tanh : public UnaryOperator<T>
{
    Tanh(): UnaryOperator<T>("TANH", 9,  
                    [](const T& x){ return tanh(x); }, 
                    [](const T& x) -> array<T,1>{return {1-pow(tanh(x), 2)}; } 
    ) {} 
};

template<typename T> 
struct Sqrt : public UnaryOperator<T>
{
    Sqrt(): UnaryOperator<T>("SQRT", 5, 
                    [](const T& x){ return sqrt(x); }, 
                    [](const T& x) -> array<T,1>{return {x/ (2*sqrt(abs(x)))}; } 
    ){}
};

template<typename T> 
struct Square : public UnaryOperator<T>
{
    Square(): UnaryOperator<T>("SQUARE", 4, 
                               [](const T& x){ return pow(x, 2); },
                               [](const T& x) -> array<T,1>{return {2*x}; }
    ){}
};

template<typename T> 
struct Cube : public UnaryOperator<T>
{
    Cube(): UnaryOperator<T>("CUBE", 4, 
        [](const T& x){ return pow(x, 3); },
        [](const T& x) -> array<T,1>{return {3*pow(x, T(2))}; }
    ){}
};

template<typename T> 
struct Logit : public UnaryOperator<T>
{
    Logit(): UnaryOperator<T>("LOGIT", 3, 
        [](const T& x){ return 1/(1+exp(-x)); },
        [](const T& x) -> array<T,1> { return { exp(-x)/pow(1+exp(-x),T(2)) }; }
    ){}
};

template<typename T>
struct Relu : public UnaryOperator<T>
{
    Relu(): UnaryOperator<T>("RELU", 3, relu<T>, 
        d_relu<T>
    ) {}  
};

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };
//TODO: reduction derivatives (mean, variance, etc.?)
/* } */
template<typename T>
vector<shared_ptr<BinaryOperator<T>>> make_binary_operators()
{
    return {
            make_shared<Add<T>>(),
            make_shared<Sub<T>>(),
            make_shared<Times<T>>(),
            make_shared<Div<T>>(),
            make_shared<Pow<T>>()
    };
}

template<typename T>
vector<shared_ptr<UnaryOperator<T>>> make_unary_operators()
{
    return {
            make_shared<Sin<T>>(),
            make_shared<Cos<T>>(),
            make_shared<Exp<T>>(),
            make_shared<SafeLog<T>>(),
            make_shared<Sqrt<T>>(),
            make_shared<Square<T>>(),
            make_shared<Cube<T>>(),
            make_shared<Tanh<T>>(),
            make_shared<Logit<T>>(),
            make_shared<Relu<T>>()
    };
}


} // Brush

#endif
