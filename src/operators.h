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

//template specializations for eigen types you are interested in
template<typename T> struct isEigenArray;
template<>
struct isEigenArray<Eigen::ArrayXf> { typedef std::true_type value; };

template<>
struct isEigenArray<Eigen::ArrayXi> { typedef std::true_type value; };

template<typename T> struct isEigenArray { typedef std::false_type value; };

/* Partial Derivative Functions 
 *
 * These fns take partial derivates of fns f(x1, ... xm).
 * Each returns an array of partial derivatives of the form 
 * df/dx1, ..., df/dxm 
 */

/// add
template<class T, class Enable = void>
struct d_add 
{
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, 1};
    }
};
/// add specialization for Eigen Arrays
template<class T>
struct d_add<T, std::enable_if_t<isEigenArray<T>::value>> 
{
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {T::Ones(lhs.size()), T::Ones(rhs.size())}; 
    }
};

template<typename T, class Enable = void>
struct d_sub {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, -1};
    }
};
/// sub specialization for Eigen Arrays
template<class T>
struct d_sub<T, std::enable_if_t<isEigenArray<T>::value>> 
{
    array<T,2> operator()(const T &lhs, 
                          const T &rhs) const 
    {
        return {T::Ones(lhs.size()), 
                -T::Ones(rhs.size())}; 
    }
};
// /// Specialization required for ArrayXi
// template<>
// struct d_sub<ArrayXi> {
//     array<ArrayXi,2> operator()(const ArrayXi &lhs, 
//                                       const ArrayXi &rhs) const 
//     {
//         return {ArrayXi::Ones(lhs.size()), 
//                 -ArrayXi::Ones(rhs.size())}; 
//     }
// };

/// multiply
template<typename T>
struct d_multiplies {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {rhs, lhs};
    }
};

/// divide
template<typename T>
struct d_div {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1/rhs, -lhs/(pow(rhs,2))};
    }
};

/// log
template<typename T>
struct d_safe_log {
    array<T,1> operator()(const T &x) const 
    {
        return {Brush::Util::limited(T(1/x))};
    }
};

/// relu
template<typename T, class Enable = void>
struct d_relu {
    array<T,1> operator()(const T &x) const 
    {
        return {x > 0 ? 1 : 0.01};
    }
};

/// relu specialization for Eigen Arrays
template<typename T>
struct d_relu<T, std::enable_if_t<isEigenArray<T>::value>> 
{
    array<ArrayXf,1> operator()(const ArrayXf &x) const 
    {
       return {(x > 0).select(ArrayXf::Ones(x.size()), 
                              ArrayXf::Zero(x.size())+0.01)};
    }
};

////////////////////////////////////////////////////////////////////////////////

/*! Operator definitions */

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
    std::string get_name(){return this->name;};
    // virtual T operator()(const U& x, const V& y) const = 0;
    // virtual T f(const U&x, const V& y) const = 0;

    // TODO: change df to a function
    // function<array<T,2>(U,V)> df(const U& x, const V& y){return this->dfdx};
    
    const function<T(U,V)> f;
    const function<array<T,2>(U,V)> df;

    inline BinaryOperator(string n, int c, const function<T(U,V)>& f, 
                   const function<array<T,2>(U,V)>& df): 
                   name(n),  complexity(c), f(f), df(df)  {}
};

template<typename T, typename U=T>
struct UnaryOperator
{
    static inline const std::string name = "UNDEFINED";
    static inline const int complexity = -1;
    static inline const function<T(U)> df();
    std::string get_name(){return this->name;};
    virtual T operator()(const U& x) const = 0;
    // virtual array<T,2> df(const U& x, const V& y) const = 0;
};

// // binary operators
template<typename T> struct Add;
template<typename T> struct Sub;
// template<typename T> struct Times;
// template<typename T> struct Div;
// // unary operators
// template<typename T> struct Sin;
// template<typename T> struct Cos;
// template<typename T> struct Tanh;
// template<typename T> struct Exp;
// template<typename T> struct SafeLog;

template<typename T> 
struct Add : public BinaryOperator<T>
{
    using BO = BinaryOperator<T>;
    Add(): BO("ADD", 2, std::plus<T>(), d_add<T>())  {}
};

template<typename T> 
struct Sub : public BinaryOperator<T>
// struct Sub : public BinaryOperator<T(T,T)>
{
    static inline const std::string name = "SUB";
    static inline const int complexity = 2;

    inline T operator()(const T& x, const T& y) const { return x - y;};

    // static inline const function<T(T,T)> df = d_sub<T>(); 
    array<T,2> df(const T& x, const T& y) override {return d_sub<T>(x,y); };
};

template<typename T> 
struct Times : public BinaryOperator<T>
{
    static inline const std::string name = "TIMES";
    static inline const int complexity = 3;

    inline T operator()(const T& x, const T& y) const { return std::multiplies<T>(x,y);};

    static inline const function<T(T,T)> df = d_multiplies<T>(); 
};

template<typename T> 
struct Div : public BinaryOperator<T>
{
    static inline const std::string name = "DIV";
    static inline const int complexity = 4;

    inline T operator()(const T& x, const T& y) const { return x/y;};

    static inline const function<T(T,T)> df = d_div<T>(); 
};

template<typename T> 
struct Pow : public BinaryOperator<T>
{
    static inline const std::string name = "POW";
    static inline const int complexity = 8;

    inline T operator()(const T& x, const T& y) const { return pow(x,y);};

    static inline const function df = \
                    [](const T& lhs, const T& rhs) -> array<T,2> 
                    {
                        return {rhs * pow(lhs, rhs-1), 
                                log(lhs) * pow(lhs, rhs)}; 
                    };
 
};

////////////////////////////////////////////////////////////////////////////////
/* Unary Operators */

template<typename T> 
struct Exp : public UnaryOperator<T>
{
    static inline const std::string name = "EXP";
    static inline const int complexity = 4;

    inline T operator()(const T& x) const override { return exp(x);};

    static inline const function<T(T,T)> df = [](const T& x) -> array<T,1>{return {exp(x)};}; 
};

template<typename T, class Enable = void>
struct SafeLog : public UnaryOperator<T> 
{
    static inline const std::string name = "LOG";
    static inline const int complexity = 4;

    inline T operator()(const T &x) const override
    {
        if (fabs(x) > NEAR_ZERO)
            return log(fabs(x));
        else
            return MIN_FLT;
    }
    static inline const function<T(T,T)> df = d_safe_log<T>(); 
};

template<typename T>
struct SafeLog<T,std::enable_if_t<isEigenArray<T>::value>> : public UnaryOperator<T>
{
    static inline const std::string name = "LOG";
    static inline const int complexity = 4;

    inline T operator()(const ArrayXf &x) const override
    {
        return (abs(x) > NEAR_ZERO).select(log(abs(x)), MIN_FLT);
    };
    static inline const function<T(T,T)> df = d_safe_log<T>(); 
};

template<typename T> 
struct Sin : public UnaryOperator<T>
{
    static inline const std::string name = "Sin";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return sin(x);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {-cos(x)};}; 
};

template<typename T> 
struct Cos : public UnaryOperator<T>
{
    static inline const std::string name = "Cos";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return cos(x);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {sin(x)};}; 
};

template<typename T> 
struct Tanh : public UnaryOperator<T>
{
    static inline const std::string name = "Tanh";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return tanh(x);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {1-pow(tanh(x), 2)}; }; 
};

template<typename T> 
struct Sqrt : public UnaryOperator<T>
{
    static inline const std::string name = "Sqrt";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return pow(x,2);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {x/ (2*sqrt(abs(x)))};}; 
};

template<typename T> 
struct Square : public UnaryOperator<T>
{
    static inline const std::string name = "SQUARE";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return pow(x,2);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {2*x}; };
};

template<typename T> 
struct Cube : public UnaryOperator<T>
{
    static inline const std::string name = "CUBE";
    static inline const int complexity = 5;

    inline T operator()(const T& x) const override { return pow(x,3);};

    static inline const function<T(T,T)> df = \
        [](const T& x) -> array<T,1>{return {3*pow(x, 2)}; };
};

template<typename T> 
struct Logit : public UnaryOperator<T>
{
    static inline const std::string name = "LOGIT";
    static inline const int complexity = 3;

    inline T operator()(const T& x) const override { return 1/(1+exp(-x));};

    static inline const function<T(T,T)> df = \
                    [](const T& x) -> array<T,1> {
                        return { exp(-x)/pow(1+exp(-x),2) }; };
};

template<typename T, class Enable = void>
struct Relu 
{
    static inline const std::string name = "RELU";
    static inline const int complexity = 3;
    
    T operator()(const T &x) const override
    {
       return max(x, 0.01); 
    }
    static inline const function<T(T,T)> df = d_relu<T>(); 
};

template<typename T>
struct Relu<T, std::enable_if_t<isEigenArray<T>::value>> 
{
    static inline const std::string name = "RELU";
    static inline const int complexity = 3;

    T operator()(const T &x) const override
    {
       return (x > 0).select(x, T::Zero(x.size())+0.01); 
    }
    static inline const function<T(T,T)> df = d_relu<T>(); 
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
