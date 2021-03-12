/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
using namespace std;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;

/*! Operator definitions, and their derivatives
 */

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };

template<typename T>
struct safe_log {
    T operator()(const T &x) const 
    {
        if (abs(x) > NEAR_ZERO)
            return log(abs(x));
        else
            return MIN_FLT;
    }
};
template<>
struct safe_log<ArrayXf> {
    ArrayXf operator()(const ArrayXf &x) const 
    {
        return (abs(x) > NEAR_ZERO).select(log(abs(x)),
                                           MIN_FLT);
    }
};

template<typename T>
struct relu {
    T operator()(const T &x) const 
    {
       return max(x, 0.01); 
    }
};

template<>
struct relu<ArrayXf> {
    ArrayXf operator()(const ArrayXf &x) const 
    {
       return (x > 0).select(x, ArrayXf::Zero(x.size())+0.01); 
    }
};
/* Partial Derivative Functions 
 *
 * These fns take partial derivates of fns f(x1, ... xm).
 * Each returns an array of partial derivatives of the form 
 * df/dx1, ..., df/dxm 
 */

template<typename T>
struct d_plus {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, 1};
    }
};
/// Specialization required for ArrayXf
template<>
struct d_plus<ArrayXf> {
    array<ArrayXf,2> operator()(const ArrayXf &lhs, 
                                      const ArrayXf &rhs) const 
    {
        return {ArrayXf::Ones(lhs.size()), ArrayXf::Ones(rhs.size())}; 
    }
};

template<typename T>
struct d_minus {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, -1};
    }
};
/// Specialization required for Array
template<>
struct d_minus<ArrayXf> {
    array<ArrayXf,2> operator()(const ArrayXf &lhs, 
                                      const ArrayXf &rhs) const 
    {
        return {ArrayXf::Ones(lhs.size()), -ArrayXf::Ones(rhs.size())}; 
    }
};

template<typename T>
struct d_multiplies {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {rhs, lhs};
    }
};


template<typename T>
struct d_divides {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1/rhs, -lhs/(pow(rhs,2))};
    }
};

template<typename T>
struct d_safe_log {
    array<T,1> operator()(const T &x) const 
    {
        return {Brush::Util::limited(T(1/x))};
    }
};
template<typename T>
struct d_relu {
    array<T,1> operator()(const T &x) const 
    {
        return {x > 0 ? 1 : 0.01};
    }
};
/// Specialization required for ArrayXf
template<>
struct d_relu<ArrayXf> {
    array<ArrayXf,1> operator()(const ArrayXf &x) const 
    {
       return {(x > 0).select(ArrayXf::Ones(x.size()), 
                             ArrayXf::Zero(x.size())+0.01)};
    }
};

//TODO: reduction derivatives (mean, variance, etc.?)
/* } */
#endif
