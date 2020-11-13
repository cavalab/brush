/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
using namespace std;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;

/* namespace Op{ */

/* template<typename T> */
/* std::function<bool(T,T)> less = std::less<T>(); */ 

/// specialization of binary op for eigen
/* template<typename T> */
/* Matrix<bool, Dynamic, Dynamic> lt( Matrix<T, Dynamic, Dynamic> A, */  
/*     Matrix<T, Dynamic, Dynamic> B) */
/* { return A < B; } */

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };


/* Partial Derivative Functions 
 * f(x1, ... xm) returns an array of partial derivatives of the form 
 * df/dx1, ..., df/dxm 
 */
template<typename T>
struct d_minus {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, -1};
    }
};
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
struct d_plus {
    array<T,2> operator()(const T &lhs, const T &rhs) const 
    {
        return {1, 1};
    }
};
template<>
struct d_plus<ArrayXf> {
    array<ArrayXf,2> operator()(const ArrayXf &lhs, 
                                      const ArrayXf &rhs) const 
    {
        return {ArrayXf::Ones(lhs.size()), ArrayXf::Ones(rhs.size())}; 
    }
};

/* } */
#endif
