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

namespace Op{

/* template<typename T> */
/* std::function<bool(T,T)> less = std::less<T>(); */ 

/// specialization of binary op for eigen
/* template<typename T> */
/* Matrix<bool, Dynamic, Dynamic> lt( Matrix<T, Dynamic, Dynamic> A, */  
/*     Matrix<T, Dynamic, Dynamic> B) */
/* { return A < B; } */

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };

/* template<typename T> */
/* inline ArrayXb lt( const ArrayBase<T>& A, const ArrayBase<T>& B) */ 
/* { return A < B; }; */

/* template<typename T> */
/* std::function<Eigen::MatrixBase<bool>(T,T)> cwise_less = lt<T>(); */ 

/* template<typename T> */
/* std::function<T(T,T)> plus = std::plus<T>(); */

/* template<typename T> */
/* std::function<T(T,T)> minus = std::minus<T>(); */

/* template<typename T> */
/* std::function<T(T,T)> multiplies = std::multiplies<T>(); */

}
#endif
