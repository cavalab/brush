#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
using Eigen::Dynamic;
using Eigen::Dynamic;

namespace Op{

template<typename T>
std::function<bool(T,T)> less = std::less<T>(); 

/// specialization of binary op for eigen
/* template<typename T> */
/* Matrix<bool, Dynamic, Dynamic> lt( Matrix<T, Dynamic, Dynamic> A, */  
/*     Matrix<T, Dynamic, Dynamic> B) */
/* { return A < B; } */

template<typename T>
inline ArrayXb lt( const ArrayBase<T>& A, const ArrayBase<T>& B) 
{ return A < B; };

template<typename T, typename U>
T lt( U A, U B) { return T(A < B); };
/* template<typename T> */
/* std::function<Eigen::MatrixBase<bool>(T,T)> cwise_less = lt<T>(); */ 

template<typename T>
std::function<T(T,T)> plus = std::plus<T>();

template<typename T>
std::function<T(T,T)> minus = std::minus<T>();

template<typename T>
std::function<T(T,T)> multiplies = std::multiplies<T>();

}
