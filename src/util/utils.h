/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <fstream>
#include <chrono>
#include <ostream>
#include <map>
#include "../init.h"
#include "error.h"
#include <typeindex>
#include <iterator> // needed for std::ostram_iterator
#include <execution> // parallel policies

using namespace Eigen;
using namespace std;

/**
* @namespace Brush::Util
* @brief namespace containing various utility functions 
*/
namespace Brush{
namespace Util{

extern string PBSTR;

extern int PBWIDTH;

/* a hash map from types to strings of their names. 
https://en.cppreference.com/w/cpp/types/type_info/hash_code
*/

// using TypeInfoPtr = const std::type_info*; 
// struct Hasher {
//     std::size_t operator()(type_index code) const
//     {
//         return code.get().hash_code();
//     }
// };
 
// struct EqualTo {
//     bool operator()(type_index lhs, type_index rhs) const
//     {
//         return lhs.get() == rhs.get();
//     }
// };

// << operator overload for printing vectors
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

template<typename T>
using TypeMap = std::map<std::type_index, T>; 
extern TypeMap<std::string> type_names; 
// enum class TYPES; // int;
// extern TypeMap<TYPES> type_enum;
// using TypeMap = std::unordered_map<TypeInfoPtr, T, Hasher, EqualTo>; 
/// limits node output to be between MIN_FLT and MAX_FLT
void clean(ArrayXf& x);

std::string ltrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
 
std::string rtrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
 
std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r ");

/// check if element is in vector.
template<typename V, typename T>
// template<template<class> class C, class T>
bool in(const V& v, const T& i)
{
    return std::find(v.begin(), v.end(), i) != v.end();
}

/// calculate median
// float median(const ArrayXf& v);
/// calculate median
template<typename T>
T median(const Array<T,-1,1>& v) 
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

/// calculate variance when mean provided
float variance(const ArrayXf& v, float mean);

/// calculate variance
float variance(const ArrayXf& v);

/// calculate skew
float skew(const ArrayXf& v);

/// calculate kurtosis
float kurtosis(const ArrayXf& v);

/// covariance of x and y
float covariance(const ArrayXf& x, const ArrayXf& y);

/// slope of x/y
float slope(const ArrayXf& x, const ArrayXf& y);

/// the normalized covariance of x and y
float pearson_correlation(const ArrayXf& x, const ArrayXf& y);

/// median absolute deviation
float mad(const ArrayXf& x);

/// return indices that sort a vector
template <typename T>
vector<size_t> argsort(const vector<T> &v, bool ascending=true) 
{
    // initialize original index locations
    vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    if (ascending)
    {
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    }
    else
    {
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    }

    return idx;
}

/// class for timing things.
class Timer 
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;

    typedef std::chrono::seconds seconds;

    public:
        explicit Timer(bool run = false);
    
        void Reset();
    
        std::chrono::duration<float> Elapsed() const;
        
        template <typename T, typename Traits>
        friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, 
                                                         const Timer& timer)
        {
            return out << timer.Elapsed().count();
        }
                                                         
        private:
            high_resolution_clock::time_point _start;
    
};

/// return the softmax transformation of a vector.
template <typename T>
vector<T> softmax(const vector<T>& w)
{
    int x;
    T sum = 0;
    vector<T> w_new;
    
    for(x = 0; x < w.size(); ++x)
        sum += exp(w[x]);
        
    for(x = 0; x < w.size(); ++x)
        w_new.push_back(exp(w[x])/sum);
        
    return w_new;
}

/// normalizes a matrix to unit variance, 0 mean centered.
struct Normalizer
{
    Normalizer(bool sa=true): scale_all(sa) {};
    vector<float> scale;
    vector<float> offset;
    vector<char> dtypes;
    bool scale_all;
    
    /// fit the scale and offset of data. 
    void fit(MatrixXf& X, const vector<char>& dt);
    
    /// normalize matrix.
    void normalize(MatrixXf& X);
    
    void fit_normalize(MatrixXf& X, const vector<char>& dtypes);
};

/// calculates data types for each column of X
vector<type_index> get_dtypes(MatrixXf &X);

/// returns unique elements in vector
template <typename T>
vector<T> unique(vector<T> w)
{
    std::sort(w.begin(),w.end());
    typename vector<T>::iterator it;
    it = std::unique(w.begin(),w.end());
    w.resize(std::distance(w.begin(), it));
    return w;
}

/// returns unique elements in Eigen matrix of variable rows/cols
template <typename T>
vector<T> unique(Matrix<T, -1, -1> w)
{
    vector<T> wv( w.data(), w.data()+w.size());
    return unique(wv);
}

/// returns unique elements in Eigen vector
template <typename T>
vector<T> unique(Matrix<T, -1, 1> w)
{
    vector<T> wv( w.data(), w.data()+w.size());
    return unique(wv);
}

/// returns unique elements in 1d Eigen array
template <typename T>
vector<T> unique(Array<T, -1, 1> w)
{
    vector<T> wv( w.data(), w.data()+w.rows()*w.cols());
    return unique(wv);
}

///template function to convert objects to string for logging
template <typename T>
string to_string(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}
///find and replace string
std::string ReplaceString(std::string subject, const std::string& search,
                          const std::string& replace);

///string find and replace in place
void ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace);



/// returns the condition number of a matrix.
float condition_number(const MatrixXf& X);
  
/// returns the pearson correlation coefficients of matrix.
MatrixXf corrcoef(const MatrixXf& X);

// returns the mean of the pairwise correlations of a matrix.
float mean_square_corrcoef(const MatrixXf& X);

/// returns the (first) index of the element with the middlest value in v
int argmiddle(vector<float>& v);

struct Log_Stats
{
    vector<int> generation;
    vector<float> time;
    vector<float> best_score;
    vector<float> best_score_v;
    vector<float> med_score;
    vector<float> med_loss_v;
    vector<unsigned> med_size;
    vector<unsigned> med_complexity;
    vector<unsigned> med_num_params;
    vector<unsigned> med_dim;
    
    void update(int index,
                float timer_count,
                float bst_score,
                float bst_score_v,
                float md_score,
                float md_loss_v,
                unsigned md_size,
                unsigned md_complexity,
                unsigned md_num_params,
                unsigned md_dim);
};

typedef struct Log_Stats Log_stats;

/// limits the output to finite real numbers
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
limited(T x)
{
    if (isnan(x))
        return 0;
    else if (x > MAX_FLT)
        return MAX_FLT;
    else if (x < MIN_FLT)
        return MIN_FLT;

    return x;
};

template<typename T> 
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
limited(T x) 
{
    x = (isnan(x)).select(0,x);
    x = (x < MIN_FLT).select(MIN_FLT,x);
    x = (x > MAX_FLT).select(MAX_FLT,x);
    return x;
};

template<typename T>
void reorder(vector<T> &v, vector<int> const &order )  
{   
    for ( int s = 1; s < order.size(); ++ s ) 
    {
        for ( int d = order[s]; d < s; d = order[d] ) 
        {
            if ( d == s ) 
            {
                while ( d = order[d], d != s ) 
                    swap( v[s], v[d] );
            }
        }
    }
};


/// convert a boolean mask to an index array
vector<size_t> mask_to_index(const ArrayXb& mask);
/// convert a boolean mask into true and false index arrays
tuple<vector<size_t>,vector<size_t>> mask_to_indices(const ArrayXb& mask);

// /// cast a float array to bool or integer if its values comply
// auto typecast(ArrayXf& x);

/// split Eigen matrix or array into two by mask
template<typename T>
array<Array<T,-1, 1>, 2> split(const Array<T,-1,1>& v, const ArrayXb& mask)
// array<DenseBase<T>, 2> split(const DenseBase<T>& v, const ArrayXb& mask)
{
    int size1 = mask.count();
    int size2 = mask.size() - size1;
    Array<T,-1,1> L(size1), R(size2);
    // DenseBase<T> L(size1), R(size2);

    int idx1 = 0, idx2 = 0;
    for (int  i = 0; i < mask.size(); ++i)
    {
        if (mask(i))
        {
            L(idx1) = v(i);
            ++idx1;
        }
        else
        {
            R(idx2) = v(i);
            ++idx2;
        }
    }
    return { L, R };
};

/// prints comma delimited container contents. 
template<typename Iter>
void print(Iter first, Iter last)
{
    std::for_each(first, last, [](const auto& i){std::cout << ", " << i; });
    std::cout << endl;
}


// /*
//  *
//  *  Convert Eigen::Tensor --> Eigen::Matrix
//  *
//  */


// // Evaluates tensor expressions if needed
// template<typename T, typename Device = Eigen::DefaultDevice>
// auto asEval(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, // An Eigen::TensorBase object (Tensor, TensorMap, TensorExpr... )
//             const Device & device = Device()                            // Override to evaluate on another device, e.g. thread pool or gpu.
//             ) {
//     using Evaluator = Eigen::TensorEvaluator<const Eigen::TensorForcedEvalOp<const T>, Device>;
//     Eigen::TensorForcedEvalOp<const T> eval = expr.eval();
//     Evaluator                          tensor(eval, device);
//     tensor.evalSubExprsIfNeeded(nullptr);
//     return tensor;
// }

// // Converts any Eigen::Tensor (or expression) to an Eigen::Matrix with shape rows/cols
// template<typename T, typename sizeType, typename Device = Eigen::DefaultDevice>
// auto MatrixCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, const sizeType rows, const sizeType cols, const Device &device = Device()) {
//     auto tensor  = asEval(expr, device);
//     using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
//     return static_cast<MatrixType<Scalar>>(Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols));
// }

// // Converts any Eigen::Tensor (or expression) to an Eigen::Vector with the same size
// template<typename T, typename Device = Eigen::DefaultDevice>
// auto VectorCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors> &expr, const Device &device = Device()) {
//     auto tensor  = asEval(expr, device);
//     auto size    = Eigen::internal::array_prod(tensor.dimensions());
//     using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
//     return static_cast<VectorType<Scalar>>(Eigen::Map<const VectorType<Scalar>>(tensor.data(), size));
// }

// // View an existing Eigen::Tensor as an Eigen::Map<Eigen::Matrix>
// template<typename Scalar, auto rank, typename sizeType>
// auto MatrixMap(const Eigen::Tensor<Scalar, rank> &tensor, const sizeType rows, const sizeType cols) {
//     return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
// }

// // View an existing Eigen::Tensor of rank 2 as an Eigen::Map<Eigen::Matrix>
// // Rows/Cols are determined from the matrix
// template<typename Scalar>
// auto MatrixMap(const Eigen::Tensor<Scalar, 2> &tensor) {
//     return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
// }

// // View an existing Eigen::Tensor of rank 1 as an Eigen::Map<Eigen::Vector>
// // Rows is the same as the size of the tensor. 
// template<typename Scalar, auto rank>
// auto VectorMap(const Eigen::Tensor<Scalar, rank> &tensor) {
//     return Eigen::Map<const VectorType<Scalar>>(tensor.data(), tensor.size());
// }


// /*
//  *
//  *  Convert Eigen::Matrix --> Eigen::Tensor
//  *
//  * example usage:
//  * 
//  *  Eigen::Tensor<double,4> my_rank4 (2,2,2,2);
//  *  my_rank4.setRandom();
//  *
//  *  Eigen::MatrixXd         mymatrix =  MatrixCast(my_rank4, 4,4);   // Cast Eigen::Tensor --> Eigen::Matrix
//  *  Eigen::Tensor<double,3> my_rank3 =  TensorCast(mymatrix, 2,2,4); // Cast Eigen::Matrix --> Eigen::Tensor
//  */


// // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
// // with dimensions specified in std::array
// template<typename Derived, typename T, auto rank>
// Eigen::Tensor<typename Derived::Scalar, rank>
// TensorCast(const Eigen::EigenBase<Derived> &matrix, const std::array<T, rank> &dims) {
//     return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
//                 (matrix.derived().eval().data(), dims);
// }

// // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
// // with dimensions specified in Eigen::DSizes
// template<typename Derived, typename T, auto rank>
// Eigen::Tensor<typename Derived::Scalar, rank>
// TensorCast(const Eigen::EigenBase<Derived> &matrix, const Eigen::DSizes<T, rank> &dims) {
//     return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
//                 (matrix.derived().eval().data(), dims);
// }

// // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
// // with dimensions as variadic arguments
// template<typename Derived, typename... Dims>
// auto TensorCast(const Eigen::EigenBase<Derived> &matrix, const Dims... dims) {
//     static_assert(sizeof...(Dims) > 0, "TensorCast: sizeof... (Dims) must be larger than 0");
//     return TensorCast(matrix, std::array<Eigen::Index, sizeof...(Dims)>{dims...});
// }

// // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
// // with dimensions directly as arguments in a variadic template
// template<typename Derived>
// auto TensorCast(const Eigen::EigenBase<Derived> &matrix) {
//     if constexpr(Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
//         return TensorCast(matrix, matrix.size());
//     } else {
//         return TensorCast(matrix, matrix.rows(), matrix.cols());
//     }
// }

// // View an existing Eigen::Matrix as Eigen::TensorMap
// // with dimensions specified in std::array
// template<typename Derived, auto rank>
// auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix, const std::array<long, rank> &dims) {
//     return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
// }

// // View an existing Eigen::Matrix as Eigen::TensorMap
// // with dimensions as variadic arguments
// template<typename Derived, typename... Dims>
// auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix, const Dims... dims) {
//     return TensorMap(matrix, std::array<long, static_cast<int>(sizeof...(Dims))>{dims...});
// }

// // View an existing Eigen::Matrix as Eigen::TensorMap
// // with dimensions determined automatically from the given matrix
// template<typename Derived>
// auto TensorMap(const Eigen::PlainObjectBase<Derived> &matrix) {
//     if constexpr(Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
//         return TensorMap(matrix, matrix.size());
//     } else {
//         return TensorMap(matrix, matrix.rows(), matrix.cols());
//     }
// }

/// unique insertion into a vector. 
/// allows a vector to be used like a set. 
/// source: http://www.lafstern.org/matt/col1.pdf
template <class Vector, class T>
void unique_insert(Vector& v, const T& t) 
{
    typename Vector::iterator i = std::lower_bound(v.begin(), v.end(), t);
    if (i == v.end() || t < *i)
    v.insert(i, t);
}

// tupleize a vector. 
// https://stackoverflow.com/questions/28410697/c-convert-vector-to-tuple
template <typename T, std::size_t... Indices>
auto vectorToTupleHelper(const std::vector<T>& v, std::index_sequence<Indices...>) {
  return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto vectorToTuple(const std::vector<T>& v) {
  assert(v.size() >= N);
  return vectorToTupleHelper(v, std::make_index_sequence<N>());
}

// State apply_unary(const Function& f, vector<State>& inputs)
// {
//     R output;
//     std::transform(
//         std::execution::par_unseq,
//         std::visit(Begin(), inputs.at(0)), 
//         std::visit(End(), inputs.at(0)), 
//         std::visit(Begin(), output), 
//         f
//     );

//     return output;
// };
// // specialization for binary operator
// State apply_binary(const Function& f, vector<State>& inputs)
// {
//     R output;
//     std::transform(
//         std::execution::par_unseq,
//         std::visit(Begin(), inputs.at(0)), 
//         std::visit(End(), inputs.at(0)), 
//         std::visit(Begin(), inputs.at(1)), 
//         std::visit(End(), inputs.at(1)), 
//         std::visit(Begin(), output), 
//         f
//     );
//     return output;
// };
template<typename R, typename Arg, typename... Args>
R apply(const std::function<R(Args...)>& f, const vector<Arg>& inputs)
{
    R output;
    switch (inputs.size())
    {
        case 1: 
            std::transform(
                    std::execution::par_unseq,
                    inputs.at(0).begin(),
                    inputs.at(0).end(),
                    output.begin(),
                    f
            );
            break;
        case 2: 
            std::transform(
                std::execution::par_unseq,
                inputs.at(0).begin(),
                inputs.at(0).end(),
                inputs.at(1).begin(),
                // inputs.at(1).end(),
                output.begin(),
                f
            );
            break;
        default: 
            HANDLE_ERROR_THROW("Wrong number of inputs for operator");
            break;
        
    };

    return output;
};


} // Util
} // Brush 
#endif
