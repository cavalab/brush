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

// serializing vector of shared ptr: https://github.com/nlohmann/json/discussions/2377
// (used in population.h, which has a shared_ptr vector)
namespace nlohmann
{
template <typename T>
struct adl_serializer<std::shared_ptr<T>>
{
    static void to_json(json& j, const std::shared_ptr<T>& opt)
    {
        if (opt)
        {
            j = *opt;
        }
        else
        {
            j = nullptr;
        }
    }

    static void from_json(const json& j, std::shared_ptr<T>& opt)
    {
        if (j.is_null())
        {
            opt = nullptr;
        }
        else
        {
            opt.reset(new T(j.get<T>()));
        }
    }
};
}

// to overload operators and compare our individuals, we need to be able to
// serialize vectors.
// this is intended to be used with DEAP (so our brush individuals
// can be hashed and compared to each other in python side)
template <> 
struct std::hash<std::vector<float>> {
    std::size_t operator()(const std::vector<float>& v) const {
        std::size_t seed = v.size();
        for (const auto& elem : v) {
            seed ^= std::hash<float>{}(elem) +  0x9e3779b9 + (seed <<  6) + (seed >>  2);
        }
        return seed;
    }
};


// namespace std
// {
//     /**
//      * @brief A std::hash specialization for tuples. 
//      * 
//      * See 
//      * 
//      * @tparam TTypes 
//      */
//     template<typename... TTypes>
//     class hash<std::tuple<TTypes...>>
//     {
//     private:
//         typedef std::tuple<TTypes...> Tuple;

//         template<int N>
//         size_t operator()(Tuple value) const { return 0; }

//         template<int N, typename THead, typename... TTail>
//         size_t operator()(Tuple value) const
//         {
//             constexpr int Index = N - sizeof...(TTail) - 1;
//             return hash<THead>()(std::get<Index>(value)) ^ operator()<N, TTail...>(value);
//         }

//     public:
//         size_t operator()(Tuple value) const
//         {
//             return operator()<sizeof...(TTypes), TTypes...>(value);
//         }
//     };
// }

#include <tuple>
namespace std{
    namespace
    {

        // Code from boost
        // Reciprocal of the golden ratio helps spread entropy
        //     and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine:
        //     http://stackoverflow.com/questions/4948780

        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, std::get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, std::get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>> 
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {                                              
            size_t seed = 0;                             
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
            return seed;                                 
        }                                              

    };
}

namespace Brush{
namespace Util{

extern string PBSTR;

extern int PBWIDTH;

// tuple hash
// https://stackoverflow.com/questions/15103975/my-stdhash-for-stdtuples-any-improvements

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
// float median(const Eigen::Ref<const ArrayXf>& v);
template<typename T, typename Scalar=T::Scalar>
Scalar median(const T& v) 
{
    // instantiate a vector
    vector<Scalar> x(v.size());
    x.assign(v.data(),v.data()+v.size());
    // middle element
    size_t n = x.size()/2;
    // sort nth element of array
    nth_element(x.begin(),x.begin()+n,x.end());
    // if evenly sized, return average of middle two elements
    if (x.size() % 2 == 0) {
        nth_element(x.begin(),x.begin()+n-1,x.end());
        return (x[n] + x[n-1]) / Scalar(2);
    }
    // otherwise return middle element
    else
        return x[n];
};

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
    vector<float> med_score_v;

    vector<unsigned> med_size;
    vector<unsigned> med_complexity;
    vector<unsigned> max_size;
    vector<unsigned> max_complexity;

    void update(int index,
                float timer_count,

                float bst_score,
                float bst_score_v,
                float md_score,
                float md_score_v,

                unsigned md_size,
                unsigned md_complexity,
                unsigned mx_size,
                unsigned mx_complexity
                );
};

typedef struct Log_Stats Log_stats;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Log_Stats,
    generation,
    time,

    best_score,
    best_score_v,
    med_score,
    med_score_v,

    med_size,
    med_complexity,
    max_size,
    max_complexity
);

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
template<std::ranges::range T>
void print(T t)
{
    /* auto out = fmt::memory_buffer(); */
    /* vector<char> out; */
    /* std::for_each(first, last, [&](const auto& i){fmt::format_to(std::back_inserter(out), "{}",i);}); */
    /* fmt::print(to_string(out)); */
    fmt::print("{}",t);
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

template<class T, class U>
std::vector<T> slice(const vector<T>& v, const U& idx)
{
    vector<T> result;
    for (const auto& i : idx)
    {
        result.push_back(v.at(i));
    }
    return result;
}

/// Given a map from keys to values, creates a new map from values to keys
template<typename K, typename V>
static map<V, K> reverse_map(const map<K, V>& m) {
    map<V, K> r;
    for (const auto& kv : m)
        r[kv.second] = kv.first;
    return r;
};

template<typename T>
ostream &operator<<( ostream &os, const vector<T>& v )
{ 
    int j = 1;
    size_t len = v.size();
    for (const auto& i : v)
    {
        os << i ;
        if (j != len)
            os << ", ";
    }
    os << endl;

    return os;            
};


// template<typename VariantType, typename T, std::size_t index = 0>
// constexpr std::size_t variant_index() {
//     static_assert(std::variant_size_v<VariantType> > index, 
//                   "Type not found in variant");
//     if constexpr (index == std::variant_size_v<VariantType>) {
//         return index;
//     } else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>) {
//         return index;
//     } else {
//         return variant_index<VariantType, T, index + 1>();
//     }
// } 


} // Util
} // Brush 
#endif
