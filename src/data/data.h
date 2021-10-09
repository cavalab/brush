/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

// internal includes
#include "../init.h"
#include "../util/utils.h"
#include "../util/error.h"
#include "../util/logger.h"
#include "../util/rnd.h"
// #include <Eigen/Sparse>
// #include <unsupported/Eigen/CXX11/Tensor> 
// using Eigen::Tensor;
//external includes
#include <variant>
using std::min;
using std::iota;
using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXi;
using Eigen::Ref;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXXb;

namespace Brush
{
/**
* @namespace Brush::data
* @brief namespace containing Data structures used in Brush
*/
namespace data
{
/// determines data types of columns of matrix X.
auto typecast(const ArrayXf& x)
{
    // get feature types (binary or continuous/categorical)
    bool isBinary = true;
    bool isCategorical = true;
    std::map<float, bool> uniqueMap;
    for(int i = 0; i < x.size(); i++)
    {
        
        if(x(i) != 0 && x(i) != 1)
            isBinary = false;
        if(x(i) != floor(x(i)) && x(i) != ceil(x(i)))
            isCategorical = false;
        else
            uniqueMap[x(i)] = true;
    } 

    return isBinary ? x.cast<bool>()
                     : (isCategorical && uniqueMap.size() < 10) ? 
                        x.cast<int>() 
                        : x;
    // if (isBinary)
    //     return x.cast<bool>();
    // else
    // {
    //     if(isCategorical && uniqueMap.size() < 10)
    //         return x.cast<int>();
    //     else
    //         return x;
    // }

}

struct TimeSeries
{
    using ts_val = std::variant<ArrayXXb, ArrayXXi, ArrayXXf>;
    /*! Wraps time and value slices to matrices
    *  TODO: define begin() and end() iterators? figure out how to handle operators that just use values versus time 
    */
    ArrayXXf time;
    ts_val value;
    TimeSeries(const ArrayXXf& t, 
               const ts_val& v): time(t), value(v) {}
    // array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const ;
    /// return a slice of the data using indices idx
    template<typename T>
    TimeSeries operator()(const T& idx) const
    {
        ArrayXXf t = time(idx, Eigen::all);
        ts_val v = value(idx, Eigen::all);
        return TimeSeries(t, v);
    };
    /// return a slice of the data by row or column
    template<typename T, typename U>
    TimeSeries operator()(const T& rows, 
                          const U& cols) const
    {
        ArrayXXf t = time(rows, cols);
        ts_val v = value(rows, cols);
        return TimeSeries(t, v);
    };
    // TODO: from_json and to_json
};
typedef std::map<std::string, TimeSeries> Longitudinal;
/// State: defines the possible types of data flowing thru nodes.
typedef std::variant<
                     ArrayXb,
                     ArrayXi, 
                     ArrayXf, 
                     ArrayXXb,
                     ArrayXXi, 
                     ArrayXXf, 
                     TimeSeries
                    > State; 

// explicit deduction guide (not needed as of C++20)
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/// return the implicit type of a State
auto typedState(const State& arg)
{
    // 4. another type-matching visitor: a class with 3 overloaded operator()'s
    // Note: The `(auto arg)` template operator() will bind to `int` and `long`
    //       in this case, but in its absence the `(double arg)` operator()
    //       *will also* bind to `int` and `long` because both are implicitly
    //       convertible to double. When using this form, care has to be taken
    //       that implicit conversions are handled correctly.
    return std::visit(overloaded {
        // [](const auto& arg) -> State { return arg; },
        [](const ArrayXf& arg) -> ArrayXf { return arg; },
        [](const ArrayXi& arg) -> ArrayXi { return arg; },
        [](const ArrayXb& arg) -> ArrayXb { return arg; },
        [](const ArrayXXf& arg) -> ArrayXXf { return arg; },
        [](const ArrayXXi& arg) -> ArrayXXi { return arg; },
        [](const ArrayXXb& arg) -> ArrayXXb { return arg; },
        [](const TimeSeries& arg) -> TimeSeries { return arg; }
    }, arg);
};

// struct Longitudinal
// {
//     /*!
//      * @class Longitudinal
//      * @brief class for holding and manipulation longitudinal data.
//      * contains of two N x p x t tensors of time and values. 
//      * columns contains p feature names and maps to col indices in time and value. 
//      * t is the longest time series. 
//      * Currently does not support sparse data.
//      * time: 
//      *  - use relative time deltas, with first observation at time t=0
//      **/
  
//     Tensor<float, 3> time(1,1,1);  /// time of observation
//     Tensor<float, 3> value(1,1,1); /// observation value
//     std::map<string, unsigned> columns; /// variable names

//     Longitudinal(size_t x) { this->resize(x); };

//     Longitudinal(Tensor<float, 3>& t, 
//                  Tensor<float, 3>& v, 
//                  std::map<string, unsigned> c) : 
//                  time{t}, value{v}, columns{c} {};

//     void resize(size_t x) 
//     {
//         time.resize(x, Eigen::NoChange, Eigen::NoChange); 
//         value.resize(x, Eigen::NoChange, Eigen::NoChange); 
//     };

//     std::array<Longitudinal, 2> split(ArrayXb& mask)
//     {
//         Tensor<float, 3> time1(mask.sum()), value1(mask.sum());
//         Tensor<float, 3> time2(mask.size()-time1.size()), value2(mask.size()-value1.size());

//         tie{ time1, time2 } = util::split(this->time, mask);
//         tie{ value1, value2 } = util::split(this->value, mask);

//         std::array<Longitudinal, 2> result{ 
//             Longitudinal Z1{time1, value1, this->columns},
//             Longitudinal Z2{time2, value2, this->columns}
//         };

//         // TODO
//         // from_json();
//         // to_json();
//     };
// };

// TODO: store names more generally, in dictionary style, instead of in map
// typedef std::map<string, TimeSeries> TimeSeriesMap;

// typedef nlohmann::json Longitudinal;

////////////////////////////////////////////////////////////////////
// using some template magic so that the Data class can 
// take lvalues or rvalues without copying the underlying data, 
// and store a const reference in the lvalue case
// https://www.fluentcpp.com/2018/07/17/how-to-construct-c-objects-without-making-copies/
// template<class T>
// struct const_reference
// {
//    using type = const std::remove_reference_t<T>&;
// };

// template <class T>
// using const_reference_t =  typename const_reference<T>::type;

// template <class T>
// struct add_const_to_value
// {
//    using type =  std::conditional_t<std::is_lvalue_reference_v<T>, const_reference_t<T>, const T>;
// };

// template <class T>
// using add_const_to_value_t =  typename add_const_to_value<T>::type;
////////////////////////////////////////////////////////////////////
// helper constant for the visitor
template<class> inline constexpr bool always_false_v = false;
 
// helper type for the visitor #4
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class Data 
{
    /*!
    * @class Data
    * @brief holds X, y, and Z data.
    * 
    * 
    * X: an N x m matrix of static features, where rows are samples and columns are features
    * y: length N array, the target label
    * Z: Longitudinal data
    */
    //TODO: make this a json object that allows elements to be fetched by
    //name 
    //Data(ArrayXXf& X, ArrayXf& y, std::map<string, 
    //std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z): X(X), y(y), Z(Z){}
    private:
    public:
        // json j;
        const std::map<string, State>& features;

        // ArrayXXf& X;
        const Ref<ArrayXf> y;
        // Longitudinal& Z;
        bool classification;
        bool validation; 
        int n_samples;
        int n_features;
        // type_index ret_type;
        // vector<type_index> data_types;
        vector<string> var_names;
        // map<string, size_t> Xidx, Zidx;

        Data operator()(const vector<size_t>& idx) const;
        /// call init at the end of constructors
        /// to define metafeatures of the data.
        void init()
        {
            n_features = this->features.size();
            // note this will have to change in unsupervised settings
            n_samples = this->y.size();
        }
        // Data()
        // {
        //     // this->y = ArrayXf();
        //     classification=false;
        // }

        Data(std::map<string, State>& d, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             bool c = false
             ): 
             features(d), y(y_), classification(c) {};

        Data(const Ref<const ArrayXXf>& X, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             const Longitudinal& Z = {},
             const vector<string>& vn = {}, 
             bool c = false
            ): 
            var_names(vn), 
            features(make_features(X,Z,vn))
            y(y_),
            classification(c),
            {} 

        /// turns a 
        std::map<string, State> make_features(const Ref<const ArrayXXf>& X,
                                              const Longitudinal& Z = {},
                                              const vector<string>& vn = {}
                                              ) 
        {
            std::map<std::string, State> tmp_features;
            if (vn.empty())
            {
                for (int i = 0; i < X.cols(); ++i)
                {
                    var_names.push_back("x_"+to_string(i));
                }
            }
            else
            {
                if (var_names.size() > X.cols())
                    HANDLE_ERROR_THROW("More var_names than columns in X");
            }

            for (int i = 0; i < var_names.size(); ++i)
            {
                tmp_features[var_names[i]] = typecast(X.col(i).array());
                
            }
            tmp_features.insert(Z.begin(), Z.end());
            return tmp_features;

        };
        // Data(ArrayXf& X, ArrayXf& y, Longitudinal& Z, 
        //      const vector<string>& variable_names = {}, bool c = false);

        void set_validation(bool v=true);
        
        /// select random subset of data for training weights.
        Data get_batch(int batch_size) const;

        std::array<Data, 2> split(const ArrayXb& mask) const;

        // void shuffle();
        // template<typename T>
        // T& operator[](std::string name) { return d.X.row(Xidx[name]); };
        // template<typename T>
        // const T& operator[](std::string name) const { return d.X.row(Xidx[name]); };
        // TODO: templatize these, when d has json. 
        // commenting out non-const for now, it isn't a use case?
        // ArrayXf operator[](std::string name) 
        // { 
        //     return X.row(Xidx.at(name)).array(); 
        // };
        const State operator[](std::string name) const 
        {
            return this->features.at(name);
        }
        /// return a typed feature
        auto get(string& name) const
        {
            return typedState(this->features.at(name));

        }
        // const ArrayXf operator[](std::string name) const 
        // { 
        //     return X.col(Xidx.at(name)).array();
        // };
        // const State get(std::string name) const 
        // {
        //     if ( Xidx.find(name) != Xidx.end() ) 
        //     {
        //         // found in X
        //         return X.col(Xidx.at(name)).array();
        //     } 
        //     else if ( Zidx.find(name) != Zidx.end() )
        //     {
        //         int loc = Zidx.at(name);
        //         Eigen::array<int, 2> offsets = {0, loc, 0};
        //         Eigen::array<int, 2> extents = {0, Z.time.dimension(1), 0};
        //         return TimeSeries(MatrixCast(Z.time.slice(offsets,extents)),
        //                           MatrixCast(Z.value.slice(offsets,extents))
        //                          );
        //         ;
        //     // not found
        //     }
        //     else
        //     {
        //         HANDLE_ERROR_THROW("Variable name not found in data");
        //     }
        // }
};

/* !
    * @class CVData
    * @brief Holds training and validation splits of data, with pointers to 
    * each.
    * */
// class CVData
// {
//     private:
//         bool oCreated;
//         bool tCreated;
//         bool vCreated;
//         // training and validation data
//         ArrayXf X_t;
//         ArrayXXf X_v;
//         ArrayXf y_t;
//         ArrayXf y_v;
//         Longitudinal Z_t;
//         Longitudinal Z_v;
        
//         bool classification;

//     public:
//         Data *o = NULL;     //< pointer to original data
//         Data *v = NULL;     //< pointer to validation data
//         Data *t = NULL;     //< pointer to training data
        
//         CVData();
        
//         ~CVData();
        

//         CVData(ArrayXXf& X, ArrayXf& y, Longitudinal& Z, 
//                const vector<string>& variable_names,
//                bool c=false);
                
//         void setOriginalData(ArrayXXf& X, ArrayXf& y, Longitudinal& Z, 
//                              const vector<string>& variable_names,
//                              bool c=false);
        
//         void setOriginalData(Data *d);
        
//         void setTrainingData(ArrayXXf& X_t, ArrayXf& y_t, Longitudinal& Z_t,
//                              const vector<string>& variable_names,
//                              bool c = false);
        
//         void setTrainingData(Data *d, bool toDelete = false);
        
//         void setValidationData(ArrayXXf& X_v, ArrayXf& y_v, Longitudinal& Z_v,
//                                const vector<string>& variable_names,
//                                bool c = false);
        
//         void setValidationData(Data *d);
        
//         /// shuffles original data
//         void shuffle_data();
        
//         /// split classification data as stratas
//         void split_stratified(float split);
        
//         /// splits data into training and validation folds.
//         void train_test_split(bool shuffle, float split);

//         void split_longitudinal(
//                     Longitudinal& Z,
//                     Longitudinal& Z_t,
//                     Longitudinal& Z_v,
//                     float split);
                    
//         /// reordering utility for shuffling longitudinal data.
//         void reorder_longitudinal(vector<ArrayXf> &vec1, 
//                 const vector<int>& order); 

// };
} // Dat
} // Brush

#endif
