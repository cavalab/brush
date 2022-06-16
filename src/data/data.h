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

typedef DenseBase<ArrayXXb>::RowwiseReturnType::iterator XXbIt;
typedef DenseBase<ArrayXXi>::RowwiseReturnType::iterator XXiIt;
typedef DenseBase<ArrayXXf>::RowwiseReturnType::iterator XXfIt;

namespace Brush
{
/**
* @namespace Brush::data
* @brief namespace containing Data structures used in Brush
*/
// DataType enum

enum class DataType : uint32_t {
    ArrayB, 
    ArrayI, 
    ArrayF, 
    MatrixB, 
    MatrixI, 
    MatrixF, 
    TimeSeriesB, 
    TimeSeriesI,
    TimeSeriesF
};

extern map<DataType,string>  DataTypeName; 
extern map<string,DataType>  DataNameType; 


namespace data
{

// Map the enum class to actual data types by calling decltype(DataTypeMap[DataType]).

template<class T>
struct TimeSeries
{
    // TODO: this should probably be templated by bool, int, float
    // using ts_val = std::variant<ArrayXXb, ArrayXXi, ArrayXXf>;
    // using ValType = Array<T, Dynamic, Dynamic>;
    using ValType = std::vector<Array<T,Dynamic,1>>;
    using TimeType = std::vector<ArrayXi>;
    /*! Wraps time and value slices to matrices
    *  TODO: define begin() and end() iterators? figure out how to handle operators that just use values versus time 
    */
    TimeSeries() = default; 

    TimeType time;
    ValType value;
    TimeSeries(const TimeType& t, 
               const ValType& v): time(t), value(v) {}
    // array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const ;
    /// return a slice of the data using indices idx
    template<typename U>
    TimeSeries operator()(const U& idx) const
    {
        // TimeType t = this->time(idx, Eigen::all);
        TimeType t = Util::slice(this->time, idx);
        // ValType v = this->value(idx, Eigen::all);
        ValType v = Util::slice(this->value, idx); //(idx, Eigen::all);
        // std::visit([&](auto&& arg) { arg = arg(idx, Eigen::all); }, v);
        return TimeSeries(t, v);
    };

    /**
     * @brief Print the time series.
     * 
     * @param output ostream object
     * @param ts time series object
     * @return ostream& 
     */
    friend ostream &operator<<( ostream &output, const TimeSeries<T> &ts )
    { 
        output << ts.value;

        return output;            
    };
    
    // /// return a slice of the data by row or colum/n
    // template<typename R, typename C>
    // TimeSeries operator()(const R& rows, 
    //                       const C& cols) const
    // {
    //     ArrayXXf t = this->time(rows, cols);
    //     ValType v = this->value(rows, cols);
    //     return TimeSeries(t, v);
    // };
    // TODO: from_json and to_json

    // TODO: custom iterator that iterates over pairs of time and value vectors.
    // for now these only iterate over values.
    
    typename ValType::iterator begin() { return this->value.begin(); };
    typename ValType::iterator end() { return this->value.end(); };
    auto cbegin() const { return this->value.cbegin(); };
    auto cend() const { return this->value.cend(); };
    typename TimeType::iterator tbegin() { return this->time.begin(); };
    typename TimeType::iterator tend() { return this->time.end(); };
    auto ctbegin() const { return this->time.cbegin(); };
    auto ctend() const { return this->time.cend(); };
    // auto rowwise() const { return this->value.rowwise(); };
    // auto colwise() const { return this->value.colwise(); };
    // return std::visit([](auto v){return v.rowwise().end();}, this->value); }

    //TODO: define overloaded operations?
    /* operators on values */
    /* template<typename O> */
    auto apply(std::function<T(ValType)> op){
        Array<T,Dynamic,1> dest(this->value.size());
        std::transform(cbegin(), cend(), 
                       dest.begin(),
                       op
        );
        return dest;
    }; 
    /* operators on time */
    /* template<typename O> */
    auto apply_time(std::function<T(TimeType)> op){
        Array<T,Dynamic,1> dest(this->time.size());
        std::transform(ctbegin(), ctend(), 
                       dest.begin(),
                       op
        );
        return dest;
    }; 

    // inline auto mean() const { return std::apply(this->cbegin(), this->cend(), 
    // [](auto i){ i.mean(); };
    // inline auto mean() const { return this->value.mean(); };
};

/**
 * @brief TimeSeries convenience typedefs.
 * 
 */
typedef TimeSeries<bool> TimeSeriesb;
typedef TimeSeries<int> TimeSeriesi;
typedef TimeSeries<float> TimeSeriesf;


///////////////////////////////////////////////////////////////////////////////
// 
/// State: defines the possible types of data flowing thru nodes.
typedef std::variant<
                     ArrayXb,
                     ArrayXi, 
                     ArrayXf, 
                     ArrayXXb,
                     ArrayXXi, 
                     ArrayXXf, 
                     TimeSeriesb,
                     TimeSeriesi,
                     TimeSeriesf
                    > State; 
/// returns the typeid held in the variant by calling 
/// StateTypeMap.at(variant.index());
extern std::vector<std::type_index> StateTypes;
/// returns the typeid held in arg
std::type_index StateType(const State& arg);
// functions for visiting beginning and end iterators of State
// template<class T>
// DenseBase<T>::RowwiseReturnType::iterator RowBegin(T arg)
// {
//     return arg.rowwise().begin();
// }
struct Begin
{
    ArrayXb::iterator operator()(ArrayXb arg){return arg.begin();};
    ArrayXi::iterator operator()(ArrayXi arg){return arg.begin();} ;
    ArrayXf::iterator operator()(ArrayXf arg){return arg.begin();} ;
    // TimeSeries
    auto operator()(TimeSeriesb arg){return arg.begin();};
    auto operator()(TimeSeriesi arg){return arg.begin();};
    auto operator()(TimeSeriesf arg){return arg.begin();};

    XXbIt operator()(ArrayXXb arg){return arg.rowwise().begin();};
    XXiIt operator()(ArrayXXi arg){return arg.rowwise().begin();};
    XXfIt operator()(ArrayXXf arg){return arg.rowwise().begin();};
    // DenseBase<ArrayXXb>::RowwiseReturnType::iterator RowBegin(T arg)
    // auto operator()(ArrayXXb arg){ return RowBegin<ArrayXXb>(arg); };
    // auto operator()(ArrayXXi arg){ return RowBegin<ArrayXXi>(arg); };
    // auto operator()(ArrayXXf arg){ return RowBegin<ArrayXXf>(arg); };
    // Eigen::VectorwiseOp<ArrayXXi>::iterator operator()(ArrayXXi arg){
    //     return arg.rowwise().begin();
    // };
    // Eigen::VectorwiseOp<ArrayXXf>::iterator operator()(ArrayXXf arg){
    //     return arg.rowwise().begin();
    // };
    // return std::visit( overloaded { 
    // [](auto arg) {return arg.begin(); },
    // [](ArrayXXb& arg) {return arg.rowwise(); }
    // [](ArrayXXi& arg) {return arg.rowwise(); },
    // [](ArrayXXf& arg) {return arg.rowwise(); },
    // },
    // x);
};
struct End
{
    ArrayXb::iterator operator()(ArrayXb arg){return arg.end();};
    ArrayXi::iterator operator()(ArrayXi arg){return arg.end();} ;
    ArrayXf::iterator operator()(ArrayXf arg){return arg.end();} ;
    // TimeSeries
    auto operator()(TimeSeriesb arg){return arg.end();};
    auto operator()(TimeSeriesi arg){return arg.end();};
    auto operator()(TimeSeriesf arg){return arg.end();};
 
    XXbIt operator()(ArrayXXb arg){return arg.rowwise().end();};
    XXiIt operator()(ArrayXXi arg){return arg.rowwise().end();} ;
    XXfIt operator()(ArrayXXf arg){return arg.rowwise().end();} ;
    // return std::visit( overloaded { 
    // [](auto arg) {return arg.begin(); },
    // [](ArrayXXb& arg) {return arg.rowwise(); }
    // [](ArrayXXi& arg) {return arg.rowwise(); },
    // [](ArrayXXf& arg) {return arg.rowwise(); },
    // },
    // x);
};
// auto end(State& x)
// {
//     return std::visit( overloaded { 
//     [](auto arg) {return arg.end(); };
//     [](ArrayXXb& arg) {return arg.rowwise()+arg.rows()-1; };
//     [](ArrayXXi& arg) {return arg.rowwise()+arg.rows()-1; };
//     [](ArrayXXf& arg) {return arg.rowwise()+arg.rows()-1; };
//     },
//     x);
// }

/// determines data types of columns of matrix X.
State check_type(const ArrayXf& x);

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
        std::vector<DataType> data_types;
        Util::TypeMap<vector<string>> features_of_type;

        std::map<string, State> features;

        // ArrayXXf& X;
        const Ref<const ArrayXf> y;
        // Longitudinal& Z;
        bool classification;
        bool validation; 
        int n_samples;
        int n_features;
        // type_index ret_type;
        // map<string, size_t> Xidx, Zidx;

        Data operator()(const vector<size_t>& idx) const;
        /// call init at the end of constructors
        /// to define metafeatures of the data.
        void init();

        /// turns input data into a feature map
        map<string,State> make_features(const Ref<const ArrayXXf>& X,
                                        const map<string, State>& Z = {},
                                        const vector<string>& vn = {}
                                       );

        /// initialize data from a map.
        Data(std::map<string, State>& d, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             bool c = false
             ) 
             : features(d) 
             , y(y_)
             , classification(c) 
             {init();};

        /// initialize data from a matrix with feature columns.
        Data(const Ref<const ArrayXXf>& X, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             const map<string, State>& Z = {},
             const vector<string>& vn = {}, 
             bool c = false
            ) 
            : features(make_features(X,Z,vn))
            , y(y_)
            , classification(c)
            {
                init();
            } 

        void set_validation(bool v=true);
        
        /// select random subset of data for training weights.
        Data get_batch(int batch_size) const;

        std::array<Data, 2> split(const ArrayXb& mask) const;

        State operator[](std::string name) const 
        {
            return this->features.at(name);
        }
};
} // data

template<DataType D = DataType::ArrayB>
struct DataMap{ inline auto operator()() { return ArrayXb(); } };
template<>
struct DataMap<DataType::ArrayI>{ inline auto operator()() { return ArrayXi(); } };
template<>
struct DataMap<DataType::ArrayF>{ inline auto operator()(){ return ArrayXf(); } };
template<>
struct DataMap<DataType::MatrixB>{ inline auto operator()() { return ArrayXXb(); } };
template<>
struct DataMap<DataType::MatrixI>{ inline auto operator()() { return ArrayXXi(); } };
template<>
struct DataMap<DataType::MatrixF>{ inline auto operator()() { return ArrayXXf(); } };
template<>
struct DataMap<DataType::TimeSeriesB>{ inline auto operator()() { return data::TimeSeriesb(); } };
template<>
struct DataMap<DataType::TimeSeriesI>{ inline auto operator()() { return data::TimeSeriesi(); } };
template<>
struct DataMap<DataType::TimeSeriesF>{ inline auto operator()() { return data::TimeSeriesf(); } };

extern map<DataType,std::type_index>  DataTypeID;
extern map<std::type_index,DataType>  DataIDType;
} // Brush

#endif
