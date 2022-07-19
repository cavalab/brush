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
    TimeSeriesF,
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
    using ElemType = T;
    using EntryType = Array<T,Dynamic,1>;
    using ValType = std::vector<EntryType>;
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
    /* transform takes a unary function, and applies it to each entry.  */
    auto transform(std::function<EntryType(EntryType)> op) const -> TimeSeries<T>
    {
        ValType dest(this->value.size());
        std::transform(cbegin(), cend(), 
                       dest.begin(),
                       op
        );
        return TimeSeries<T>(dest, this->time);
    }
    /* reduce takes a unary aggregating function, applies it to each entry, and returns an Array.*/
    template<typename R=T>
    auto reduce(const auto& op) const 
    {
        using RetType = Array<R,Dynamic,1>;
        // output "dest" has one entry for each sample. 
        RetType dest(this->value.size());
        std::transform(cbegin(), cend(), 
                       dest.begin(),
                       [&](const EntryType& i){return R(op(i));}
        );
        return dest;
    }; 

    // transformation overloads
    inline auto abs() { return this->transform([](const EntryType& i){ return i.abs(); }); };
    inline auto pow() { return this->transform([](const EntryType& i){ return i.pow(); } ); };
    inline auto log() { return this->transform([](const EntryType& i){ return i.log(); } ); };
    inline auto logabs() { return this->transform([](const EntryType& i){ return i.abs().log(); } ); };
    inline auto log1p() { return this->transform([](const EntryType& i){ return i.log1p(); } ); };
    inline auto ceil() { return this->transform([](const EntryType& i){ return i.ceil(); } ); };
    inline auto floor() { return this->transform([](const EntryType& i){ return i.floor(); } ); };
    inline auto exp() { return this->transform([](const EntryType& i){ return i.exp(); } ); };
    inline auto sin() { return this->transform([](const EntryType& i){ return i.sin(); } ); };
    inline auto cos() { return this->transform([](const EntryType& i){ return i.cos(); } ); };
    inline auto tan() { return this->transform([](const EntryType& i){ return i.tan(); } ); };
    inline auto asin() { return this->transform([](const EntryType& i){ return i.asin(); } ); };
    inline auto acos() { return this->transform([](const EntryType& i){ return i.acos(); } ); };
    inline auto atan() { return this->transform([](const EntryType& i){ return i.atan(); } ); };
    inline auto sinh() { return this->transform([](const EntryType& i){ return i.sinh(); } ); };
    inline auto cosh() { return this->transform([](const EntryType& i){ return i.cosh(); } ); };
    inline auto tanh() { return this->transform([](const EntryType& i){ return i.tanh(); } ); };
    inline auto sqrt() { return this->transform([](const EntryType& i){ return i.sqrt(); } ); };
    inline auto sqrtabs() { return this->transform([](const EntryType& i){ return i.abs().sqrt(); } ); };
    inline auto square() { return this->transform([](const EntryType& i){ return i.square(); } ); };
    // reduction overloads
    inline auto median() { return this->reduce([](const EntryType& i){ return Util::median(i); } ); };
    inline auto mean() { return this->reduce([](const EntryType& i){ return i.mean(); } ); };
    inline auto std() { return this->reduce([](const EntryType& i){ return i.std(); } ); };
    inline auto max() { return this->reduce([](const EntryType& i){ return i.maxCoeff(); } ); };
    inline auto min() { return this->reduce([](const EntryType& i){ return i.minCoeff(); } ); };
    inline auto sum() { return this->reduce<float>([](const EntryType& i){ return i.sum(); } ); };
    inline auto count() { return this->reduce<float>([](const EntryType& i){ return i.size(); } ); };

    template<typename T2>
    inline auto before(const TimeSeries<T2>& t2) const { 
        // return elements of this that are before elements in t2
        // TODO
        return (*this); 
    };
    template<typename T2>
    inline auto after(const TimeSeries<T2>& t2) const { 
        // return elements of this that are after elements in t2
        // TODO
        return (*this); 
    };
    template<typename T2>
    inline auto during(const TimeSeries<T2>& t2) const { 
        // return elements of this that occur within the window in which elements of t2 occur
        // TODO
        return (*this); 
    };

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

/// determines data types of columns of matrix X.
State check_type(const ArrayXf& x);
///////////////////////////////////////////////////////////////////////////////

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
        std::vector<DataType> unique_data_types;
        std::vector<DataType> feature_types;
        std::unordered_map<DataType,vector<string>> features_of_type;

        // TODO: this should probably be a more complex type to include feature type 
        // and potentially other info, like arbitrary relations between features
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
}; // class data
} // data

// TODO: make this a typedef
template<DataType D> struct DataEnumType; 
template<> struct DataEnumType<DataType::ArrayB>{ using type = ArrayXb; };
template<> struct DataEnumType<DataType::ArrayI>{ using type = ArrayXi; };
template<> struct DataEnumType<DataType::ArrayF>{ using type = ArrayXf; };
template<> struct DataEnumType<DataType::MatrixB>{ using type = ArrayXXb; };
template<> struct DataEnumType<DataType::MatrixI>{ using type = ArrayXXi; };
template<> struct DataEnumType<DataType::MatrixF>{ using type = ArrayXXf; };
template<> struct DataEnumType<DataType::TimeSeriesB>{ using type = data::TimeSeriesb; };
template<> struct DataEnumType<DataType::TimeSeriesI>{ using type = data::TimeSeriesi; }; 
template<> struct DataEnumType<DataType::TimeSeriesF>{ using type = data::TimeSeriesf; };

template<typename T> struct DataTypeEnum; 
template<> struct DataTypeEnum<ArrayXb>{ static constexpr DataType value = DataType::ArrayB; };
template<> struct DataTypeEnum<ArrayXi>{ static constexpr DataType value = DataType::ArrayI; };
template<> struct DataTypeEnum<ArrayXf>{ static constexpr DataType value = DataType::ArrayF; };
template<> struct DataTypeEnum<ArrayXXb>{ static constexpr DataType value = DataType::MatrixB; };
template<> struct DataTypeEnum<ArrayXXi>{ static constexpr DataType value = DataType::MatrixI; };
template<> struct DataTypeEnum<ArrayXXf>{ static constexpr DataType value = DataType::MatrixF; };
template<> struct DataTypeEnum<data::TimeSeriesb>{ static constexpr DataType value = DataType::TimeSeriesB; };
template<> struct DataTypeEnum<data::TimeSeriesi>{ static constexpr DataType value = DataType::TimeSeriesI; };
template<> struct DataTypeEnum<data::TimeSeriesf>{ static constexpr DataType value = DataType::TimeSeriesF; };

extern const map<DataType,std::type_index>  DataTypeID;
extern map<std::type_index,DataType>  DataIDType;

} // Brush

#endif
