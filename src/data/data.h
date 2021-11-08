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


struct TimeSeries
{
    // TODO: this should probably be templated by bool, int, float
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
        ts_val v = this->value;
        std::visit([&](auto&& arg) { arg = arg(idx, Eigen::all); }, v);
        return TimeSeries(t, v);
    };
    friend ostream &operator<<( ostream &output, const TimeSeries &ts );
    
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
/// returns the typeid held in the variant by calling 
/// StateTypeMap.at(variant.index());
extern std::vector<std::type_index> StateTypeMap;
/// returns the typeid held in arg
std::type_index StateType(const State& arg);

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
        std::vector<type_index> data_types;
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
                                              const Longitudinal& Z = {},
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
             const Longitudinal& Z = {},
             const vector<string>& vn = {}, 
             bool c = false
            ) 
            : features(make_features(X,Z,vn))
            , y(y_)
            , classification(c)
            {
                cout << "Reached constructor...\n";
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

} // Dat
} // Brush

#endif
