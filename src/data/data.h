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
    /**
     * @brief Print the time series.
     * 
     * @param output ostream object
     * @param T time series object
     * @return ostream& 
     */
    friend ostream &operator<<( ostream &output, const TimeSeries &T ) 
    { 
        //  for (int i = 0; i < T.time.rows(); ++i)
        //  {
            // output << "("+T.time(i)+", "+T.value(i)+")";
            // if (i != T.time.rows()-1)
            //     output << ", ";
        //  }
        std::visit([&](const auto& a){output << a;}, T.value);
         return output;            
    }
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
        std::vector<std::string> var_names;
        std::vector<type_index> data_types;
        Util::TypeMap<vector<string>> features_of_type;

        const std::map<string, State>& features;

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
        void init()
        {
            //TODO: populate var_names, var_data_types, data_types, features_of_type
            n_features = this->features.size();
            // note this will have to change in unsupervised settings
            n_samples = this->y.size();

            // debug: print data
            for (const auto& [var, arg] : this->features)
            {
                std::cout << "feature: " << var << endl; 
                std::visit([&](const auto& a){ cout << a << "\n"; }, arg);
            }
        }
        // Data()
        // {
        //     // this->y = ArrayXf();
        //     classification=false;
        // }
        /// turns input data into a feature map
        std::map<string, State> make_features(const Ref<const ArrayXXf>& X,
                                              const Longitudinal& Z = {},
                                              const vector<string>& vn = {}
                                              ) 
        {
            std::map<std::string, State> tmp_features;

            for (int i = 0; i < X.cols(); ++i)
            {
                State tmp = check_type(X.col(i).array());
                // std::type_index feature_type = typeid(std::decay_t<std::decltype(tmp)>);
                std::type_index feature_type = StateType(tmp);

                tmp_features[var_names.at(i)] = tmp;
                // save feature types
                Util::unique_insert(data_types, feature_type);
                // add feature to appropriate map list 
                features_of_type[feature_type].push_back(var_names.at(i));
            }
            cout << "Data:: loaded data_types: ";
            // Util::print(data_types.begin(), data_types.end());
            tmp_features.insert(Z.begin(), Z.end());
            return tmp_features;

        };

        vector<string> init_var_names(const Ref<const ArrayXXf>& X,
                                     const vector<string>& vn = {}
                                    )
        {
            vector<string> tmp;
            if (vn.empty())
            {
                for (int i = 0; i < X.cols(); ++i)
                {
                    string v = "x_"+to_string(i);
                    tmp.push_back(v);
                }
            }
            else
            {
                if (vn.size() != X.cols())
                    HANDLE_ERROR_THROW(to_string(vn.size())
                                       +"variable names and "
                                       +to_string(X.cols())+" features");
            }

            return tmp;
        } 

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
            : var_names(init_var_names(X, vn))
            , features(make_features(X,Z,vn))
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
