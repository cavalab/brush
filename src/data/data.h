/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

#include "../init.h"
using namespace std;
// internal includes
//#include "params.h"
#include "../util/utils.h"
#include "../util/error.h"
#include "../util/logger.h"
#include "../util/rnd.h"
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor> 
//#include "node/node.h"
//external includes

namespace Brush
{
/**
* @namespace Brush::data
* @brief namespace containing Data structures used in Brush
*/
namespace data{
/*!
    * @class Data
    * @brief data holding X, y, and Z data.
    * 
    * 
    * X: an N x m matrix of static features, where rows are samples and columns are features
    * y: length N array, the target label
    * Z: a json object of Longitudinal data. 
    *   each key denotes a TimeSeries that contains two N x t sparse matrices, where t is the maximum number of observations of a feature. One matrix, value, contains the observation, and the other matrix, time, contains the time of observation. 
    * Or...
    * 
    * Z: Longitudinal data
    *     * 
    */

struct Longitudinal
{
    /*!
     * @class Longitudinal
     * @brief class for holding and manipulation longitudinal data.
     * contains of two N x p x t tensors of time and values. 
     * columns contains p feature names and maps to col indices in time and value. 
     * t is the longest time series. 
     * Currently does not support sparse data.
     * time: 
     *  - use relative time deltas, with first observation at time t=0
     **/
  
    Tensor<float, 3> time;  /// time of observation
    Tensor<float, 3> value; /// observation value
    std::map<string, unsigned> columns; /// variable names

    Longitudinal(size_t x) { this->resize(x); }

    void resize(size_t x) 
    {
        time.resize(x, Eigen::NoChange, Eigen::NoChange); 
        value.resize(x, Eigen::NoChange, Eigen::NoChange); 
    };

    tuple<Longitudinal, 2> split(ArrayXb& mask)
    {
        Tensor<float, 3> time1(mask.sum()), value1(mask.sum());
        Tensor<float, 3> time2(mask.size()-Z1.size()), value2(mask.size()-Z1.size());

        tie{ time1, time2 } = util::split(this->time, mask);
        tie{ value1, value2 } = util::split(this->value, mask);

        tuple<Longitudinal, 2> result = { 
            Longitudinal Z1{time1, value1, this->columns};
            Longitudinal Z2{time2, value2, this->columns};
        };

        // TODO
        // from_json();
        // to_json();

    };
};
struct TimeSeries
{
    SparseMatrix<float> time;
    SparseMatrix<float> value;

    TimeSeries(size_t x) { time.resize(x, Eigen::NoChange); value.resize(x, Eigen::NoChange); };
    // array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const ;
    // TODO: from_json and to_json
};

// TODO: store names more generally, in dictionary style, instead of in map
// typedef std::map<string, TimeSeries> TimeSeriesMap;

typedef nlohmann::json Longitudinal;

class Data
{
    //TODO: make this a json object that allows elements to be fetched by
    //name 
    //Data(MatrixXf& X, ArrayXf& y, std::map<string, 
    //std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z): X(X), y(y), Z(Z){}
    public:

        MatrixXf& X;
        ArrayXf& y;
        Longitudinal& Z;
        bool classification;
        bool validation; 
        // type_index ret_type;
        vector<type_index> data_types;
        vector<string> var_names;
        map<string, size_t> name_to_idx;

        Data(MatrixXf& X, ArrayXf& y, Longitudinal& Z, 
             const vector<string>& variable_names = {}, bool c = false);

        void set_validation(bool v=true);
        
        /// select random subset of data for training weights.
        Data get_batch(int batch_size) const;

        array<Data, 2> split(const ArrayXb& mask) const;

        // template<typename T>
        // T& operator[](std::string name) { return d.X.row(name_to_idx[name]); };
        // template<typename T>
        // const T& operator[](std::string name) const { return d.X.row(name_to_idx[name]); };
        // TODO: templatize these, when d has json. 
        // commenting out non-const for now, it isn't a use case?
        // ArrayXf operator[](std::string name) 
        // { 
        //     return X.row(name_to_idx.at(name)).array(); 
        // };
        const ArrayXf operator[](std::string name) const 
        { 
            return X.col(name_to_idx.at(name)).array();
        };
};

/* !
    * @class CVData
    * @brief Holds training and validation splits of data, with pointers to 
    * each.
    * */
class CVData
{
    private:
        bool oCreated;
        bool tCreated;
        bool vCreated;
        // training and validation data
        MatrixXf X_t;
        MatrixXf X_v;
        ArrayXf y_t;
        ArrayXf y_v;
        Longitudinal Z_t;
        Longitudinal Z_v;
        
        bool classification;

    public:
        Data *o = NULL;     //< pointer to original data
        Data *v = NULL;     //< pointer to validation data
        Data *t = NULL;     //< pointer to training data
        
        CVData();
        
        ~CVData();
        

        CVData(MatrixXf& X, ArrayXf& y, Longitudinal& Z, 
               const vector<string>& variable_names,
               bool c=false);
                
        void setOriginalData(MatrixXf& X, ArrayXf& y, Longitudinal& Z, 
                             const vector<string>& variable_names,
                             bool c=false);
        
        void setOriginalData(Data *d);
        
        void setTrainingData(MatrixXf& X_t, ArrayXf& y_t, Longitudinal& Z_t,
                             const vector<string>& variable_names,
                             bool c = false);
        
        void setTrainingData(Data *d, bool toDelete = false);
        
        void setValidationData(MatrixXf& X_v, ArrayXf& y_v, Longitudinal& Z_v,
                               const vector<string>& variable_names,
                               bool c = false);
        
        void setValidationData(Data *d);
        
        /// shuffles original data
        void shuffle_data();
        
        /// split classification data as stratas
        void split_stratified(float split);
        
        /// splits data into training and validation folds.
        void train_test_split(bool shuffle, float split);

        void split_longitudinal(
                    Longitudinal& Z,
                    Longitudinal& Z_t,
                    Longitudinal& Z_v,
                    float split);
                    
        /// reordering utility for shuffling longitudinal data.
        void reorder_longitudinal(vector<ArrayXf> &vec1, 
                const vector<int>& order); 

};
} // Dat
} // Brush

#endif
