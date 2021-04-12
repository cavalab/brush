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
    * @brief data holding X, y, and Z data
    */

struct TimeSeries
{
    ArrayXf time;
    ArrayXf value;
};

// TODO: store names more generally, in dictionary style, instead of in map
typedef std::map<string, TimeSeries> TimeSeriesMap;

typedef vector<TimeSeriesMap> Longitudinal;

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
        void get_batch(Data &db, int batch_size) const;

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
            return X.row(name_to_idx.at(name)).array();
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
