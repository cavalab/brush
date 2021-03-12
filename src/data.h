/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>

using std::vector;
using Eigen::MatrixXf;
using Eigen::ArrayXf;
using Eigen::VectorXi;
using Eigen::Dynamic;
using Eigen::Map;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;
// internal includes
//#include "params.h"
#include "util/utils.h"
#include "util/error.h"

//#include "node/node.h"
//external includes

namespace Brush
{
/**
* @namespace Brush::Dat
* @brief namespace containing Data structures used in Feat
*/
namespace Dat{
    /*!
     * @class Data
     * @brief data holding X, y, and Z data
     */
    
    struct TimeSeries
    {
        ArrayXf time;
        ArrayXf value;
    };

    typedef std::map<string, TimeSeries> TimeSeriesMap;

    typedef vector<TimeSeriesMap> Longitudinal;

    class Data
    {
        //Data(MatrixXf& X, ArrayXf& y, std::map<string, 
        //std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z): X(X), y(y), Z(Z){}
        public:
 
            MatrixXf& X;
            ArrayXf& y;
            Longitudinal& Z;
            bool classification;
            bool validation; 
            vector<string> X_dtypes;

            Data(MatrixXf& X, ArrayXf& y, Longitudinal& Z, bool c = false);

            void set_validation(bool v=true);
            
            /// select random subset of data for training weights.
            void get_batch(Data &db, int batch_size) const;

            array<Data, 2> split(const ArrayXb& mask) const;
    };
    
    /* !
     * @class DataRef
     * @brief Holds training and validation splits of data, with pointers to 
     * each.
     * */
    class DataRef
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
            
            DataRef();
            
            ~DataRef();
            
    
            DataRef(MatrixXf& X, ArrayXf& y, 
                             Longitudinal& Z, 
                             bool c=false);
                    
            void setOriginalData(MatrixXf& X, ArrayXf& y, 
                    Longitudinal& Z, bool c=false);
            
            void setOriginalData(Data *d);
            
            void setTrainingData(MatrixXf& X_t, ArrayXf& y_t, 
                               Longitudinal& Z_t,
                               bool c = false);
            
            void setTrainingData(Data *d, bool toDelete = false);
            
            void setValidationData(MatrixXf& X_v, ArrayXf& y_v, 
                               Longitudinal& Z_v,
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
}
}

#endif
