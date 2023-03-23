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
#include "timeseries.h"

//external includes
#include <variant>

namespace Brush
{

extern map<DataType,string>  DataTypeName; 
extern map<string,DataType>  DataNameType; 
ostream& operator<<(ostream& os, DataType n);


namespace Data
{
/**
* @namespace Brush::Data
* @brief namespace containing Data structures used in Brush
*/


/// determines data types of columns of matrix X.
State check_type(const ArrayXf& x);
DataType StateType(const State& arg);
///////////////////////////////////////////////////////////////////////////////

class Dataset 
{
    /*!
    * @class Dataset
    * @brief holds X, y, and Z data.
    * 
    * 
    * X: an N x m matrix of static features, where rows are samples and columns are features
    * y: length N array, the target label
    * Z: Longitudinal data
    */
    //TODO: make this a json object that allows elements to be fetched by
    //name 
    //Dataset(ArrayXXf& X, ArrayXf& y, std::map<string, 
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
        // const Ref<const ArrayXf> y;
        ArrayXf y;
        // Longitudinal& Z;
        bool classification;
        bool validation; 
        int n_samples;
        int n_features;
        // type_index ret_type;
        // map<string, size_t> Xidx, Zidx;

        Dataset operator()(const vector<size_t>& idx) const;
        /// call init at the end of constructors
        /// to define metafeatures of the data.
        void init();

        /// turns input data into a feature map
        map<string,State> make_features(const Ref<const ArrayXXf>& X,
                                        const map<string, State>& Z = {},
                                        const vector<string>& vn = {}
                                       );

        /// initialize data from a map.
        Dataset(std::map<string, State>& d, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             bool c = false
             ) 
             : features(d) 
             , y(y_)
             , classification(c) 
             {init();};

        /// initialize data from a matrix with feature columns.
        Dataset(const Ref<const ArrayXXf>& X, 
             const Ref<const ArrayXf>& y_ = ArrayXf(), 
             const vector<string>& vn = {}, 
             const map<string, State>& Z = {},
             bool c = false
            ) 
            : features(make_features(X,Z,vn))
            , y(y_)
            , classification(c)
            {
                init();
            } 
        Dataset(const Ref<const ArrayXXf>& X, const vector<string>& vn) 
            : features(make_features(X,map<string, State>{},vn))
            , classification(false)
            {
                init();
            } 

        void set_validation(bool v=true);
        inline int get_n_samples() const { return this->y.size(); };
        inline int get_n_features() const { return this->features.size(); };
        /// select random subset of data for training weights.
        Dataset get_batch(int batch_size) const;

        std::array<Dataset, 2> split(const ArrayXb& mask) const;

        State operator[](std::string name) const 
        {
            if (this->features.find(name) == features.end())
                HANDLE_ERROR_THROW(fmt::format("Couldn't find feature {} in data\n",name));
            return this->features.at(name);
        };

        /* template<> ArrayXb get<ArrayXb>(std::string name) */
}; // class data

// // read csv
// Dataset read_csv(const std::string & path, MatrixXf& X, VectorXf& y, 
//     vector<string>& names, vector<char> &dtypes, bool& binary_endpoint, char sep) ;

} // data

extern const map<DataType,std::type_index>  DataTypeID;
extern map<std::type_index,DataType>  DataIDType;

} // Brush

// format overload for DataType
template <> struct fmt::formatter<Brush::DataType>: formatter<string_view> {
  template <typename FormatContext>
  auto format(Brush::DataType x, FormatContext& ctx) const {
    return formatter<string_view>::format(Brush::DataTypeName.at(x), ctx);
  }
};
// TODO: fmt overload for Data
// template <> struct fmt::formatter<Brush::Data::Dataset>: formatter<string_view> {
//   template <typename FormatContext>
//   auto format(Brush::Data::Dataset& x, FormatContext& ctx) const {
    // return formatter<string_view>::format(Brush::DataTypeName.at(x), ctx);
//   }
// };

#endif
