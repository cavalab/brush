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
#include <optional> 

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

/*!
* @class Dataset
* @brief holds variable type data.
* 
*/
class Dataset 
{
    //TODO: make this a json object that allows elements to be fetched by
    //name 
    //Dataset(ArrayXXf& X, ArrayXf& y, std::map<string, 
    //std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z): X(X), y(y), Z(Z){}
    private:
    public:
        /// @brief keeps track of the unique data types in the dataset. 
        std::vector<DataType> unique_data_types;
        /// @brief types of data in the features.  
        std::vector<DataType> feature_types;
        /// @brief map from data types to features having that type.
        std::unordered_map<DataType,vector<string>> features_of_type;

        
        /// @brief dataset features, as key value pairs
        std::map<string, State> features;
        // TODO: this should probably be a more complex type to include feature type 
        // and potentially other info, like arbitrary relations between features

        
        /// @brief length N array, the target label
        ArrayXf y;
        /// @brief whether this is a classification problem
        bool classification;
        std::optional<std::reference_wrapper<const ArrayXXf>> Xref;

        Dataset operator()(const vector<size_t>& idx) const;
        /// call init at the end of constructors
        /// to define metafeatures of the data.
        void init();

        /// turns input data into a feature map
        map<string,State> make_features(const ArrayXXf& X,
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
        Dataset(const ArrayXXf& X, 
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
                Xref = optional<reference_wrapper<const ArrayXXf>>{X};
            } 

        Dataset(const ArrayXXf& X, const vector<string>& vn) 
            : classification(false)
            , features(make_features(X,map<string, State>{},vn))
            {
                init();
                Xref = optional<reference_wrapper<const ArrayXXf>>{X};
            } 

        void print() const
        {
            fmt::print("Dataset contains {} samples and {} features\n",
                get_n_samples(), get_n_features()
            );
            for (auto& [key, value] : this->features) 
            {
                if (std::holds_alternative<ArrayXf>(value))
                    fmt::print("{}: {}\n", key, std::get<ArrayXf>(value));
                else if (std::holds_alternative<ArrayXi>(value))
                    fmt::print("{}: {}\n", key, std::get<ArrayXi>(value));
                else if (std::holds_alternative<ArrayXb>(value))
                    fmt::print("{}: {}\n", key, std::get<ArrayXb>(value));
            }

        };
        auto get_X() const
        {
            if (Xref.has_value())
                return this->Xref.value().get();
            else
                HANDLE_ERROR_THROW("Dataset does not hold a reference to X.");
        }
        void set_validation(bool v=true);
        inline int get_n_samples() const { 
            return std::visit(
                [&](auto&& arg) -> int { return int(arg.size());}, 
                features.begin()->second
            );
        };
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
