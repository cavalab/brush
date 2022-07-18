/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

//internal includes
#include "data.h"

using namespace Brush::Util;
using std::min;

namespace Brush{ 

map<DataType,string>  DataTypeName = {
    {DataType::ArrayB, "ArrayB"},
    {DataType::ArrayI, "ArrayI"},
    {DataType::ArrayF, "ArrayF"},
    {DataType::MatrixB, "MatrixB"},
    {DataType::MatrixI, "MatrixI"},
    {DataType::MatrixF, "MatrixF"},
    {DataType::TimeSeriesB, "TimeSeriesB"},
    {DataType::TimeSeriesI,"TimeSeriesI"},
    {DataType::TimeSeriesF, "TimeSeriesF"},
};
map<string,DataType> DataNameType = Util::reverse_map(DataTypeName);

const map<DataType,std::type_index>  DataTypeID = {
    {DataType::ArrayB, typeid(ArrayXb)},
    {DataType::ArrayI, typeid(ArrayXi)},
    {DataType::ArrayF, typeid(ArrayXf)},
    {DataType::MatrixB, typeid(ArrayXXb)},
    {DataType::MatrixI, typeid(ArrayXXi)},
    {DataType::MatrixF, typeid(ArrayXXf)},
    {DataType::TimeSeriesB, typeid(data::TimeSeriesb)},
    {DataType::TimeSeriesI,typeid(data::TimeSeriesi)},
    {DataType::TimeSeriesF, typeid(data::TimeSeriesf)},
};
map<std::type_index,DataType> DataIDType = Util::reverse_map(DataTypeID);

namespace data{

std::vector<std::type_index> StateTypes = {
                      typeid(ArrayXb),
                      typeid(ArrayXi), 
                      typeid(ArrayXf), 
                      typeid(ArrayXXb),
                      typeid(ArrayXXi), 
                      typeid(ArrayXXf), 
                      typeid(data::TimeSeriesb),
                      typeid(data::TimeSeriesi),
                      typeid(data::TimeSeriesf)
};

// /// returns the type_index held in arg
std::type_index StateType(const State& arg)
{
    return StateTypes.at(arg.index());
}
State check_type(const ArrayXf& x)
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
    
    State tmp; // = x;

    if (isBinary)
    {
        tmp = ArrayXb(x.cast<bool>());
    }
    else
    {
        if(isCategorical && uniqueMap.size() < 10)
        {
            tmp = ArrayXi(x.cast<int>());
        }
        else
        {
            tmp = x;
        }
    }
    return tmp;

}

/// return a slice of the data using indices idx
Data Data::operator()(const vector<size_t>& idx) const
{
    std::map<std::string, State> new_d;
    for (auto& [key, value] : this->features) 
    {
        std::visit([&](auto&& arg) 
        {
            // overloaded {
            //     [](auto arg){new_d[key] = T(arg(idx));};
            //     [](auto arg){new_d[key] = T(arg(idx));};
            // }
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ArrayXb> 
                          || std::is_same_v<T, ArrayXi> 
                          || std::is_same_v<T, ArrayXf> 
                          || std::is_same_v<T, TimeSeriesb> 
                          || std::is_same_v<T, TimeSeriesi> 
                          || std::is_same_v<T, TimeSeriesf> 
                         )
                new_d[key] = T(arg(idx));
            else if constexpr (std::is_same_v<T, ArrayXXb> 
                               || std::is_same_v<T, ArrayXXi> 
                               || std::is_same_v<T, ArrayXXf> 
                              )
                new_d[key] = T(arg(idx, Eigen::all));
            else 
                static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        value
        );
    }
    auto new_y = this->y(idx);
    return Data(new_d, new_y, this->classification);
}

Data Data::get_batch(int batch_size) const
{

    batch_size =  std::min(batch_size,int(this->n_samples));
    return (*this)(r.shuffled_index(n_samples));
}

array<Data, 2> Data::split(const ArrayXb& mask) const
{
    // split data into two based on mask. 
    auto idx1 = Util::mask_to_index(mask);
    auto idx2 = Util::mask_to_index((!mask));
    return std::array<Data, 2>{ (*this)(idx1), (*this)(idx2) };
}
/// call init at the end of constructors
/// to define metafeatures of the data.
void Data::init()
{
    //TODO: populate var_names, var_data_types, data_types, features_of_type
    n_features = this->features.size();
    // note this will have to change in unsupervised settings
    n_samples = this->y.size();


    for (const auto& [name, value]: this->features)
    {
        // save feature types
        std::type_index feature_type = StateType(value);

        /* Util::unique_insert(this->data_types, DataIDType[feature_type]); */
        this->data_types.push_back( DataIDType[feature_type]);
        // add feature to appropriate map list 
        this->features_of_type[feature_type].push_back(name);
    }
    // debug: print data
    // for (const auto& [var, arg] : this->features)
    // {
    //     std::cout << "feature: " << var << endl; 
    //     std::visit([&](const auto& a){ cout << a << "\n"; }, arg);
    // }
}

/// turns input data into a feature map
map<string, State> Data::make_features(const Ref<const ArrayXXf>& X,
                                       const map<string,State>& Z,
                                       const vector<string>& vn 
                                       ) 
{
    map<string, State> tmp_features;
    vector<string> var_names;
    // check variable names
    if (vn.empty())
    {
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            var_names.push_back(v);
        }
    }
    else
    {
        if (vn.size() != X.cols())
            HANDLE_ERROR_THROW(to_string(vn.size())
                                +"variable names and "
                                +to_string(X.cols())+" features");
    }

    for (int i = 0; i < X.cols(); ++i)
    {
        State tmp = check_type(X.col(i).array());

        tmp_features[var_names.at(i)] = tmp;
    }
    cout << "Data:: loaded data_types: ";
    // Util::print(data_types.begin(), data_types.end());
    tmp_features.insert(Z.begin(), Z.end());
    return tmp_features;
};

    
} // data
} // Brush
