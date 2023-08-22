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
    {DataType::ArrayBJet, "ArrayBJet"},
    {DataType::ArrayIJet, "ArrayIJet"},
    {DataType::ArrayFJet, "ArrayFJet"},
    {DataType::MatrixBJet, "MatrixBJet"},
    {DataType::MatrixIJet, "MatrixIJet"},
    {DataType::MatrixFJet, "MatrixFJet"},
    {DataType::TimeSeriesBJet, "TimeSeriesBJet"},
    {DataType::TimeSeriesIJet,"TimeSeriesIJet"},
    {DataType::TimeSeriesFJet, "TimeSeriesFJet"}
};
map<string,DataType> DataNameType = Util::reverse_map(DataTypeName);

const map<DataType,std::type_index>  DataTypeID = {
    {DataType::ArrayB, typeid(ArrayXb)},
    {DataType::ArrayI, typeid(ArrayXi)},
    {DataType::ArrayF, typeid(ArrayXf)},
    {DataType::MatrixB, typeid(ArrayXXb)},
    {DataType::MatrixI, typeid(ArrayXXi)},
    {DataType::MatrixF, typeid(ArrayXXf)},
    {DataType::TimeSeriesB, typeid(Data::TimeSeriesb)},
    {DataType::TimeSeriesI,typeid(Data::TimeSeriesi)},
    {DataType::TimeSeriesF, typeid(Data::TimeSeriesf)},
};
map<std::type_index,DataType> DataIDType = Util::reverse_map(DataTypeID);

namespace Data{

std::vector<DataType> StateTypes = {
    DataType::ArrayB,
    DataType::ArrayI, 
    DataType::ArrayF, 
    DataType::MatrixB,
    DataType::MatrixI, 
    DataType::MatrixF, 
    DataType::TimeSeriesB,
    DataType::TimeSeriesI,
    DataType::TimeSeriesF
};

// /// returns the type_index held in arg
DataType StateType(const State& arg)
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
        if(isCategorical && uniqueMap.size() <= 10)
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
Dataset Dataset::operator()(const vector<size_t>& idx) const
{
    std::map<std::string, State> new_features;
    for (auto& [key, value] : this->features) 
    {
        std::visit([&](auto&& arg) 
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr ( T::NumDimensions == 1)
                new_features[key] = T(arg(idx));
            else if constexpr (T::NumDimensions==2)
                new_features[key] = T(arg(idx, Eigen::all));
            else 
                static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        value
        );
    }
    ArrayXf new_y;
    if (this->y.size()>0)
    {
        new_y = this->y(idx);
    }
    return Dataset(new_features, new_y, this->classification);
}

Dataset Dataset::get_batch() const
{
    // will always return a new dataset, even when use_batch is false (this case, returns itself)

    if (!use_batch) 
        return (*this);

    auto n_samples = int(this->get_n_samples());
    // garantee that at least one sample is going to be returned, since
    // use_batch is true only if batch_size is (0, 1), and ceil will round
    // up
    n_samples = int(ceil(n_samples*batch_size));

    return (*this)(r.shuffled_index(n_samples));
}

array<Dataset, 2> Dataset::split(const ArrayXb& mask) const
{
    // TODO: assert that mask is not filled with zeros or ones (would create
    // one empty partition)

    // split data into two based on mask. 
    auto idx1 = Util::mask_to_index(mask);
    auto idx2 = Util::mask_to_index((!mask));
    return std::array<Dataset, 2>{ (*this)(idx1), (*this)(idx2) };
}

Dataset Dataset::get_training_data() const { return (*this)(training_data_idx); }
Dataset Dataset::get_validation_data() const { return (*this)(validation_data_idx); }

/// call init at the end of constructors
/// to define metafeatures of the data.
void Dataset::init()
{
    //TODO: populate var_names, var_data_types, data_types, features_of_type
    // n_features = this->features.size();
    // note this will have to change in unsupervised settings
    // n_samples = this->y.size();

    if (this->features.size() == 0){
        HANDLE_ERROR_THROW(
            fmt::format("Error during the initialization of the dataset. It "
                        "does not contain any data\n") 
            );
    }

    // fmt::print("Dataset::init()\n");
    for (const auto& [name, value]: this->features)
    {
        // fmt::print("name:{}\n",name);
        // save feature types
        auto feature_type = StateType(value);

        Util::unique_insert(unique_data_types, feature_type);
        feature_types.push_back( feature_type);
        // add feature to appropriate map list 
        this->features_of_type[feature_type].push_back(name);
    }

    // setting the training and validation data indexes
    auto n_samples = int(this->get_n_samples());
    auto idx = r.shuffled_index(n_samples);

    // garantee that at least one sample is going to be returned, since
    // use_batch is true only if batch_size is (0, 1), and ceil will round
    // up
    auto n_train_samples = int(ceil(n_samples*(1-validation_size)));

    training_data_idx.resize(0);
    std::transform(idx.begin(), idx.begin() + n_train_samples,
            back_inserter(training_data_idx),
            [&](int element) { return element; });

    if ( use_validation && (n_samples - n_train_samples != 0) ) {
        validation_data_idx.resize(0);
        std::transform(idx.begin() + n_train_samples, idx.end(),
                back_inserter(validation_data_idx),
                [&](int element) { return element; });
    }
    else {
        validation_data_idx = training_data_idx;
    } 
}

float Dataset::get_batch_size() { return batch_size; }
void Dataset::set_batch_size(float new_size) {
    batch_size = new_size;
    use_batch = batch_size > 0.0 && batch_size < 1.0;
}

/// turns input data into a feature map
map<string, State> Dataset::make_features(const ArrayXXf& X,
                                       const map<string,State>& Z,
                                       const vector<string>& vn 
                                       ) 
{
    // fmt::print("Dataset::make_features()\n");
    map<string, State> tmp_features;
    vector<string> var_names;
    // fmt::print("vn: {}\n",vn);
    // check variable names
    if (vn.empty())
    {
        // fmt::print("vn empty\n");
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            var_names.push_back(v);
        }
    }
    else
    {
        if (vn.size() != X.cols())
            HANDLE_ERROR_THROW(
                fmt::format("Variable names and data size mismatch: "
                "{} variable names and {} features in X", 
                vn.size(), 
                X.cols()
                )
            );
        var_names = vn;
    }

    for (int i = 0; i < X.cols(); ++i)
    {
        // fmt::print("X({}): {} \n",i,var_names.at(i));
        State tmp = check_type(X.col(i).array());

        tmp_features[var_names.at(i)] = tmp;
    }
    // fmt::print("tmp_features insert\n");
    tmp_features.insert(Z.begin(), Z.end());
    return tmp_features;
};

ostream& operator<<(ostream& os, DataType dt)
{
    os << DataTypeName[dt];
    return os;
}
    
} // data
} // Brush
