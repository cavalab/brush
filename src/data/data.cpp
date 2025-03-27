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

// we have 3 basic types (bool, integer, float), specialized into
// arrays, matrices, and timeseries. Notice that all dataset and operators
// right now only work with arrays. TODO: implement timeseries and matrices.
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
State check_type(const ArrayXf& x, const string t)
{
    State tmp;

    if (!t.empty())
    {
        // Use DataNameType to get the statetype given the string representation
        DataType feature_type = DataNameType.at(t);
            
        if (feature_type == DataType::ArrayB)
            tmp = ArrayXb(x.cast<bool>());
        else if (feature_type == DataType::ArrayI)
            tmp = ArrayXi(x.cast<int>());
        else if (feature_type == DataType::ArrayF)
            tmp = ArrayXf(x.cast<float>());
        else
            HANDLE_ERROR_THROW(
                "Invalid feature type. check_type does not support this type: " + t);
    }
    else
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
    }

    return tmp;
}

template<typename StateRef>
State cast_type(const ArrayXf& x, const StateRef& x_ref)
{
    if (std::holds_alternative<ArrayXi>(x_ref))
        return ArrayXi(x.cast<int>());
    else if (std::holds_alternative<ArrayXb>(x_ref))
        return ArrayXb(x.cast<bool>());
    
    return x;
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
    // using constructor 1
    return Dataset(new_features, new_y, this->classification);
}


// TODO: i need to improve how get batch works. Maybe a function to update batch indexes, and always using the same dataset?
// TODO: also, i need to make sure the get batch will sample only from training data and not test
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
    //TODO: populate feature_names, var_data_types, data_types, features_of_type
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
        feature_types.push_back( feature_type );

        // add feature to appropriate map list 
        this->features_of_type[feature_type].push_back(name);
    }

    // setting the training and validation data indexes
    auto n_samples = int(this->get_n_samples());

    training_data_idx.resize(0);
    validation_data_idx.resize(0);

    if (!use_validation)
    {
        vector<size_t> idx(n_samples);

        std::iota(idx.begin(), idx.end(), 0);
        
        std::transform(idx.begin(), idx.end(),
                back_inserter(training_data_idx),
                [&](int element) { return element; });
        
        std::transform(idx.begin(), idx.end(),
                back_inserter(validation_data_idx),
                [&](int element) { return element; });
    }
    else if (classification && true) // figuring out training and validation data indexes
    { // Stratified split for classification problems. TODO: parameters to change stratify behavior? (and set false by default)
        std::map<float, vector<int>> class_indices; // TODO: I think I can remove many std:: from the code..
        for (size_t i = 0; i < n_samples; ++i) {
            class_indices[y[i]].push_back(i);
        }

        for (auto& class_group : class_indices) {
            auto& indices = class_group.second;

            int n_class_samples = indices.size();
            
            vector<size_t> idx(n_class_samples);
            if (shuffle_split)
                idx = r.shuffled_index(n_class_samples);
            else
                std::iota(idx.begin(), idx.end(), 0);

            auto n_train_samples = int(ceil(n_class_samples*(1.0-validation_size)));

            std::transform(idx.begin(), idx.begin() + n_train_samples,
                    back_inserter(training_data_idx),
                    [&](int element) { return indices[element]; });
                    
            if (n_class_samples - n_train_samples == 0)
            {
                // same indices from the training data to the validation data
                std::transform(idx.begin(), idx.begin() + n_train_samples,
                        back_inserter(validation_data_idx),
                        [&](int element) { return indices[element]; });
            }
            else 
            {
                std::transform(idx.begin() + n_train_samples, idx.end(),
                        back_inserter(validation_data_idx),
                        [&](int element) { return indices[element]; });
            }
        }
    }
    else { // regression, or classification without stratification
        // logic for non-classification problems
        vector<size_t> idx(n_samples);

        if (shuffle_split) // TODO: make sure this works with multiple threads and fixed random state
            idx = r.shuffled_index(n_samples);
        else
            std::iota(idx.begin(), idx.end(), 0);
            
        // garantee that at least one sample is going to be returned, since
        // use_batch is true only if batch_size is (0, 1), and ceil will round
        // up
        auto n_train_samples = int(ceil(n_samples*(1-validation_size)));

        std::transform(idx.begin(), idx.begin() + n_train_samples,
                back_inserter(training_data_idx),
                [&](int element) { return element; });

        if (n_samples - n_train_samples == 0) { // training_data_idx contains all data
            validation_data_idx = training_data_idx;
        }
        else 
        {
            std::transform(idx.begin() + n_train_samples, idx.end(),
                    back_inserter(validation_data_idx),
                    [&](int element) { return element; });
        }   
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
                                          const vector<string>& vn,
                                          const vector<string>& ft
                                         ) 
{
    // fmt::print("Dataset::make_features()\n");
    map<string, State> tmp_features;

    // fmt::print("vn: {}\n",vn);

    // check variable names
    feature_names.resize(0);
    if (vn.empty())
    {
        // fmt::print("vn empty\n");
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            feature_names.push_back(v);
        }
    }
    else
    {
        if (vn.size() != X.cols())
            HANDLE_ERROR_THROW(
                fmt::format("Variable names and data size mismatch: "
                "{} variable names and {} features in X", 
                vn.size(), X.cols()) );
        feature_names = vn;
    }

    // check variable types
    vector<string> var_types;
    if (ft.empty())
    {
        for (int i = 0; i < X.cols(); ++i)
        {
            var_types.push_back("");
        }
    }
    else {
    if (ft.size() != X.cols())
        HANDLE_ERROR_THROW(
            fmt::format("Feature type names and data size mismatch: "
            "{} feature type names and {} features in X", 
            ft.size(),  X.cols()) );
        var_types = ft;
    }

    for (int i = 0; i < X.cols(); ++i)
    {
        // fmt::print("X({}): {} \n",i,feature_names.at(i));
        State tmp = check_type(X.col(i).array(), var_types.at(i));

        tmp_features[feature_names.at(i)] = tmp;
    }
    // fmt::print("tmp_features insert\n");
    tmp_features.insert(Z.begin(), Z.end());

    return tmp_features;
};

/// turns input into a feature map, with feature types copied from a reference
map<string,State> Dataset::copy_and_make_features(const ArrayXXf& X,
                                         const Dataset& ref_dataset,
                                         const vector<string>& vn
                                        )
{
    feature_names.resize(0);
    if (vn.empty())
    {
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            feature_names.push_back(v);
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
        feature_names = vn;
    }

    if (ref_dataset.features.size() != feature_names.size())
        HANDLE_ERROR_THROW(
            fmt::format("Reference dataset with incompatible number of variables: "
            "Reference has {} variable names, but X has {}", 
            ref_dataset.features.size(), 
            feature_names.size()
            )
        );

    map<string, State> tmp_features;
    for (int i = 0; i < X.cols(); ++i)
    {
        State tmp = cast_type(
            X.col(i).array(),
            ref_dataset.features.at(feature_names.at(i))
        );

        tmp_features[feature_names.at(i)] = tmp;
    }

    return tmp_features;
};

ostream& operator<<(ostream& os, DataType dt)
{
    os << DataTypeName[dt];
    return os;
}
    
} // data
} // Brush
