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
        auto& k = key;
        std::visit([&](auto&& arg) 
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr ( T::NumDimensions == 1)
                new_features[k] = T(arg(idx));
            else if constexpr (T::NumDimensions==2)
                new_features[k] = T(arg(idx, Eigen::internal::all_t{}));
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
    Dataset result(new_features, new_y, this->classification);
    // Preserve the original feature name and type order from parent
    // The constructor's init() may have reordered them, so we fix that here
    result.feature_names = this->feature_names;
    result.feature_types = this->feature_types;
    result.feature_name_order_ = this->feature_names;
    
    // Rebuild features_of_type to match the corrected order
    result.features_of_type.clear();
    result.unique_data_types.clear();
    for (size_t i = 0; i < result.feature_names.size(); ++i) {
        const auto& name = result.feature_names[i];
        const auto& ftype = result.feature_types[i];
        result.features_of_type[ftype].push_back(name);
        Util::unique_insert(result.unique_data_types, ftype);
    }
    
    return result;
}


Dataset Dataset::get_batch() const
{
    if (!use_batch) 
        return (*this);

    const auto total_samples = static_cast<size_t>(this->get_n_samples());
    const auto target_batch_size = static_cast<size_t>(ceil(total_samples * batch_size));

    // Start a new epoch only after every row from the preceding epoch has
    // been returned. This prevents replacement before a complete pass through
    // the training partition.
    if (batch_cycle_idx.empty() || batch_cycle_batch_cursor >= batch_cycle_sizes.size())
    {
        // resets everything
        batch_cycle_idx.clear();
        batch_cycle_sizes.clear();
        batch_cycle_cursor = 0;
        batch_cycle_batch_cursor = 0;

        // repopulate the indexes
        if (classification)
        {
            // Shuffle each class queue. Every batch preserves the original
            // class prevalence and reserves one slot per class. Once a class
            // queue is exhausted, that class is sampled with replacement so
            // later batches still train against it while larger class queues
            // complete their no-replacement pass.
            std::map<float, vector<size_t>> class_indices;
            for (size_t i = 0; i < total_samples; ++i)
                class_indices[y(i)].push_back(i);

            for (auto& [label, indices] : class_indices)
                r.shuffle(indices.begin(), indices.end());

            // creating the batches
            vector<size_t> positions(class_indices.size(), 0); // this gets incremented everytime a row is added to a batch
            const auto class_count = class_indices.size();
            const auto current_batch_size = std::max(target_batch_size, class_count);
            while (true) // this while(true) iterates once for each batch creation
            {
                // searches for classes with unseen samples
                bool has_unseen_rows = false;
                size_t class_position = 0;
                for (const auto& [label, indices] : class_indices) {
                    has_unseen_rows |= positions[class_position] < indices.size();
                    ++class_position;
                }
                if (!has_unseen_rows) // all rows were added to a batch -- this is the escape condition to the while (true) loop
                    break;

                vector<size_t> counts(class_indices.size(), 0); // will count how many samples per class for the current batch
                vector<std::pair<float, size_t>> remainders; // will store how many samples are left per class after the current batch 
                
                // free space in the batch. Notice that one sample per class is already being considered here
                const auto allocatable_slots = current_batch_size - class_count;

                class_position = 0;
                for (const auto& [label, indices] : class_indices)
                {
                    counts[class_position] = 1; // this ensures one sample per class

                    // find rounded number of slots that will be used for the class
                    // to ensure they keep approximate prevalence w.r.t. original data
                    const float exact = static_cast<float>(indices.size() * allocatable_slots)
                                      / static_cast<float>(total_samples);
                    const auto floor_allocation = static_cast<size_t>(std::floor(exact));

                    counts[class_position] += floor_allocation;
                    remainders.emplace_back(exact - std::floor(exact), class_position);
                    ++class_position;
                }

                std::stable_sort(remainders.begin(), remainders.end(),
                    [](const auto& left, const auto& right) {
                        return left.first > right.first;
                    });

                size_t assigned = class_count;
                for (const auto count : counts)
                    assigned += count - 1;
                for (size_t i = 0; assigned < current_batch_size; ++i, ++assigned)
                    ++counts[remainders.at(i).second];

                // filling the batch
                vector<size_t> batch_indices;
                batch_indices.reserve(current_batch_size);
                class_position = 0;
                for (const auto& [label, indices] : class_indices)
                {
                    const auto available = indices.size() - positions[class_position];
                    const auto unique_count = std::min(counts[class_position], available);
                    for (size_t i = 0; i < unique_count; ++i)
                        batch_indices.push_back(indices[positions[class_position]++]);

                    // Repeat only a class that was already exhausted before
                    // this batch began. If its final unseen row was consumed
                    // above, return a smaller final batch instead of repeating
                    // that row immediately.
                    if (available == 0) {
                        for (size_t i = 0; i < counts[class_position]; ++i)
                            batch_indices.push_back(indices.at(r.rnd_int(0, indices.size() - 1)));
                    }
                    ++class_position;
                }
                r.shuffle(batch_indices.begin(), batch_indices.end());
                batch_cycle_idx.insert(batch_cycle_idx.end(), batch_indices.begin(), batch_indices.end());
                batch_cycle_sizes.push_back(batch_indices.size());
            }
        }
        else
        {
            // Regression batches are stratified by target rank. The four
            // equal-frequency y quartiles are shuffled independently, then
            // apportioned evenly into each batch without replacement.
            vector<size_t> sorted_indices(total_samples);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                [this](size_t left, size_t right) { return y(left) < y(right); });

            vector<vector<size_t>> quartiles(4);
            for (size_t rank = 0; rank < total_samples; ++rank)
                quartiles.at((rank * quartiles.size()) / total_samples).push_back(sorted_indices.at(rank));

            size_t nonempty_quartiles = 0;
            for (auto& quartile : quartiles) {
                if (!quartile.empty()) {
                    r.shuffle(quartile.begin(), quartile.end());
                    ++nonempty_quartiles;
                }
            }

            vector<size_t> positions(quartiles.size(), 0);
            const auto current_batch_size = std::max(target_batch_size, nonempty_quartiles);
            while (true)
            {
                bool has_unseen_rows = false;
                for (size_t quartile = 0; quartile < quartiles.size(); ++quartile)
                    has_unseen_rows |= positions[quartile] < quartiles[quartile].size();
                if (!has_unseen_rows)
                    break;

                vector<size_t> counts(quartiles.size(), 0);
                vector<std::pair<float, size_t>> remainders;
                const auto allocatable_slots = current_batch_size - nonempty_quartiles;
                for (size_t quartile = 0; quartile < quartiles.size(); ++quartile) {
                    if (quartiles[quartile].empty())
                        continue;
                    counts[quartile] = 1;
                    const float exact = static_cast<float>(quartiles[quartile].size() * allocatable_slots)
                                      / static_cast<float>(total_samples);
                    counts[quartile] += static_cast<size_t>(std::floor(exact));
                    remainders.emplace_back(exact - std::floor(exact), quartile);
                }

                std::stable_sort(remainders.begin(), remainders.end(),
                    [](const auto& left, const auto& right) {
                        return left.first > right.first;
                    });
                size_t assigned = 0;
                for (const auto count : counts)
                    assigned += count;
                for (size_t i = 0; assigned < current_batch_size; ++i, ++assigned)
                    ++counts[remainders.at(i).second];

                vector<size_t> batch_indices;
                batch_indices.reserve(current_batch_size);
                for (size_t quartile = 0; quartile < quartiles.size(); ++quartile) {
                    const auto available = quartiles[quartile].size() - positions[quartile];
                    const auto unique_count = std::min(counts[quartile], available);
                    for (size_t i = 0; i < unique_count; ++i)
                        batch_indices.push_back(quartiles[quartile][positions[quartile]++]);
                }
                r.shuffle(batch_indices.begin(), batch_indices.end());
                batch_cycle_idx.insert(batch_cycle_idx.end(), batch_indices.begin(), batch_indices.end());
                batch_cycle_sizes.push_back(batch_indices.size());
            }
        }
    }

    // get the indices and move the two cursors
    const auto count = batch_cycle_sizes.at(batch_cycle_batch_cursor++);
    vector<size_t> indices(batch_cycle_idx.begin() + batch_cycle_cursor,
                           batch_cycle_idx.begin() + batch_cycle_cursor + count);
    batch_cycle_cursor += count;
    return (*this)(indices);
}

std::array<Dataset, 2> Dataset::split(const ArrayXb& mask) const
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

vector<string> Dataset::get_feature_types() const {
    // iterate through each feature name in order, get the data type, and return it. This is
    // used in the python front-end to save the feature types from the training dataset
    // when calling predict. 

    vector<string> python_feature_types;
    // Iterate through feature_names to preserve order, not through features map
    for (const auto& name: this->feature_names)
    {
        // fmt::print("name:{}\n",name);
        const auto& value = this->features.at(name);
        
        // save feature types
        auto feature_type = StateType(value);

        if (feature_type == DataType::ArrayB)
            python_feature_types.push_back("ArrayB");
        else if (feature_type == DataType::ArrayI)
            python_feature_types.push_back("ArrayI");
        else if (feature_type == DataType::ArrayF)
            python_feature_types.push_back("ArrayF");
        else
            HANDLE_ERROR_THROW(
                "get_feature_type does not support the type of this feature yet: " + name + 
                "as a notice, this function is suposed to be used in the python side, to extract data types inferred by Brush type sniffer.");
    }

    return python_feature_types;
}


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
    // Use the stored original feature name order if available, otherwise iterate through the map.
    // IMPORTANT: use a value (not a const-ref) to avoid a dangling reference from binding
    // a const-ref to the temporary produced by a mixed lvalue/prvalue ternary expression.
    vector<string> names_to_use;
    if (this->feature_name_order_.empty()) {
        for (const auto& [name, value] : this->features) {
            names_to_use.push_back(name);
        }
    } else {
        names_to_use = this->feature_name_order_;
    }

    for (const auto& name : names_to_use)
    {
        // fmt::print("name:{}\n",name);
        const auto& value = this->features.at(name);
        
        // save feature types
        auto feature_type = StateType(value);

        Util::unique_insert(unique_data_types, feature_type);
        feature_types.push_back( feature_type );

        // add feature to appropriate map list 
        this->features_of_type[feature_type].push_back(name);

        // populate feature names in the original order
        this->feature_names.push_back(name);
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
    else if (classification) // figuring out training and validation data indexes
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
            
            // stratified split so train/validation never overlap when a class is
            // too small. Now, if a class has no remaining samples for validation,
            // it contributes only to training (validation gets none for that class).
            if (n_class_samples - n_train_samples > 0)
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

        if (shuffle_split)
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
    batch_cycle_idx.clear();
    batch_cycle_sizes.clear();
    batch_cycle_cursor = 0;
    batch_cycle_batch_cursor = 0;
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
    vector<string> tmp_feature_names = {};
    if (vn.empty())
    {
        // fmt::print("vn empty\n");
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            tmp_feature_names.push_back(v);
        }
    }
    else
    {
        if (vn.size() != X.cols())
            HANDLE_ERROR_THROW(
                fmt::format("Variable names and data size mismatch: "
                "{} variable names and {} features in X", 
                vn.size(), X.cols()) );
        tmp_feature_names = vn;
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
        // fmt::print("X({}): {} \n",i,tmp_feature_names.at(i));
        State tmp = check_type(X.col(i).array(), var_types.at(i));

        tmp_features[tmp_feature_names.at(i)] = tmp;
    }
    // fmt::print("tmp_features insert\n");
    tmp_features.insert(Z.begin(), Z.end());

    // Store the original feature name order (X features first, then Z features)
    this->feature_name_order_ = tmp_feature_names;
    for (const auto& [name, value] : Z) {
        this->feature_name_order_.push_back(name);
    }

    return tmp_features;
};

/// turns input into a feature map, with feature types copied from a reference
map<string,State> Dataset::copy_and_make_features(const ArrayXXf& X,
                                         const Dataset& ref_dataset,
                                         const vector<string>& vn
                                        )
{
    vector<string> tmp_feature_names = {};
    if (vn.empty())
    {
        for (int i = 0; i < X.cols(); ++i)
        {
            string v = "x_"+to_string(i);
            tmp_feature_names.push_back(v);
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
        tmp_feature_names = vn;
    }

    if (ref_dataset.features.size() != tmp_feature_names.size())
        HANDLE_ERROR_THROW(
            fmt::format("Reference dataset with incompatible number of variables: "
            "Reference has {} variable names, but X has {}", 
            ref_dataset.features.size(), 
            tmp_feature_names.size()
            )
        );

    map<string, State> tmp_features;
    for (int i = 0; i < X.cols(); ++i)
    {
        State tmp = cast_type(
            X.col(i).array(),
            ref_dataset.features.at(tmp_feature_names.at(i))
        );

        tmp_features[tmp_feature_names.at(i)] = tmp;
    }

    // Store the original feature name order
    this->feature_name_order_ = tmp_feature_names;

    return tmp_features;
};

ostream& operator<<(ostream& os, DataType dt)
{
    os << DataTypeName[dt];
    return os;
}
    
} // data
} // Brush
