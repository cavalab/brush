/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

//internal includes
#include "data.h"

using namespace Brush::Util;
using std::min;

namespace Brush{ 
namespace data{

std::vector<std::type_index> StateTypeMap = {
                      typeid(ArrayXb),
                      typeid(ArrayXi), 
                      typeid(ArrayXf), 
                      typeid(ArrayXXb),
                      typeid(ArrayXXi), 
                      typeid(ArrayXXf), 
                      typeid(TimeSeries)
};
// /// returns the typeid held in arg
std::type_index StateType(const State& arg)
{
    return StateTypeMap.at(arg.index());
}
State check_type(const ArrayXf& x)
{
    //TODO: make this use a variant and over-loading, or something. 
    // Eigen doesn't like the variable output casting. 

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
    // return isBinary ? std::get<ArrayXb>(tmp)
    //                  : (isCategorical && uniqueMap.size() < 10) ? 
    //                     std::get<ArrayXi>(tmp) 
    //                     :  std::get<ArrayXf>(tmp);
    // return isBinary ? x.cast<bool>()
    //                  : (isCategorical && uniqueMap.size() < 10) ? 
    //                     x.cast<int>() 
    //                     : x.cast<float>();
    if (isBinary)
    {
        tmp = ArrayXb(x.cast<bool>());
        // return std::get<ArrayXb>(tmp);
    }
    else
    {
        if(isCategorical && uniqueMap.size() < 10)
        {
            tmp = ArrayXi(x.cast<int>());
            // return std::get<ArrayXi>(tmp);
        }
        else
        {
            tmp = x;
            // return std::get<ArrayXf>(tmp);
        }
    }
    return tmp;

}
// Data::Data(MatrixXf& X, ArrayXf& y, Longitudinal& Z, 
//             const vector<string>& variable_names, 
//             bool c): X(X), y(y), Z(Z), var_names(variable_names), classification(c) 
// {
//     validation=false;
//     data_types = get_dtypes(X);
//     // ret_type = typeid(y);

//     if (var_names.empty())
//     {
//         vector<int> varnum(X.cols());
//         iota(varnum.begin(), varnum.end(), 0);
//         for (const auto& v : varnum)
//             var_names.push_back("x_"+to_string(v));
//     }
//     assert(X.cols() == var_names.size());

//     for (int i = 0; i< X.cols(); ++i)
//     {
//         name_to_idx[var_names.at(i)] = i;
//     }
// }
// void Data::shuffle()
// {
//     *this = (*this)(r.shuffled_index(n_samples));
// }
/// return a slice of the data using indices idx
Data Data::operator()(const vector<size_t>& idx) const
{
    std::map<std::string, State> new_d;
    for (auto& [key, value] : this->features) 
    {
        std::visit([&](auto&& arg) 
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ArrayXb> 
                          || std::is_same_v<T, ArrayXi> 
                          || std::is_same_v<T, ArrayXf> 
                          || std::is_same_v<T, TimeSeries> 
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
    // Data batch;
    // batch.X.resize(batch_size, X.cols());
    // batch.y.resize(batch_size);
    // batch.Z.time.resize(batch_size);
    // batch.Z.value.resize(batch_size);

    // for (unsigned i = 0; i<batch_size; ++i)
    // {
        
    //     batch.X.col(i) = X.col(idx.at(i)); 
    //     batch.y(i) = y(idx.at(i)); 
    //     batch.Z.time.row(i) = Z.time.row(idx.at(i));
    //     batch.Z.value.row(i) = Z.value.row(idx.at(i));

    // }
}
// array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const
// {

// }
array<Data, 2> Data::split(const ArrayXb& mask) const
{
    // split data into two based on mask. 
    auto idx1 = Util::mask_to_index(mask);
    auto idx2 = Util::mask_to_index((!mask));
    return std::array<Data, 2>{ (*this)(idx1), (*this)(idx2) };
}

// CVData::CVData()
// {
//     oCreated = false;
//     tCreated = false;
//     vCreated = false;
// }

// CVData::CVData(MatrixXf& X, ArrayXf& y, Longitudinal & Z, 
//                 const vector<string>& variable_names,
//                 bool c)
// {
//     o = new Data(X, y, Z, variable_names, c);
//     oCreated = true;
    
//     t = new Data(X_t, y_t, Z_t, variable_names, c);
//     tCreated = true;
    
//     v = new Data(X_v, y_v, Z_v, variable_names, c);
//     vCreated = true;
    
//     classification = c;
    
//     // split data into training and test sets
//     //train_test_split(params.shuffle, params.split);
// }

// CVData::~CVData()
// {
//     if(o != NULL && oCreated)
//     {
//         delete(o);
//         o = NULL;
//     }
    
//     if(t != NULL && tCreated)
//     {
//         delete(t);
//         t = NULL;
//     }
    
//     if(v != NULL && vCreated)
//     {
//         delete(v);
//         v = NULL;
//     }
// }

// void CVData::setOriginalData(MatrixXf& X, ArrayXf& y, Longitudinal& Z,
//                                 const vector<string>& variable_names,
//                                 bool c)
// {
//     o = new Data(X, y, Z, variable_names, c);
//     oCreated = true;
    
//     t = new Data(X_t, y_t, Z_t, variable_names, c);
//     tCreated = true;
    
//     v = new Data(X_v, y_v, Z_v, variable_names, c);
//     vCreated = true;
    
//     classification = c;
// }

// void CVData::setOriginalData(Data *d)
// {
//     o = d;
//     oCreated = false;
    
//     t = new Data(X_t, y_t, Z_t, d->var_names, d->classification);
//     tCreated = true;
    
//     v = new Data(X_v, y_v, Z_v, d->var_names, d->classification);
//     vCreated = true;
    
//     classification = d->classification;
// }

// void CVData::setTrainingData(MatrixXf& X_t, ArrayXf& y_t, Longitudinal& Z_t,
//                                 const vector<string>& variable_names,
//                                 bool c)
// {
//     t = new Data(X_t, y_t, Z_t, variable_names, c);
//     tCreated = true;
    
//     classification = c;
// }

// void CVData::setTrainingData(Data *d, bool toDelete)
// {
//     t = d;
//     if(!toDelete)
//         tCreated = false;
//     else
//         tCreated = true;
// }

// void CVData::setValidationData(MatrixXf& X_v, ArrayXf& y_v, 
//                                 Longitudinal& Z_v,
//                                 const vector<string>& variable_names,
//                                 bool c)
// {
//     v = new Data(X_v, y_v, Z_v, variable_names,  c);
//     vCreated = true;
// }

// void CVData::setValidationData(Data *d)
// {
//     v = d;
//     vCreated = false;
// }

// void CVData::shuffle_data()
// {
//     Eigen::PermutationMatrix<Dynamic,Dynamic> perm(o->X.cols());
    
//     o.shuffle()
//     perm.setIdentity();
//     r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
//     /* cout << "X before shuffle: \n"; */
//     /* cout << o->X.transpose() << "\n"; */
//     o->X = o->X * perm;       // shuffles columns of X

//     /* cout << "X after shuffle: \n"; */
//     /* cout << o->X.transpose() << "\n"; */
//     o->y = (o->y.matrix().transpose() * perm).transpose().array() ;       // shuffle y too
    
//     if(o->Z.size() > 0)
//     {
//         std::vector<int> zidx(o->y.size());
//         // zidx maps the perm_indices values to their indices, i.e. the inverse transform
//         for (unsigned i = 0; i < perm.indices().size(); ++i)
//             zidx.at(perm.indices()(i)) = i;
//         /* cout << "zidx :\n"; */
//         /* for (const auto& zi : zidx) */
//         /*     cout << zi << "," ; */
//         /* cout << "\n"; */
//         reorder(o->Z, zidx);
//     }
// }

// void CVData::split_stratified(float split)
// {
//     logger.log("Stratify split called with initial data size as " 
//             + std::to_string(o->X.cols()), 3);
                    
//     std::map<float, vector<int>> label_indices;
        
//     //getting indices for all labels
//     for(int x = 0; x < o->y.size(); x++)
//         label_indices[o->y[x]].push_back(x);
            
//     std::map<float, vector<int>>::iterator it = label_indices.begin();
    
//     vector<int> t_indices;
//     vector<int> v_indices;
    
//     int t_size;
//     int x;
    
//     for(; it != label_indices.end(); it++)
//     {
//         t_size = ceil(it->second.size()*split);
        
//         for(x = 0; x < t_size; x++)
//             t_indices.push_back(it->second[x]);
            
//         for(; x < it->second.size(); x++)
//             v_indices.push_back(it->second[x]);
        
//         logger.log("Label is " + to_string(it->first), 3, "\t");
//         logger.log("Total size = " + to_string(it->second.size()), 3, 
//                 "\t");
//         logger.log("training_size = " + to_string(t_size), 3, "\t");
//         logger.log("verification size = " 
//                 + to_string(it->second.size() - t_size), 3, "\t");
        
//     }
    
//     X_t.resize(o->X.cols(), t_indices.size());
//     X_v.resize(o->X.cols(), v_indices.size());
//     y_t.resize(t_indices.size());
//     y_v.resize(v_indices.size());
    
//     sort(t_indices.begin(), t_indices.end());
    
//     for(int x = 0; x < t_indices.size(); x++)
//     {
//         t->X.col(x) = o->X.col(t_indices[x]);
//         t->y[x] = o->y[t_indices[x]];
//         t->Z[x] = o->Z[t_indices[x]];
//     }
    
//     sort(v_indices.begin(), v_indices.end());
    
//     for(int x = 0; x < v_indices.size(); x++)
//     {
//         v->X.col(x) = o->X.col(v_indices[x]);
//         v->y[x] = o->y[v_indices[x]];
//         v->Z[x] = o->Z[t_indices[x]];
//     }

    
// }

// void CVData::train_test_split(bool shuffle, float split)
// {
//     /* @param X: n_features x n_samples matrix of training data
//         * @param y: n_samples vector of training labels
//         * @param shuffle: whether or not to shuffle X and y
//         * @param[out] X_t, X_v, y_t, y_v: training and validation matrices
//         */
        
                                
//     if (shuffle)     // generate shuffle index for the split
//         o->shuffle();
        
//     if(classification)
//         split_stratified(split);
//     else
//     {        
//         // resize training and test sets
//         X_t.resize(o->X.cols(),int(o->X.cols()*split));
//         X_v.resize(o->X.cols(),int(o->X.cols()*(1-split)));
//         y_t.resize(int(o->y.size()*split));
//         y_v.resize(int(o->y.size()*(1-split)));
        
//         // map training and test sets  
//         t->X = MatrixXf::Map(o->X.data(),t->X.cols(),t->X.cols());
//         v->X = MatrixXf::Map(o->X.data()+t->X.cols()*t->X.cols(),
//                                     v->X.cols(),v->X.cols());

//         t->y = ArrayXf::Map(o->y.data(),t->y.size());
//         v->y = ArrayXf::Map(o->y.data()+t->y.size(),v->y.size());
//         if(o->Z.size() > 0)
//             split_longitudinal(o->Z, t->Z, v->Z, split);
//     }

// }  

// void CVData::split_longitudinal(
//                         Longitudinal& Z,
//                         Longitudinal& Z_t,
//                         Longitudinal& Z_v,
//                         float split)
// {

    
//     /* for ( const auto val: Z ) */
//     /* { */
//     /*     size = Z[val.first].first.size(); */
//     /*     break; */
//     /* } */
    
//     int trainSize = int(Z.size()*split);
//     int validateSize = int(Z.size()*(1-split));

//     Z_t.assign(Z.begin(), Z.begin() + trainSize);
//     Z_v.assign(Z.begin()+trainSize, Z.begin() + trainSize+validateSize);
        
//     /* for ( const auto &val: Z ) */
//     /* { */
//     /*     vector<ArrayXf> _Z_t_v, _Z_t_t, _Z_v_v, _Z_v_t; */
//     /*     _Z_t_v.assign(Z[val.first].first.begin(), */ 
//     /*             Z[val.first].first.begin()+testSize); */
//     /*     _Z_t_t.assign(Z[val.first].second.begin(), */ 
//     /*             Z[val.first].second.begin()+testSize); */
//     /*     _Z_v_v.assign(Z[val.first].first.begin()+testSize, */ 
//     /*                   Z[val.first].first.begin()+testSize+validateSize); */
//     /*     _Z_v_t.assign(Z[val.first].second.begin()+testSize, */ 
//     /*                   Z[val.first].second.begin()+testSize+validateSize); */
        
//     /*     Z_t[val.first] = make_pair(_Z_t_v, _Z_t_t); */
//     /*     Z_v[val.first] = make_pair(_Z_v_v, _Z_v_t); */
//     /* } */
// }
    
    
} // data
} // Brush
