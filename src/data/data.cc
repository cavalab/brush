/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

//internal includes
#include "data.h"

using namespace Brush::Util;

namespace Brush{ 
    namespace data{

Data::Data(MatrixXf& X, ArrayXf& y, Longitudinal& Z, 
            const vector<string>& variable_names, 
            bool c): X(X), y(y), Z(Z), var_names(variable_names), classification(c) 
{
    validation=false;
    data_types = get_dtypes(X);
    // ret_type = typeid(y);

    if (var_names.empty())
    {
        vector<int> varnum(X.cols());
        iota(varnum.begin(), varnum.end(), 0);
        for (const auto& v : varnum)
            var_names.push_back("x_"+to_string(v));
    }
    assert(X.cols() == var_names.size());

    for (int i = 0; i< X.cols(); ++i)
    {
        name_to_idx[var_names.at(i)] = i;
    }
}

void Data::set_validation(bool v){validation=v;}

Data Data::get_batch(int batch_size) const
{

    batch_size =  std::min(batch_size,int(y.size()));
    vector<size_t> idx(y.size());
    std::iota(idx.begin(), idx.end(), 0);
//        r.shuffle(idx.begin(), idx.end());
    Data batch;
    batch.X.resize(batch_size, X.cols());
    batch.y.resize(batch_size);
    batch.Z.time.resize(batch_size);
    batch.Z.value.resize(batch_size);

    for (unsigned i = 0; i<batch_size; ++i)
    {
        
        batch.X.col(i) = X.col(idx.at(i)); 
        batch.y(i) = y(idx.at(i)); 
        batch.Z.time.row(i) = Z.time.row(idx.at(i));
        batch.Z.value.row(i) = Z.value.row(idx.at(i));

    }
}
// array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const
// {

// }
array<Data, 2> Data::split(const ArrayXb& mask) const
{
    // split data into two based on mask. 
    int size1 = mask.count();
    int size2 = mask.size() - size1;
    MatrixXf X1(size1, X.cols()), X2(size2, X.cols());
    ArrayXf y1(size1), y2(size2);
    Longitudinal Z1, Z2; 

    tie( X1, X2 ) = split(X, mask);
    tie( y1, y2 ) = split(y, mask);


    // int idx1 = 0, idx2 = 0;

    // for (int  i = 0; i < mask.size(); ++i)
    // {
    //     if (mask(i))
    //     {
    //         X1.row(idx1) = X.row(i);
    //         y1(idx1) = y(i);
    //         ++idx1;
    //     }
    //     else
    //     {
    //         X2.row(idx2) = X.row(i);
    //         y2(idx2) = y(i);
    //         ++idx2;
    //     }
    // }
    // split longitudinal variables
    // for (auto& el: Z.items())
    // {
    //     string& key = el.key();
    //     TimeSeries& feature = el.value();

    //     Z1[el.key()] = TimeSeries(size1);
    //     Z2[el.key()] = TimeSeries(size2);

    //     tie( Z1[el.key()].value, Z2[el.key()].value ) = split(feature.value, mask);
    //     tie( Z1[el.key()].time, Z2[el.key()].time ) = split(feature.time, mask);
    // }
    std::tie{ Z1, Z2 } = Z.split(mask);

    // create two new data objects and return
    array<Data, 2> result = {Data(X1, y1, Z1, 
                                    var_names, classification), 
                                Data(X2, y2, Z2, 
                                    var_names, classification)
                            };

    return result;
}

CVData::CVData()
{
    oCreated = false;
    tCreated = false;
    vCreated = false;
}

CVData::CVData(MatrixXf& X, ArrayXf& y, Longitudinal & Z, 
                const vector<string>& variable_names,
                bool c)
{
    o = new Data(X, y, Z, variable_names, c);
    oCreated = true;
    
    t = new Data(X_t, y_t, Z_t, variable_names, c);
    tCreated = true;
    
    v = new Data(X_v, y_v, Z_v, variable_names, c);
    vCreated = true;
    
    classification = c;
    
    // split data into training and test sets
    //train_test_split(params.shuffle, params.split);
}

CVData::~CVData()
{
    if(o != NULL && oCreated)
    {
        delete(o);
        o = NULL;
    }
    
    if(t != NULL && tCreated)
    {
        delete(t);
        t = NULL;
    }
    
    if(v != NULL && vCreated)
    {
        delete(v);
        v = NULL;
    }
}

void CVData::setOriginalData(MatrixXf& X, ArrayXf& y, Longitudinal& Z,
                                const vector<string>& variable_names,
                                bool c)
{
    o = new Data(X, y, Z, variable_names, c);
    oCreated = true;
    
    t = new Data(X_t, y_t, Z_t, variable_names, c);
    tCreated = true;
    
    v = new Data(X_v, y_v, Z_v, variable_names, c);
    vCreated = true;
    
    classification = c;
}

void CVData::setOriginalData(Data *d)
{
    o = d;
    oCreated = false;
    
    t = new Data(X_t, y_t, Z_t, d->var_names, d->classification);
    tCreated = true;
    
    v = new Data(X_v, y_v, Z_v, d->var_names, d->classification);
    vCreated = true;
    
    classification = d->classification;
}

void CVData::setTrainingData(MatrixXf& X_t, ArrayXf& y_t, Longitudinal& Z_t,
                                const vector<string>& variable_names,
                                bool c)
{
    t = new Data(X_t, y_t, Z_t, variable_names, c);
    tCreated = true;
    
    classification = c;
}

void CVData::setTrainingData(Data *d, bool toDelete)
{
    t = d;
    if(!toDelete)
        tCreated = false;
    else
        tCreated = true;
}

void CVData::setValidationData(MatrixXf& X_v, ArrayXf& y_v, 
                                Longitudinal& Z_v,
                                const vector<string>& variable_names,
                                bool c)
{
    v = new Data(X_v, y_v, Z_v, variable_names,  c);
    vCreated = true;
}

void CVData::setValidationData(Data *d)
{
    v = d;
    vCreated = false;
}

void CVData::shuffle_data()
{
    Eigen::PermutationMatrix<Dynamic,Dynamic> perm(o->X.cols());
    perm.setIdentity();
    r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
    /* cout << "X before shuffle: \n"; */
    /* cout << o->X.transpose() << "\n"; */
    o->X = o->X * perm;       // shuffles columns of X

    /* cout << "X after shuffle: \n"; */
    /* cout << o->X.transpose() << "\n"; */
    o->y = (o->y.matrix().transpose() * perm).transpose().array() ;       // shuffle y too
    
    if(o->Z.size() > 0)
    {
        std::vector<int> zidx(o->y.size());
        // zidx maps the perm_indices values to their indices, i.e. the inverse transform
        for (unsigned i = 0; i < perm.indices().size(); ++i)
            zidx.at(perm.indices()(i)) = i;
        /* cout << "zidx :\n"; */
        /* for (const auto& zi : zidx) */
        /*     cout << zi << "," ; */
        /* cout << "\n"; */
        reorder(o->Z, zidx);
    }
}

void CVData::split_stratified(float split)
{
    logger.log("Stratify split called with initial data size as " 
            + std::to_string(o->X.cols()), 3);
                    
    std::map<float, vector<int>> label_indices;
        
    //getting indices for all labels
    for(int x = 0; x < o->y.size(); x++)
        label_indices[o->y[x]].push_back(x);
            
    std::map<float, vector<int>>::iterator it = label_indices.begin();
    
    vector<int> t_indices;
    vector<int> v_indices;
    
    int t_size;
    int x;
    
    for(; it != label_indices.end(); it++)
    {
        t_size = ceil(it->second.size()*split);
        
        for(x = 0; x < t_size; x++)
            t_indices.push_back(it->second[x]);
            
        for(; x < it->second.size(); x++)
            v_indices.push_back(it->second[x]);
        
        logger.log("Label is " + to_string(it->first), 3, "\t");
        logger.log("Total size = " + to_string(it->second.size()), 3, 
                "\t");
        logger.log("training_size = " + to_string(t_size), 3, "\t");
        logger.log("verification size = " 
                + to_string(it->second.size() - t_size), 3, "\t");
        
    }
    
    X_t.resize(o->X.cols(), t_indices.size());
    X_v.resize(o->X.cols(), v_indices.size());
    y_t.resize(t_indices.size());
    y_v.resize(v_indices.size());
    
    sort(t_indices.begin(), t_indices.end());
    
    for(int x = 0; x < t_indices.size(); x++)
    {
        t->X.col(x) = o->X.col(t_indices[x]);
        t->y[x] = o->y[t_indices[x]];
        t->Z[x] = o->Z[t_indices[x]];
    }
    
    sort(v_indices.begin(), v_indices.end());
    
    for(int x = 0; x < v_indices.size(); x++)
    {
        v->X.col(x) = o->X.col(v_indices[x]);
        v->y[x] = o->y[v_indices[x]];
        v->Z[x] = o->Z[t_indices[x]];
    }

    
}

void CVData::train_test_split(bool shuffle, float split)
{
    /* @param X: n_features x n_samples matrix of training data
        * @param y: n_samples vector of training labels
        * @param shuffle: whether or not to shuffle X and y
        * @param[out] X_t, X_v, y_t, y_v: training and validation matrices
        */
        
                                
    if (shuffle)     // generate shuffle index for the split
        shuffle_data();
        
    if(classification)
        split_stratified(split);
    else
    {        
        // resize training and test sets
        X_t.resize(o->X.cols(),int(o->X.cols()*split));
        X_v.resize(o->X.cols(),int(o->X.cols()*(1-split)));
        y_t.resize(int(o->y.size()*split));
        y_v.resize(int(o->y.size()*(1-split)));
        
        // map training and test sets  
        t->X = MatrixXf::Map(o->X.data(),t->X.cols(),t->X.cols());
        v->X = MatrixXf::Map(o->X.data()+t->X.cols()*t->X.cols(),
                                    v->X.cols(),v->X.cols());

        t->y = ArrayXf::Map(o->y.data(),t->y.size());
        v->y = ArrayXf::Map(o->y.data()+t->y.size(),v->y.size());
        if(o->Z.size() > 0)
            split_longitudinal(o->Z, t->Z, v->Z, split);
    }

}  

void CVData::split_longitudinal(
                        Longitudinal& Z,
                        Longitudinal& Z_t,
                        Longitudinal& Z_v,
                        float split)
{

    
    /* for ( const auto val: Z ) */
    /* { */
    /*     size = Z[val.first].first.size(); */
    /*     break; */
    /* } */
    
    int trainSize = int(Z.size()*split);
    int validateSize = int(Z.size()*(1-split));

    Z_t.assign(Z.begin(), Z.begin() + trainSize);
    Z_v.assign(Z.begin()+trainSize, Z.begin() + trainSize+validateSize);
        
    /* for ( const auto &val: Z ) */
    /* { */
    /*     vector<ArrayXf> _Z_t_v, _Z_t_t, _Z_v_v, _Z_v_t; */
    /*     _Z_t_v.assign(Z[val.first].first.begin(), */ 
    /*             Z[val.first].first.begin()+testSize); */
    /*     _Z_t_t.assign(Z[val.first].second.begin(), */ 
    /*             Z[val.first].second.begin()+testSize); */
    /*     _Z_v_v.assign(Z[val.first].first.begin()+testSize, */ 
    /*                   Z[val.first].first.begin()+testSize+validateSize); */
    /*     _Z_v_t.assign(Z[val.first].second.begin()+testSize, */ 
    /*                   Z[val.first].second.begin()+testSize+validateSize); */
        
    /*     Z_t[val.first] = make_pair(_Z_t_v, _Z_t_t); */
    /*     Z_v[val.first] = make_pair(_Z_v_v, _Z_v_t); */
    /* } */
}
    
    
} // data
} // Brush
