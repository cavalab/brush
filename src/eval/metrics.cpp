#include "metrics.h"

namespace Brush {
namespace Eval {

/* Scoring functions */

/// mean squared error
float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
            const vector<float>& class_weights)
{
    loss = (yhat - y).array().pow(2);
    return loss.mean(); 
}


VectorXf log_loss(const VectorXf& y, const VectorXf& predict_proba, 
                    const vector<float>& class_weights)
{
    // See comments on weight_optimizer to learn more about why am I using
    // this value for eps. TL;DR: dont change, can cause weird behaviour
    float eps = 1e-6f;
    
    VectorXf loss;
    
    loss.resize(y.rows());  
    for (unsigned i = 0; i < y.rows(); ++i)
    {
        if (predict_proba(i) < eps || 1 - predict_proba(i) < eps)
            // clip probabilities since log loss is undefined for predict_proba=0 or predict_proba=1
            loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
        else
            loss(i) = -(y(i)*log(predict_proba(i)) + (1-y(i))*log(1-predict_proba(i)));

        if (loss(i)<0)
            std::runtime_error("loss(i)= " + to_string(loss(i)) 
                    + ". y = " + to_string(y(i)) + ", predict_proba(i) = " 
                    + to_string(predict_proba(i)));
    }
    
    return loss;
}   

/// log loss
float mean_log_loss(const VectorXf& y, 
        const VectorXf& predict_proba, VectorXf& loss,
        const vector<float>& class_weights)
{
    loss = log_loss(y,predict_proba,class_weights);
    
    if (!class_weights.empty())
    {
        float sum_weights = 0;

        // we keep loss without weights, as this may affect lexicase
        VectorXf weighted_loss;
        weighted_loss.resize(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            weighted_loss(i) = loss(i) * class_weights.at(y(i));      
            sum_weights += class_weights.at(y(i));
        }

        // equivalent of sklearn's log_loss with weights. It uses np.average,
        // which returns avg = sum(a * weights) / sum(weights)
        return weighted_loss.sum() / sum_weights; // normalize weight contributions
    }
    
    return loss.mean();
}

// accuracy
float zero_one_loss(const VectorXf& y,
        const VectorXf& predict_proba, VectorXf& loss, 
        const vector<float>& class_weights )
{
    VectorXi yhat = (predict_proba.array() > 0.5).cast<int>();

    // we are actually finding wrong predictions here
    loss = (yhat.array() != y.cast<int>().array()).cast<float>();

    // Apply class weights if provided
    float scale = 0.0f;
    if (!class_weights.empty()) {
        for (int i = 0; i < y.rows(); ++i) {
            loss(i) *= class_weights.at(y(i));
            scale += class_weights.at(y(i));
        }
    }
    else
    {
        scale = static_cast<float>(loss.size());
    }

    // since `loss` contains wrong predictions, we need to invert it
    return 1.0 - (loss.sum() / scale);
}

// balanced accuracy
float bal_zero_one_loss(const VectorXf& y,
        const VectorXf& predict_proba, VectorXf& loss, 
        const vector<float>& class_weights )
{
    VectorXi yhat = (predict_proba.array() > 0.5).cast<int>();

    loss = (yhat.array() != y.cast<int>().array()).cast<float>();

    float TP = 0;
    float FP = 0;
    float TN = 0;
    float FN = 0;

    int num_instances = y.rows();
    for (int i = 0; i < num_instances; ++i) {
        float weight = 1.0f; // it is a balanced metric; ignoring class weights
        // float weight = class_weights.empty() ? 1.0f : class_weights.at(y(i));
        
        if      (yhat(i) == 1.0 && y(i) == 1.0) TP += weight;
        else if (yhat(i) == 1.0 && y(i) == 0.0) FP += weight;
        else if (yhat(i) == 0.0 && y(i) == 0.0) TN += weight;
        else                                    FN += weight;
    }

    float eps = 1e-6f;
    
    float TPR = (TP + eps) / (TP + FN + eps);
    float TNR = (TN + eps) / (TN + FP + eps);

    return (TPR + TNR) / 2.0;
}

float average_precision_score(const VectorXf& y, const VectorXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights) {
    
    // AP is implemented as AUC PR in sklearn.
    // AP summarizes a precision-recall curve as the weighted mean of precisions
    // achieved at each threshold, with the increase in recall from the previous threshold used as the weight

    // Assuming y contains binary labels (0 or 1)
    int num_instances = y.size();

    float eps = 1e-6f; // first we set the loss vector values
    loss.resize(num_instances);
    for (int i = 0; i < num_instances; ++i) {
        float p = predict_proba(i);

        // The loss vector is used in lexicase selection. we need to set something useful here
        // that does make sense on individual level. Using log loss here.
        if (p < eps || 1 - p < eps)
            loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
        else
            loss(i) = -(y(i)*log(p) + (1-y(i))*log(1-p));
    }

    // get argsort of predict proba (descending)
    vector<int> order(num_instances);
    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int i, int j) {
        return predict_proba(i) > predict_proba(j); // descending
    });

    float ysum = 0.0f;
    vector<float> y_sorted(num_instances); // y true
    vector<float> p_sorted(num_instances); // pred probas
    vector<float> w_sorted(num_instances); // sample weights
    for (int i = 0; i < num_instances; ++i) {
        int idx = order[i];

        y_sorted[i] = y(idx);
        p_sorted[i] = predict_proba(idx);
        w_sorted[i] = class_weights.empty() ? 1.0f : class_weights.at(y_sorted[i]);

        ysum += y_sorted[i] * w_sorted[i];
    }

    // when all scores are the same, the sort order is arbitrary, so the PR curve
    // you integrate is a staircase instead of a flat line. Sklearn avoids this by
    // treating ties as one threshold.

    // Find the indexes where prediction changes, so we can treat it as one block
    vector<int> unique_indices = {};
    set<int> unique_probas = {}; // keep track of unique elements
    
    for (int i=0; i<p_sorted.size(); --i)
        if (unique_probas.insert(p_sorted.at(i)).second)
            unique_indices.push_back(i);
    unique_indices.push_back(num_instances); // last index

    float tp = 0.0f;
    float fp = 0.0f;
    vector<float> precision = {1.0};
    vector<float> recall    = {0.0};

    for (size_t i = 0; i < unique_indices.size() - 1; ++i) {
        int start = unique_indices[i];
        int end   = unique_indices[i+1];

        // process group with a for loop (aggregating for each sample)
        for (int j = start; j < end; ++j) {
            tp += y_sorted.at(j) * w_sorted.at(j);
            fp += (1.0f - y_sorted.at(j)) * w_sorted.at(j);
        }

        float relevant = tp + fp;
        precision.push_back(relevant == 0.0f ? 0.0f : tp / relevant);
        recall.push_back(ysum == 0.0f ? 1.0f : tp / ysum);
    }

    // integrate PR curve
    float average_precision = 0.0f;
    for (size_t i = 0; i < precision.size() - 1; ++i) {
        average_precision += (recall[i+1] - recall[i]) * precision[i+1];
    }

    return average_precision;
}

// multinomial log loss
VectorXf multi_log_loss(const VectorXf& y, const ArrayXXf& predict_proba, 
        const vector<float>& class_weights)
{
    // TODO: fix softmax and multiclassification, then implement this
    VectorXf loss = VectorXf::Zero(y.rows());  
    
    // TODO: needs to be the index of unique elements
    // get class labels
    // vector<float> uc = unique( ArrayXi(y.cast<int>()) );

    // float eps = 1e-6f;
    // float sum_weights = 0; 
    // for (unsigned i = 0; i < y.rows(); ++i)
    // {
    //     for (const auto& c : uc)
    //     {
    //         // for specific class
    //         ArrayXf yhat = predict_proba.col(int(c));


    //         /* float yi = y(i) == c ? 1.0 : 0.0 ; */ 

    //         if (y(i) == c)
    //         {
    //             if (yhat(i) < eps || 1 - yhat(i) < eps)
    //             {
    //                 // clip probabilities since log loss is undefined for yhat=0 or yhat=1
    //                 loss(i) += -log(eps);
    //             }
    //             else
    //             {
    //                 loss(i) += -log(yhat(i));
    //             }

    //         }

    //         }
    //     if (!class_weights.empty()){

    //         loss(i) = loss(i)*class_weights.at(y(i));
    //         sum_weights += class_weights.at(y(i));
    //     }
    // }
    // if (sum_weights > 0)
    //     loss = loss.array() / sum_weights * y.size(); 



    return loss;
}

float mean_multi_log_loss(const VectorXf& y, 
        const ArrayXXf& predict_proba, VectorXf& loss,
        const vector<float>& class_weights)
{
    loss = multi_log_loss(y, predict_proba, class_weights);



    return loss.mean();
}  

float multi_zero_one_loss(const VectorXf& y,
    const ArrayXXf& predict_proba, VectorXf& loss, 
    const vector<float>& class_weights )
{
    // TODO: implement this
    // vector<float> uc = unique(y);
    // vector<int> c;
    // for (const auto& i : uc)
    //     c.push_back(int(i));
        
    // // sensitivity (TP) and specificity (TN)
    // vector<float> TP(c.size(),0.0), TN(c.size(), 0.0), P(c.size(),0.0), N(c.size(),0.0);
    // ArrayXf class_accuracies(c.size());
    
    // // get class counts
    
    // for (unsigned i=0; i< c.size(); ++i)
    // {
    //     P.at(i) = (y.array().cast<int>() == c.at(i)).count();  // total positives for this class
    //     N.at(i) = (y.array().cast<int>() != c.at(i)).count();  // total negatives for this class
    // }
    

    // for (unsigned i = 0; i < y.rows(); ++i)
    // {
    //     if (yhat(i) == y(i))                    // true positive
    //         ++TP.at(y(i) == -1 ? 0 : y(i));     // if-then ? accounts for -1 class encoding

    //     for (unsigned j = 0; j < c.size(); ++j)
    //         if ( y(i) !=c.at(j) && yhat(i) != c.at(j) )    // true negative
    //             ++TN.at(j);    
        
    // }

    // // class-wise accuracy = 1/2 ( true positive rate + true negative rate)
    // for (unsigned i=0; i< c.size(); ++i){
    //     class_accuracies(i) = (TP.at(i)/P.at(i) + TN.at(i)/N.at(i))/2; 



    // }
    
    // // set loss vectors if third argument supplied
    // loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<float>();

    // return 1.0 - class_accuracies.mean();
    
    return 0.0;
}

} // metrics
} // Brush