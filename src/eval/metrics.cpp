#include "metrics.h"

#include <algorithm>

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
        w_sorted[i] = class_weights.empty() ? 1.0f : class_weights.at(y(idx));

        ysum += y_sorted[i] * w_sorted[i];
    }

    // when all scores are the same, the sort order is arbitrary, so the PR curve
    // you integrate is a staircase instead of a flat line. Sklearn avoids this by
    // treating ties as one threshold.
    // however, this does not produce consistent results, so we will handle flat
    // lines below

    // detect constant prediction case (all p_sorted equal within tolerance).
    // because p_sorted is sorted, the first element is the maximum, and the last is the minimum,
    if (fabs(p_sorted.back() - p_sorted.front()) <= eps) {
        // All predictions are (effectively) constant.
        float total_weight = std::accumulate(w_sorted.begin(), w_sorted.end(), 0.0f);

        // Return weighted positives / total weight, matching sklearn's result for constant scores
        // (kinda weighted prevalence)
        return total_weight == 0.0f ? 0.0f : ysum / total_weight;
    }

    // Find the indexes where prediction changes, so we can treat it as one block
    vector<int> unique_indices = {}; // this one will be used to calculate the AUC
    set<float> unique_probas = {}; // keep track of unique elements (this wont be used other than that)
    
    for (int i=0; i<p_sorted.size(); ++i)
        if (unique_probas.insert(p_sorted.at(i)).second)
            unique_indices.push_back(i);

    unique_indices.push_back(num_instances); // last index is the number of elements

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

            float relevant = tp + fp;
            precision.push_back(relevant == 0.0f ? 0.0f : tp / relevant);
            recall.push_back(ysum == 0.0f ? 1.0f : tp / ysum);
        }
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
    if (predict_proba.rows() != y.rows())
        HANDLE_ERROR_THROW("Multiclass probabilities and labels have different numbers of rows");

    constexpr float eps = 1e-6f;
    VectorXf loss(y.rows());
    for (int i = 0; i < y.rows(); ++i)
    {
        const int label = static_cast<int>(y(i)); // labels are always encoded as integers for clf/multiclf
        
        // if (label < 0 || label >= predict_proba.cols())
        //     HANDLE_ERROR_THROW("Class label is outside the predicted probability columns");

        // per sample log loss
        loss(i) = -std::log(std::clamp(predict_proba(i, label), eps, 1.0f - eps));
    }
    return loss;
}

float mean_multi_log_loss(const VectorXf& y, 
        const ArrayXXf& predict_proba, VectorXf& loss,
        const vector<float>& class_weights)
{
    loss = multi_log_loss(y, predict_proba, class_weights);

    if (class_weights.empty())
        return loss.mean();

    // apply class weights to the log loss
    float sum_weights = 0.0f;
    float weighted_loss = 0.0f;
    for (int i = 0; i < y.rows(); ++i)
    {
        const float weight = class_weights.at(static_cast<int>(y(i)));
        weighted_loss += loss(i) * weight;
        sum_weights += weight;
    }
    return sum_weights == 0.0f ? 0.0f : weighted_loss / sum_weights;
}  

float multi_zero_one_loss(const VectorXf& y,
    const ArrayXXf& predict_proba, VectorXf& loss, 
    const vector<float>& class_weights )
{
    if (predict_proba.rows() != y.rows())
        HANDLE_ERROR_THROW("Multiclass probabilities and labels have different numbers of rows");

    ArrayXi yhat(y.rows());
    for (int i = 0; i < predict_proba.rows(); ++i)
        predict_proba.row(i).maxCoeff(&yhat(i)); // pick the predicted class

    loss = (yhat.array() != y.cast<int>().array()).cast<float>(); // check if it was a hit or a miss

    if (class_weights.empty()) // accuracy
        return 1.0f - loss.mean();

    float weighted_errors = 0.0f;
    float sum_weights = 0.0f;
    for (int i = 0; i < y.rows(); ++i)
    {
        const float weight = class_weights.at(static_cast<int>(y(i)));
        weighted_errors += loss(i) * weight;
        sum_weights += weight;
    }
    return sum_weights == 0.0f ? 0.0f : 1.0f - weighted_errors / sum_weights;
}

float multi_bal_zero_one_loss(const VectorXf& y,
    const ArrayXXf& predict_proba, VectorXf& loss,
    const vector<float>& class_weights)
{
    if (predict_proba.rows() != y.rows())
        HANDLE_ERROR_THROW("Multiclass probabilities and labels have different numbers of rows");

    ArrayXi yhat(y.rows());
    for (int i = 0; i < predict_proba.rows(); ++i)
        predict_proba.row(i).maxCoeff(&yhat(i));
    loss = (yhat.array() != y.cast<int>().array()).cast<float>();

    VectorXf correct = VectorXf::Zero(predict_proba.cols());
    VectorXf support = VectorXf::Zero(predict_proba.cols());
    for (int i = 0; i < y.rows(); ++i)
    {
        const int label = static_cast<int>(y(i));

        // if (label < 0 || label >= predict_proba.cols())
        //     HANDLE_ERROR_THROW("Class label is outside the predicted probability columns");

        // balanced, weighted by support
        support(label) += 1.0f;
        if (yhat(i) == label)
            correct(label) += 1.0f;
    }

    float recall_sum = 0.0f;
    int present_classes = 0;
    for (int label = 0; label < support.size(); ++label)
        if (support(label) > 0.0f)
        {
            recall_sum += correct(label) / support(label);
            ++present_classes;
        }
    return present_classes == 0 ? 0.0f : recall_sum / present_classes;
}

} // metrics
} // Brush
