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

// anonymous namespace. make the headers private to metrics.cpp. only affects linkage
namespace {

// per-sample weights from class weights (all ones if no class weights)
vector<float> sample_weights(const VectorXf& y, const vector<float>& class_weights)
{
    vector<float> w(y.size(), 1.0f);
    if (!class_weights.empty())
        for (int i = 0; i < y.size(); ++i)
            w[i] = class_weights.at(static_cast<int>(y(i)));
    return w;
}

// Binary average precision. `y` holds 0/1 labels and `w` per-sample weights.
float binary_average_precision(const VectorXf& y, const VectorXf& predict_proba,
                               const vector<float>& w)
{
    int num_instances = y.size();
    float eps = 1e-6f;

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
        w_sorted[i] = w[idx];

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

// Binary AUROC (trapezoidal rule over the ROC curve, treating tied scores as
// a single threshold, like sklearn). `y` holds 0/1 labels and `w` per-sample
// weights. Returns 0.5 if only one class is present (AUROC is undefined).
float binary_roc_auc(const VectorXf& y, const VectorXf& predict_proba,
                     const vector<float>& w)
{
    int num_instances = y.size();

    vector<int> order(num_instances);
    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int i, int j) {
        return predict_proba(i) > predict_proba(j); // descending
    });

    float pos = 0.0f;
    float neg = 0.0f;
    for (int i = 0; i < num_instances; ++i) {
        // remember: this is for the binary case!
        pos += y(i) * w[i];
        neg += (1.0f - y(i)) * w[i];
    }

    // default case, copying sklearn, returns 0.5 if only one class exists in the y
    if (pos == 0.0f || neg == 0.0f)
        return 0.5f;

    float tp = 0.0f, fp = 0.0f;
    float tp_prev = 0.0f, fp_prev = 0.0f;
    float area = 0.0f;
    for (int i = 0; i < num_instances; ++i) {
        int idx = order[i];
        tp += y(idx) * w[idx];
        fp += (1.0f - y(idx)) * w[idx];

        // only add a point to the curve at the end of a block of tied scores
        bool last_of_block = (i == num_instances - 1)
            || (predict_proba(order[i+1]) != predict_proba(idx));

        if (last_of_block) {
            area += (fp - fp_prev) * (tp + tp_prev) / 2.0f;
            tp_prev = tp;
            fp_prev = fp;
        }
    }

    return area / (pos * neg);
}

// Weighted confusion matrix entries for class `label` (one-vs-rest).
void confusion(const VectorXf& y, const ArrayXi& yhat, int label,
               const vector<float>& w, float& TP, float& FP, float& FN)
{
    // Used to calculate precision and recall for multiclass settings.
    // TP, FP, FN, passed as reference

    TP = FP = FN = 0.0f;
    for (int i = 0; i < y.size(); ++i) {
        bool is_true = static_cast<int>(y(i)) == label;
        bool is_pred = yhat(i) == label;

        if      ( is_true &&  is_pred) TP += w[i];
        else if (!is_true &&  is_pred) FP += w[i];
        else if ( is_true && !is_pred) FN += w[i];
    }
}

ArrayXi argmax_rows(const ArrayXXf& predict_proba)
{
    // converting the pred proba matrix to predictions

    ArrayXi yhat(predict_proba.rows());
    for (int i = 0; i < predict_proba.rows(); ++i)
        predict_proba.row(i).maxCoeff(&yhat(i));

    return yhat;
}

// Macro average of precision or recall over the classes present in either
// the true or predicted labels (sklearn's default label set).
float multi_macro_precision_recall(const VectorXf& y, const ArrayXXf& predict_proba,
                                   VectorXf& loss, const vector<float>& class_weights,
                                   bool precision)
{
    if (predict_proba.rows() != y.rows())
        HANDLE_ERROR_THROW("Multiclass probabilities and labels have different numbers of rows");

    ArrayXi yhat = argmax_rows(predict_proba);

    // again setting the loss here as hit or miss, a.k.a. accuracy
    loss = (yhat != y.cast<int>().array()).cast<float>();

    vector<float> w = sample_weights(y, class_weights);

    float sum = 0.0f;
    int n_labels = 0;
    for (int label = 0; label < predict_proba.cols(); ++label) {
        bool present = (y.cast<int>().array() == label).any() || (yhat == label).any();
        if (!present)
            continue;

        float TP, FP, FN;
        confusion(y, yhat, label, w, TP, FP, FN);

        float denom = precision ? TP + FP : TP + FN;
        sum += denom == 0.0f ? 0.0f : TP / denom;
        ++n_labels;
    }
    return n_labels == 0 ? 0.0f : sum / n_labels;
}

} // anonymous namespace

float average_precision_score(const VectorXf& y, const VectorXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights) {
    
    // AP is implemented as AUC PR in sklearn.
    // AP summarizes a precision-recall curve as the weighted mean of precisions
    // achieved at each threshold, with the increase in recall from the previous threshold used as the weight

    // The loss vector is used in lexicase selection. we need to set something useful here
    // that does make sense on individual level. Using log loss here.
    loss = log_loss(y, predict_proba, class_weights);

    return binary_average_precision(y, predict_proba, sample_weights(y, class_weights));
}

// implementing precision_score and recall_score for the binary case.
// it will be used per-class in the multiclass case below.
float precision_score(const VectorXf& y, const VectorXf& predict_proba,
                      VectorXf& loss, const vector<float>& class_weights)
{
    ArrayXi yhat = (predict_proba.array() > 0.5).cast<int>();

    // Again updating the loss vector. Doing the same way as binary accuracy (zero_one_loss) here
    loss = (yhat != y.cast<int>().array()).cast<float>();

    float TP, FP, FN;
    confusion(y, yhat, 1, sample_weights(y, class_weights), TP, FP, FN);

    return (TP + FP) == 0.0f ? 0.0f : TP / (TP + FP);
}

float recall_score(const VectorXf& y, const VectorXf& predict_proba,
                   VectorXf& loss, const vector<float>& class_weights)
{
    ArrayXi yhat = (predict_proba.array() > 0.5).cast<int>();
    loss = (yhat != y.cast<int>().array()).cast<float>();

    float TP, FP, FN;
    confusion(y, yhat, 1, sample_weights(y, class_weights), TP, FP, FN);

    return (TP + FN) == 0.0f ? 0.0f : TP / (TP + FN);
}

float roc_auc_score(const VectorXf& y, const VectorXf& predict_proba,
                    VectorXf& loss, const vector<float>& class_weights)
{
    // AUROC is not decomposable per sample; log loss is used for lexicase
    loss = log_loss(y, predict_proba, class_weights);

    return binary_roc_auc(y, predict_proba, sample_weights(y, class_weights));
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

float multi_precision_score(const VectorXf& y, const ArrayXXf& predict_proba,
    VectorXf& loss, const vector<float>& class_weights)
{
    return multi_macro_precision_recall(y, predict_proba, loss, class_weights, true);
}

float multi_recall_score(const VectorXf& y, const ArrayXXf& predict_proba,
    VectorXf& loss, const vector<float>& class_weights)
{
    return multi_macro_precision_recall(y, predict_proba, loss, class_weights, false);
}

float multi_roc_auc_score(const VectorXf& y, const ArrayXXf& predict_proba,
    VectorXf& loss, const vector<float>& class_weights)
{
    loss = multi_log_loss(y, predict_proba, class_weights);

    vector<float> w = sample_weights(y, class_weights);

    float sum = 0.0f;
    int n_labels = 0;
    for (int label = 0; label < predict_proba.cols(); ++label) {
        VectorXf y_bin = (y.cast<int>().array() == label).cast<float>();

        // one-vs-rest AUROC is undefined if the class is absent (or is the only one)
        if (y_bin.sum() == 0.0f || y_bin.sum() == y_bin.size())
            continue;

        sum += binary_roc_auc(y_bin, predict_proba.col(label).matrix(), w);
        ++n_labels;
    }
    return n_labels == 0 ? 0.5f : sum / n_labels;
}

float multi_average_precision_score(const VectorXf& y, const ArrayXXf& predict_proba,
    VectorXf& loss, const vector<float>& class_weights)
{
    loss = multi_log_loss(y, predict_proba, class_weights);

    vector<float> w = sample_weights(y, class_weights);

    float sum = 0.0f;
    int n_labels = 0;
    for (int label = 0; label < predict_proba.cols(); ++label) {
        VectorXf y_bin = (y.cast<int>().array() == label).cast<float>();

        // recall is undefined if the class is absent
        if (y_bin.sum() == 0.0f)
            continue;

        sum += binary_average_precision(y_bin, predict_proba.col(label).matrix(), w);
        ++n_labels;
    }
    return n_labels == 0 ? 0.0f : sum / n_labels;
}

} // metrics
} // Brush
