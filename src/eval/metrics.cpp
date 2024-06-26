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
    float eps = pow(10,-10);
    
    VectorXf loss;
    
    float sum_weights = 0; 
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

        if (!class_weights.empty())
        {
            loss(i) = loss(i) * class_weights.at(y(i));
            sum_weights += class_weights.at(y(i));
        }
    }
    
    if (sum_weights > 0)
        loss = loss.array() / sum_weights * y.size(); // normalize weight contributions
    
    return loss;
}   

/// log loss
float mean_log_loss(const VectorXf& y, 
        const VectorXf& predict_proba, VectorXf& loss,
        const vector<float>& class_weights)
{
    loss = log_loss(y,predict_proba,class_weights);
    return loss.mean();
}

float average_precision_score(const VectorXf& y, const VectorXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights) {
    
    // get argsort of predict proba (descending)
    vector<int> argsort(y.rows());
    iota(argsort.begin(), argsort.end(), 0);
    sort(argsort.begin(), argsort.end(), [&](int i, int j) {
        return predict_proba[i] > predict_proba[j];
    });

    float ysum = 0;
    if (!class_weights.empty()) 
        for (int i = 0; i < y.size(); i++) {
            ysum += y(i) * class_weights.at(y(i));
        }
    else
        ysum = y.sum();

    // Calculate the precision and recall values
    VectorXf precision(y.rows());
    VectorXf recall(y.rows());
    
    loss.resize(y.rows()); 

    float true_positives  = 0;
    float false_positives = 0;

    float positives = 0;

    for (int i = 0; i < y.rows(); i++) {
        if (predict_proba[argsort[i]] > 0.5)
        {
            if (y[argsort[i]] == 1) {
                if (!class_weights.empty())
                    true_positives = class_weights[y(argsort[i])];
                else
                    true_positives += 1;
            }
            else {
                if (!class_weights.empty())
                    false_positives = class_weights[y(argsort[i])];
                else
                    false_positives += 1;
            }
        }
        positives = true_positives + false_positives;

        precision[i] = positives==0.0 ? 0.0 : true_positives/positives;
        recall[i]    = ysum==0.0 ? 1.0 : true_positives/ysum;

        // cout << "i: " << i << ", predict_proba[argsort[i]]: " << predict_proba[argsort[i]] << ", y[argsort[i]]: " << y[argsort[i]] << ", true_positives: " << true_positives << ", false_positives: " << false_positives << ", precision[i]: " << precision[i] << ", recall[i]: " << recall[i] << endl;
    }

    // Calculate the average precision score
    float average_precision = 0;
    float last_recall = 0;
    float last_loss   = 0;

    for (int i = 0; i < y.rows(); i++) {
        if (recall[i] != last_recall) {
            loss[i] = precision[i] * (recall[i] - last_recall);
            average_precision += loss[i];

            last_recall = recall[i];
            last_loss = loss[i];
        }
        else
            loss[i] = last_loss;

        // cout << "i: " << i << ", recall[i]: " << recall[i] << ", loss[i]: " << loss[i] << ", average_precision: " << average_precision << endl;
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

    // float eps = pow(10,-10);
    // float sum_weights = 0; 
    // for (unsigned i = 0; i < y.rows(); ++i)
    // {
    //     for (const auto& c : uc)
    //     {
    //         // for specific class
    //         ArrayXf yhat = predict_proba.col(int(c));
    //         /* std::cout << "class " << c << "\n"; */

    //         /* float yi = y(i) == c ? 1.0 : 0.0 ; */ 
    //         /* std::cout << "yi: " << yi << ", yhat(" << i << "): " << yhat(i) ; */  
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
    //             /* std::cout << ", loss(" << i << ") = " << loss(i); */
    //         }
    //         /* std::cout << "\n"; */
    //         }
    //     if (!class_weights.empty()){
    //         /* std::cout << "weights.at(y(" << i << ")): " << class_weights.at(y(i)) << "\n"; */
    //         loss(i) = loss(i)*class_weights.at(y(i));
    //         sum_weights += class_weights.at(y(i));
    //     }
    // }
    // if (sum_weights > 0)
    //     loss = loss.array() / sum_weights * y.size(); 

    /* cout << "loss.mean(): " << loss.mean() << "\n"; */
    /* cout << "loss.sum(): " << loss.sum() << "\n"; */
    return loss;
}

float mean_multi_log_loss(const VectorXf& y, 
        const ArrayXXf& predict_proba, VectorXf& loss,
        const vector<float>& class_weights)
{
    loss = multi_log_loss(y, predict_proba, class_weights);

    /* std::cout << "loss: " << loss.transpose() << "\n"; */
    /* std::cout << "mean loss: " << loss.mean() << "\n"; */
    return loss.mean();
}  

} // metrics
} // Brush