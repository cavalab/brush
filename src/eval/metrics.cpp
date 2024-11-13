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
    if (!class_weights.empty()) {
        for (int i = 0; i < y.rows(); ++i) {
            loss(i) *= class_weights.at(y(i));
        }
    }

    // since loss is wrong predictions, we need to invert it
    return 1.0 - loss.mean();
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
        // float weight = class_weights.empty() ? 1.0f : class_weights.at(y(i));
        float weight = 1.0f; // it is already balanced; ignoring class weights
        
        if      (yhat(i) == 1.0 && y(i) == 1.0) TP += weight;
        else if (yhat(i) == 1.0 && y(i) == 0.0) FP += weight;
        else if (yhat(i) == 0.0 && y(i) == 0.0) TN += weight;
        else                                    FN += weight;
    }

    float eps = pow(10,-10);
    
    float TPR = (TP + eps) / (TP + FN + eps);
    float TNR = (TN + eps) / (TN + FP + eps);

    return (TPR + TNR) / 2.0;
}

float average_precision_score(const VectorXf& y, const VectorXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights) {
    
    // TODO: revisit this
    
    float eps = pow(10,-10);

    // Assuming y contains binary labels (0 or 1)
    int num_instances = y.rows();

    // get argsort of predict proba (descending)
    vector<int> argsort(num_instances);

    iota(argsort.begin(), argsort.end(), 0);
    sort(argsort.begin(), argsort.end(), [&](int i, int j) {
        return predict_proba(i) > predict_proba(j);
    });

    float ysum = 0;
    if (!class_weights.empty()) 
        for (int i = 0; i < y.size(); i++) {
            ysum += y(i) * class_weights.at(y(i));
        }
    else
        ysum = y.sum();

    // Calculate the precision and recall values
    VectorXf precision(num_instances);
    VectorXf recall(num_instances);
    
    loss.resize(num_instances); 

    float true_positives  = 0;
    float false_positives = 0;
    float positives = 0; // y.sum();

    for (int i = 0; i < num_instances; ++i) {
        int index = argsort[i];
        
        if (predict_proba(index) > 0.5)
        {
            float weight = class_weights.empty() ? 1.0f : class_weights.at(y(index));

            if (y(index) > 0.5) {
                true_positives += weight;
            }
            else {
                false_positives += weight;
            }
        }
        
        positives = true_positives + false_positives;

        precision(i) = positives==0.0 ? 0.0 : true_positives/positives;
        recall(i)    = ysum==0.0 ? 1.0 : true_positives/ysum;
    }

    // Calculate the average precision score
    float average_precision = 0.0;
    float last_recall = recall(0);

    for (int i = 0; i < num_instances; ++i) {
        if (recall(i) != last_recall) {
            average_precision += precision(i) * (recall(i) - last_recall);
            last_recall = recall(i);
        }
        
        // The loss vector is used in lexicase selection. we need to set something useful here
        // that does make sense on individual level. Using log loss here.
        if (predict_proba(i) < eps || 1 - predict_proba(i) < eps)
            loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
        else
            loss(i) = -(y(i)*log(predict_proba(i)) + (1-y(i))*log(1-predict_proba(i)));
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
    //     //std::cout << "TP(" << i << "): " << TP.at(i) << ", P[" << i << "]: " << P.at(i) << "\n";
    //     //std::cout << "TN(" << i << "): " << TN.at(i) << ", N[" << i << "]: " << N.at(i) << "\n";
    //     //std::cout << "class accuracy(" << i << "): " << class_accuracies(i) << "\n";
    // }
    
    // // set loss vectors if third argument supplied
    // loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<float>();

    // return 1.0 - class_accuracies.mean();
    
    return 0.0;
}

} // metrics
} // Brush