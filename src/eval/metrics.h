#ifndef METRICS_H
#define METRICS_H

#include "../data/data.h"

namespace Brush {
namespace Eval {

/* Scoring functions */

/// mean squared error
float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
            const vector<float>& class_weights=vector<float>() );

// TODO: test cases for the metrics
// TODO: implement the metrics for classification

/// log loss (2 methods below)
VectorXf log_loss(const VectorXf& y, const VectorXf& predict_proba, 
                    const vector<float>& class_weights=vector<float>());

float mean_log_loss(const VectorXf& y, const VectorXf& predict_proba, VectorXf& loss,
                    const vector<float>& class_weights = vector<float>());

/// multinomial log loss (2 methods below)
VectorXf multi_log_loss(const VectorXf& y, const ArrayXXf& predict_proba, 
        const vector<float>& class_weights=vector<float>());

float mean_multi_log_loss(const VectorXf& y, const ArrayXXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

// TODO: average_precision_score for classification
// TODO: implement other metrics. Right know I have just the MSE

} // metrics
} // Brush

#endif