#ifndef METRICS_H
#define METRICS_H

#include "../data/data.h"

namespace Brush {
namespace Eval {

/* Scoring functions */

/// mean squared error
float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
            const vector<float>& weights=vector<float>() );
            
// TODO: implement other metrics. Right know I have just the MSE

} // metrics
} // Brush

#endif