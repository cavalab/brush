#include "metrics.h"

namespace Brush {
namespace Eval {

/* Scoring functions */

/// mean squared error
float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
            const vector<float>& weights)
{
    loss = (yhat - y).array().pow(2);
    return loss.mean(); 
}


// TODO: implement other metrics. Right know I have just the MSE

} // metrics
} // Brush