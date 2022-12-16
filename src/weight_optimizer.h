#ifndef WEIGHT_OPTIMIZER_H
#define WEIGHT_OPTIMIZER_H
#include "init.h"
#include "tree_node.h"

namespace Brush
{

auto mean_squared_error(ArrayXf y, ArrayXf y_pred)
{
    return (y-y_pred).norm();
}

struct WeightOptimizer
{
    // put ceres stuff in here!

    auto new_weights update(Program& p, const Dataset& d)
    {
        // update weights to return new_weights using Non-linear least squares.
        // target: d.y
        // get a copy of the weights from the tree. 
        auto weights = p.get_weights();
        int n_weights = weights.size();

    }
}

}
#endif