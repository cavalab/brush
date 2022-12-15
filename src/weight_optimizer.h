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

    auto new_weights update(const Dataset& d, tree<Node>& prg, ArrayXf& weights)
    {
        // update weights to return new_weights using Non-linear least squares.
        // target: d.y

    }
}

}
#endif