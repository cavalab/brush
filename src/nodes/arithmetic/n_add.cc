/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEADD_H
#define NODEADD_H
//external includes
//
#include <iostream>
#include <string>
// internal includes
#include "n_add.h"

using std::cout;
using std::string;

namespace BR {

NodeAdd::NodeAdd(vector<float> W0)
{
    name = "+";
    otype = 'f';
    arity['f'] = 2;
    complexity = 1;

    if (W0.empty())
        this->init_weights();
    else
        W = W0;
}

State NodeAdd::evaluate(const Data& d, TreeNode* child1,
                        TreeNode* child2, bool train)
{
    State s1 = child1->eval(d);
    State s2 = child2->eval(d);
    ArrayXf x1 = s1.get_data<float>();
    ArrayXf x2 = s2.get_data<float>();
    State s3;
    s3.set<float>(W.at(0)*x1 + W.at(1)*x2);

    if (train)
    {
        this->g.clear();
        ddw.push_back(x1);
        ddw.push_back(x2);
        //ddx0
        ddx.push_back(W.at(0));
        //ddx1
        ddx.push_back(W.at(1));
    }

    return  s3;
}

State NodeAdd::backprop(const Data& d, const ArrayXf gradient, 
                          TreeNode* child1, TreeNode* child2)
{
    this->update_weights(d, gradient);
    child1->backprop(d, gradient*ddx.at(0));
    child2->backprop(d, gradient*ddx.at(1));

}

}
#endif
