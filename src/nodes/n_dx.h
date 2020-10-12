#ifndef NODE_DIF_H
#define NODE_DIF_H

#include "node.h"

namespace BR{

class NodeDx : public Node
{
    public:
        // weights
        std::vector<float> W;
        // gradient descent parameters
        // intermediate weights 
        std::vector<float> V;  
        // gradients
        std::vector<ArrayXf> ddw; 
        std::vector<ArrayXf> ddx; 
        // learning rate
        float lr;
        // momentum
        float m;

        virtual ~NodeDx();

        virtual ArrayXf getDerivative() = 0;
        
        /* void derivative(vector<ArrayXf>& gradients, Trace& state, int loc); */
        virtual backprop(const Data& d, const ArrayXf gradient, 
                          TreeNode* child1, TreeNode* child2);

        virtual update_weights(const Data& d, const ArrayXf gradient, 
                          TreeNode* child1, TreeNode* child2);
        void update(const Data& d);

        void print_weight();

        bool isNodeDx(){ return true; }
};

}

#endif
