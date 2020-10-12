/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#include "n_dx.h"
    		
namespace BR{

NodeDx::~NodeDx(){}

void NodeDx::init_weights()
{
    for (int i = 0; i < arity['f']; i++) 
    {
        W.push_back(r.rnd_dbl());
    }
}

void NodeDx::derivative(vector<ArrayXf>& gradients, Trace& state, int loc) 
{
    gradients.push_back(getDerivative(state, loc));
}

/* void NodeDx::update(vector<ArrayXf>& gradients, Trace& state, float n, float a) */ 
void NodeDx::update_weights(const Data& d, const ArrayXf& gradient)
{
    /*! update weights via gradient descent + momentum
     * @param lr : learning rate
     * @param m : momentum
     * v(t+1) = a * v(t) - n * gradient
     * w(t+1) = w(t) + v(t+1)
     * */
    if (V.empty())  // first time through, V is zeros
    {
        for (const auto& w : W)
            V.push_back(0.0);
    }
    std::cout << "***************************\n";
    std::cout << "Updating " << this->name << "\n";

    // Update all weights
    std::cout << "Current gradient" << gradient << "\n";
    vector<float> W_temp(W);
    vector<float> V_temp(V);
    
    cout << "*****n value is "<< learning_rate <<"\n"; 
    // Have to use temporary weights so as not to compute updates with 
    // updated weights
    for (int i = 0; i < arity['f']; ++i) 
    {
        std::cout << "V[i]: " << V[i] << "\n";
        V_temp[i] = (m * V.at(i) 
                     - lr * (gradient*this->ddw.at(i)).mean() );
        std::cout << "V_temp: " << V_temp[i] << "\n";
    }
    for (int i = 0; i < W.size(); ++i)
    {
        if (std::isfinite(V_temp[i]) && !std::isnan(V_temp[i]))
        {
            this->W[i] += V_temp[i];
            this->V[i] = V_temp[i];
        }
    }

    std::cout << "Updated\n";
    std::cout << "***************************\n";
    print_weight();

    // back propagate
    TreeNode* sib = child1;
    int i = 0;
    while (sib != 0)
    {
        cout << "+= " << sib->data->name << "\n";
        //TODO: handle nodes that accomodate backprop, but do not have 
        // weights (e.g. split nodes).  Idea: make a
        //backprop() fn that calls update_weights for NodeDx, and otherwise
        // passes the gradient through appropriately to child nodes
        sib->update_weights(d, this->derivative(gradient, i));
        sib = sib->next_sibling;
        ++i;
    }
}

void NodeDx::print_weight()
{
    std::cout << this->name << "| W has value";
    for (int i = 0; i < this->arity['f']; i++) {
        std::cout << " " << this->W[i];
    }
    std::cout << "\n";
}

}
}
}
