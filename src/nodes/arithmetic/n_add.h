/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_ADD
#define NODE_ADD
//external includes
// internal includes
#include "../n_dx.h"
using std::cout;
using std::string;

namespace BR {

class NodeAdd : public NodeDx
{
    public:
    
        NodeAdd(vector<float> W0 = vector<float>());
                    
        /// Evaluates the node and updates the state states. 
        State evaluate(const Data& d, NodeTree* child1=0, 
                       NodeTree* child2=0, bool train=false)

        ArrayXf getDerivative(Trace& state, int loc);
};

}
#endif
