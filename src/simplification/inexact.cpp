#include "inexact.h"


// simplification maps are based on trainingdata
// should ignore fixed nodes ---> does not change subtrees if they contain fixed nodes 
// should I implement json serialization?
        
namespace Brush { namespace Simpl{
    
// Inexact_simplifier* Inexact_simplifier::instance = NULL;
    
Inexact_simplifier::Inexact_simplifier()
{
}


void Inexact_simplifier::init(int hashSize, const Dataset &data, int numPlanes)
{
    // The greater the number of planes, the more accurate the hash, but the slower the search
    
    // cut-off for performance at 100 samples
    inputDim = std::min(1000, data.get_training_data().get_n_samples());

    uniformPlanes.clear();
    for (int i=0; i<numPlanes; ++i)
    {
        MatrixXf plane = MatrixXf::Random(hashSize, inputDim);
        // plane /= plane.norm();
        uniformPlanes.push_back(plane);
    }

    equivalentExpressions.clear();
    for (const auto& dtype : data.unique_data_types)
        equivalentExpressions[dtype] = HashStorage(numPlanes);
}


Inexact_simplifier::~Inexact_simplifier() {}

} // Simply
} // Brush
