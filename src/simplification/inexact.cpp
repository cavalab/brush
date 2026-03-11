#include "inexact.h"
#include "../util/rnd.h"

using Brush::Util::r;

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
    inputDim = std::min(inputDim, data.get_training_data().get_n_samples());

    uniformPlanes.clear();
    for (int i=0; i<numPlanes; ++i)
    {
        // Use deterministic random number generator instead of Eigen::Random()
        // to ensure reproducibility with seed
        MatrixXf plane(hashSize, inputDim);
        for (int row = 0; row < hashSize; ++row) {
            for (int col = 0; col < inputDim; ++col) {
                plane(row, col) = r.rnd_flt(-1.0f, 1.0f);
            }
        }
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
