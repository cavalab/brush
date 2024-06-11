#ifndef SELECTION_OPERATOR_H
#define SELECTION_OPERATOR_H

// virtual class. selection must be made with static methods

// #include "../init.h"
// #include "../data/data.h"
// #include "../types.h"
// #include "../params.h"
#include "../pop/population.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;

/*!
 * @class SelectionOperator
 * @brief base class for selection operators.
 */ 
template<ProgramType T> 
/**
 * @brief The SelectionOperator class represents a base class for selection operators in a genetic algorithm.
 * 
 * This class provides common functionality and interface for selection operators.
 */
class SelectionOperator
{
public:
    bool survival; /**< Flag indicating whether the selection operator is used for survival selection. */
    string name; /**< The name of the selection operator. */

    /**
     * @brief Destructor for the SelectionOperator class.
     */
    virtual ~SelectionOperator();
        
    /**
     * @brief Selects individuals from the population based on the selection operator's strategy.
     * 
     * @param pop The population from which to select individuals.
     * @param island The index of the island in a parallel genetic algorithm.
     * @param p The parameters for the selection operator.
     * @return A vector of indices representing the selected individuals.
     */
    virtual vector<size_t> select(Population<T>& pop, int island, const Parameters& p);
    
    /**
     * @brief Applies the selection operator to determine which individuals survive in the population.
     * 
     * @param pop The population in which to apply the survival selection.
     * @param island The index of the island in a parallel genetic algorithm.
     * @param p The parameters for the selection operator.
     * @return A vector of indices representing the surviving individuals.
     */
    virtual vector<size_t> survive(Population<T>& pop, int island, const Parameters& p);
};

} // selection
} // Brush
#endif
