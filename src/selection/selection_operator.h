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
class SelectionOperator 
{
public:
    bool survival; 
    string name;

    // shoudn't have a constructor
    // SelectionOperator(){};

    virtual ~SelectionOperator();
     
    virtual vector<size_t> select(Population<T>& pop, int island, 
            const Parameters& p);
    
    virtual vector<size_t> survive(Population<T>& pop, int island, 
            const Parameters& p);
};

} // selection
} // Brush
#endif
