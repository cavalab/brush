#ifndef LEXICASE_H
#define LEXICASE_H

#include "selection_operator.h"
#include "../util/utils.h"


namespace Brush {
namespace Sel {


using namespace Brush;
using namespace Pop;
using namespace Sel;


////////////////////////////////////////////////////////////// Declarations
/*!
* @class Lexicase
* @brief Lexicase selection operator.
*/

template<ProgramType T> 
class Lexicase : public SelectionOperator<T>
{
public:
    Lexicase(bool surv=false);
    ~Lexicase(){};

    /// function returns a set of selected indices from pop 
    vector<size_t> select(Population<T>& pop, int island,
            const Parameters& p);
    
    /// lexicase survival
    vector<size_t> survive(Population<T>& pop, int island, 
            const Parameters& p);
};


}
}

#endif