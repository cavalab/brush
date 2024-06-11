#include "selection_operator.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;

template<ProgramType T> 
SelectionOperator<T>::~SelectionOperator(){};
    
template<ProgramType T> 
vector<size_t> SelectionOperator<T>::select(Population<T>& pop, int island, 
        const Parameters& p)
{   
    HANDLE_ERROR_THROW("Undefined select() operation");
    return vector<size_t>();
};

template<ProgramType T> 
vector<size_t> SelectionOperator<T>::survive(Population<T>& pop, int island, 
        const Parameters& p)
{   
    HANDLE_ERROR_THROW("Undefined select() operation");
    return vector<size_t>();
};

} // selection
} // Brush