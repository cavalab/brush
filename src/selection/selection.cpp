#include "selection.h"
#include "nsga2.h"

// TODO: organize all namespaces
namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;

template<ProgramType T> 
Selection<T>::Selection(string type, bool survival)
{
    /*!
     * set type of selection operator.
     */
    this->type = type;
    this->survival = survival;
    this->set_operator();
}

template<ProgramType T>
void Selection<T>::set_operator()
{
    if (this->type == "nsga2")
        pselector = new NSGA2<T>(survival);
    else
        HANDLE_ERROR_THROW("Undefined Selection Operator " + this->type + "\n");
        
}

/// return type of selectionoperator
template<ProgramType T> 
string Selection<T>::get_type(){ return pselector->name; }

/// set type of selectionoperator
template<ProgramType T> 
void Selection<T>::set_type(string in){ type = in; set_operator();}

/// perform selection 
template<ProgramType T> 
vector<size_t> Selection<T>::select(Population<T>& pop, int island,  
        const Parameters& params, const Dataset& data)
{       
    return pselector->select(pop, island, params, data);
}

/// perform survival
template<ProgramType T> 
vector<size_t> Selection<T>::survive(Population<T>& pop, int island, 
        const Parameters& params, const Dataset& data)
{       
    return pselector->survive(pop, island, params, data);
}

} // selection
} // Brush