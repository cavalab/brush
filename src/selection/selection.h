/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "selection_operator.h"
#include "nsga2.h"
#include "lexicase.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;

// struct Parameters; // forward declaration of Parameters      

/*!
* @class Selection
* @brief interfaces with selection operators. 
*/
template<ProgramType T>
struct Selection
{
public:
    SelectionOperator<T>* pselector; // TODO: THIS SHOULD BE A SHARED POINTER 
    string type;
    bool survival;
    
    Selection();
    ~Selection(){};
    Selection(string type, bool survival);

    void set_operator();
    
    /// return type of selectionoperator
    string get_type();
    void set_type(string);
    
    /// perform selection. selection uses a pop that has no offspring space
    vector<size_t> select(Population<T>& pop, int island, 
            const Parameters& params);
    
    /// perform survival. uses a pop with offspring space
    vector<size_t> survive(Population<T>& pop, int island,  
            const Parameters& params);
};

// TODO: MAKE THIS WORK
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Selection, type, survival);    

} // selection
} // Brush
#endif