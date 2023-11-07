/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "../init.h"
#include "../params.h"
#include "../population.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;

/*!
 * @class SelectionOperator
 * @brief base class for selection operators.
 */ 
class SelectionOperator 
{
public:
    bool survival; 
    string name;

    SelectionOperator(){}

    virtual ~SelectionOperator();
    
    template<ProgramType T> 
    vector<size_t> select(Population<T>& pop,  
            const Parameters& p, const Dataset& data)
    {   
        // THROW_INVALID_ARGUMENT("Undefined select() operation");
        return vector<size_t>();
    }
    
    template<ProgramType T>
    vector<size_t> survive(Population<T>& pop,  
            const Parameters& p, const Dataset& data)
    {   
        // THROW_INVALID_ARGUMENT("Undefined select() operation");
        return vector<size_t>();
    }
};

struct Parameters; // forward declaration of Parameters      

/*!
* @class Selection
* @brief interfaces with selection operators. 
*/
struct Selection
{
public:
    shared_ptr<SelectionOperator> pselector; 
    string type;
    bool survival;
    
    Selection(); 
    ~Selection();
    Selection(string type, bool survival);

    void set_operator();
    
    /// return type of selectionoperator
    string get_type();
    void set_type(string);
    
    /// perform selection 
    template<ProgramType T> 
    vector<size_t> select(Population<T>& pop,  
            const Parameters& params, const Dataset& data);
    
    /// perform survival
    template<ProgramType T> 
    vector<size_t> survive(Population<T>& pop,  
            const Parameters& params, const Dataset& data);
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Selection, type, survival);    

} // selection
} // Brush
#endif