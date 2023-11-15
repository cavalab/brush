/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "../init.h"
#include "../params.h"
#include "../types.h"
#include "../population.h"
#include "../variation.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Var;

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

    virtual ~SelectionOperator(){};
     
    virtual vector<size_t> select(Population<T>& pop, tuple<size_t, size_t> island_range, 
            const Parameters& p, const Dataset& data)
    {   
        HANDLE_ERROR_THROW("Undefined select() operation");
        return vector<size_t>();
    };
    
    virtual vector<size_t> survive(Population<T>& pop, tuple<size_t, size_t> island_range, 
            const Parameters& p, const Dataset& data)
    {   
        HANDLE_ERROR_THROW("Undefined select() operation");
        return vector<size_t>();
    };
};

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
    
    //TODO: rewrite it as initializing parameters
    Selection()
    {
        this->type = "nsga2";
        this->survival = false;
        this->set_operator();
    };

    ~Selection(){};
    Selection(string type, bool survival);

    void set_operator();
    
    /// return type of selectionoperator
    string get_type();
    void set_type(string);
    
    /// perform selection. selection uses a pop that has no offspring space
    vector<size_t> select(Population<T>& pop, tuple<size_t, size_t> island_range, 
            const Parameters& params, const Dataset& data);
    
    /// perform survival. uses a pop with offspring space
    vector<size_t> survive(Population<T>& pop, tuple<size_t, size_t> island_range,  
            const Parameters& params, const Dataset& data);
};

// TODO: MAKE THIS WORK
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Selection, type, survival);    

} // selection
} // Brush
#endif