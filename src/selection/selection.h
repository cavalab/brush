/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "../init.h"
#include "../params.h"
#include "../population.h"

namespace selection {

/*!
 * @class SelectionOperator
 * @brief base class for selection operators.
 */ 
struct SelectionOperator 
{
    bool survival; 
    string name;

    //SelectionOperator(){}

    virtual ~SelectionOperator();
    
    virtual vector<size_t> select();
    
    virtual vector<size_t> survive();
};

struct Parameters; // forward declaration of Parameters      

/*!
* @class Selection
* @brief interfaces with selection operators. 
*/
struct Selection
{
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
    vector<size_t> select();
    
    /// perform survival
    vector<size_t> survive();
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Selection, type, survival);    

} // selection
#endif