#include "selection.h"

namespace selection {

Selection::Selection()
{
    /*!
     * set type of selection operator.
     */
    this->type = "lexicase";
    this->survival = false;
    this->set_operator();
}

Selection::Selection(string type, bool survival)
{
    /*!
     * set type of selection operator.
     */
    this->type = type;
    this->survival = survival;
    this->set_operator();
}

void Selection::set_operator()
{
    // if (this->type == "lexicase")
    //     pselector = std::make_shared<Lexicase>(survival); 
    // else if (this->type == "fair_lexicase")
    //     pselector = std::make_shared<FairLexicase>(survival);
    // else if (this->type == "pareto_lexicase")
    //     pselector = std::make_shared<ParetoLexicase>(survival);
    // else if (this->type == "nsga2")
    //     pselector = std::make_shared<NSGA2>(survival);
    // else if (this->type == "tournament")
    //     pselector = std::make_shared<Tournament>(survival);
    // else if (this->type == "offspring")    // offspring survival
    //     pselector = std::make_shared<Offspring>(survival);
    // else if (this->type == "random")    // offspring survival
    //     pselector = std::make_shared<Random>(survival);
    // else if (this->type == "simanneal")    // offspring survival
    //     pselector = std::make_shared<SimAnneal>(survival);
    // else
    //     WARN("Undefined Selection Operator " + this->type + "\n");
        
}

Selection::~Selection(){}

/// return type of selectionoperator
string Selection::get_type(){ return pselector->name; }

/// set type of selectionoperator
void Selection::set_type(string in){ type = in; set_operator();}

/// perform selection 
vector<size_t> Selection::select()
{       
    return pselector->select(pop, params, d);
}

/// perform survival
vector<size_t> Selection::survive(
{       
    return pselector->survive(pop, params, d);
}


SelectionOperator::~SelectionOperator(){}

vector<size_t> SelectionOperator::select() 
{   
    THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}

vector<size_t> SelectionOperator::survive()
{
    THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}

} // selection