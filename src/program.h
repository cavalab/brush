/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef PROGRAM_H
#define PROGRAM_H
//external includes
//
#include <iostream>
#include <string>
// internal includes
#include "tree.h"
#include "data.h"
#include "state.h"
#include "template_nodes.h"

using std::cout;
using std::string;
using BR::State;
using BR::Dat::Data;

namespace BR {

class Program : public tree<NodeBase*>
{
    
    public:
      
        // constructs a tree using functions, terminals, and settings
        void make_program(const vector<NodeBase*> & functions, 
                         const vector<NodeBase*> & terminals, 
                         int max_d,  
                         const vector<float>& term_weights, 
                         const vector<float>& op_weights, 
                         char otype, 
                         const vector<char>& term_types){};

        Program mutate(){}; 
        Program cross(Program& other){}; 
       
        State fit(const Data& d)
		{
			/* auto start = this->begin(); */
            iterator start = this->begin(); 
            /* tree_node_<NodeBase*> root = (*start).fit(d); */
            State out = start.node->fit(d);
			return out;
		};


};
}
#endif
