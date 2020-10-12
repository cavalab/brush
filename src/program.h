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
#include "nodes/node.h"
using std::cout;
using std::string;
using BR::Dat::State;
using BR::Dat::Data;

namespace BR {
class NodeVector : public vector<Node*>
{
    //TODO: destructor class that de-allocates
    ~NodeVector()
    {
       for (NodeVector::iterator pObj = this->begin();
            pObj != this->end(); ++pObj) 
       {
           delete *pObj; // Note that this is deleting what pObj points 
                         // to, which is a pointer
       }

       this->clear(); // Purge the contents so no one tries to delete them
                    // again
    }
};

class Program: public tree<Node*>
{
    public:
      
        // constructs a tree using functions, terminals, and settings
        void make_program(const NodeVector& functions, 
                         const NodeVector& terminals, 
                         int max_d,  
                         const vector<float>& term_weights, 
                         const vector<float>& op_weights, 
                         char otype, 
                         const vector<char>& term_types){};

        Program mutate(){}; 
        Program cross(Program& other){}; 
       

};
}
#endif
