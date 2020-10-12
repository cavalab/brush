/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_H
#define NODE_H
//external includes
//
#include <iostream>
#include <string>
// internal includes
#include "../node.h"
using std::cout;
using std::string;

namespace BR {

class NodeVariable : public Node
{
    public:
        int index;
        
        NodeVariable(){name = "x"; this->index = 1;}
        NodeVariable(int value){name = "x"; this->index = value;}

        State evaluate(const Data& d, TreeNode* child1=0, 
                TreeNode* child2=0)
        {
            
            /* return d.get<T>(this->value); */
            State s;
            s.set<float>(d.X.row(this->index));
            return s;
        }

        void swap(Node& b)
        {
            using std::swap;
            Node::swap(b);
            swap(this->index, b.index);
        }
}
class Node
{
    public:
        // name of the node
        string name;
        // sample probability of this node
        float probability;

        Node(string name)
        {
            this->name = name;
            probability = 1.0;
        }
        Node() = default;
        ~Node() = default;
        Node(const Node&) = default;
        //TODO: implement this
        Node& operator=(Node && other) = default;

        //TODO: revisit this
        /* bool operator==(Node && other){return false;}; */
        bool operator==(const Node & other){return this->name==other.name;};

        // note this can be called in derived classes via Node::swap(b)
        void swap(Node& b)
        {
            using std::swap;
            swap(this->name,b.name);
            swap(this->probability,b.probability);
        }

        virtual State evaluate(const Data& d, 
                               tree_node_<Node*>* child1=0, 
                               tree_node_<Node*>* child2=0) = 0;
};
}
#endif
