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
#include "../tree.h"
#include "../data/data.h"
#include "../state.h"
using std::cout;
using std::string;

namespace Brush {

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
                               tree_node_<Node*>* first_child=0, 
                               tree_node_<Node*>* last_child=0) = 0;
};

typedef tree_node_<Node*> TreeNode;  

}
#endif


class NodeTimes : public Node
{
    public:
        NodeTimes(){name = "times";}
        State evaluate(const Data& d, tree_node_<Node*>* first_child=0, 
                tree_node_<Node*>* last_child=0)
        {
            /* return  first_child->eval(x) * last_child->eval(x); */
            State s1 = first_child->eval(d);
            State s2 = last_child->eval(d);
            State s3;
            s3.set<float>(s1.get_data<float>() * s2.get_data<float>());
            return  s3;
        }
};

class NodeSum : public Node
{
    public:
        NodeSum(){name = "sum";}
        State evaluate(const Data& d, tree_node_<Node*>* first_child=0, 
                tree_node_<Node*>* last_child=0)
        {
            ArrayXf sum(d.X.cols());
            tree_node_<Node*>* sib = first_child;
            while (sib != 0)
            {
                cout << "+= " << sib->data->name << "\n";
                sum += sib->eval(d).get_data<float>();
                sib = sib->next_sibling;
            }
            State s; 
            s.set<float>(sum);
            return  s;
        }
};
