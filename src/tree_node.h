#ifndef TREE_NODE_H
#define TREE_NODE_H
#include <tuple>
#include <unordered_map>

#include "init.h"
#include "data/data.h"
#include "node.h"
#include "functions.h"
#include "nodemap.h"
#include "dispatch_table.h"
#include "thirdparty/tree.hh"
/* #include "operator.h" */
/* #include "interpreter.h" */

using std::string;
using Brush::data::Data;
using Brush::ExecType;
using Brush::Node;

/* namespace Brush { */
/// A node in the tree, combining links to other nodes as well as the actual data.
/* template<class T> class tree_node_; */ 

// /**
//  * @brief tree node specialization for Node.
//  * 
//  */
//
template<>
class tree_node_<Node> { // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
	public:
        tree_node_()
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0)
            {}

        tree_node_(const Node& val)
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), data(val)
            {}

        tree_node_(Node&& val)
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), data(val)
            {}

		tree_node_<Node> *parent;
	    tree_node_<Node> *first_child, *last_child;
		tree_node_<Node> *prev_sibling, *next_sibling;
		Node data;

        /* template<typename T> */
        template<typename T>
        auto fit(const Data& d); //{ State s; return std::get<T>(s);};
        template<typename T>
        auto predict(const Data& d); //{ State s; return std::get<T>(s);};
        /* /1* void grad_descent(const ArrayXf&, const Data&); *1/ */
		string get_model(bool pretty=false) const;
		string get_tree_model(bool pretty=false, string offset="") const;
}; 
using TreeNode = class tree_node_<Node>; 

//////////////////////////////////////////////////////////////////////////////////
// fit, eval, predict

template<typename T>
auto TreeNode::fit(const Data& d)
{ 
    fmt::print("Getting {}({})\n",data.node_type, data.sig_hash);
    auto F = dtable_fit.template Get<T>(data.node_type, data.sig_hash);
    fmt::print("return F(d,(*this))\n");
    return F(d, (*this));
};

template<typename T>
auto TreeNode::predict(const Data& d)
{ 
    auto F = dtable_predict.template Get<T>(data.node_type, data.sig_hash);
    return F(d, (*this));
};

/* }// Brush */
#endif
