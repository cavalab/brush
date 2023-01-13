#ifndef TREE_NODE_H
#define TREE_NODE_H
#include <tuple>
#include <unordered_map>

#include "../init.h"
#include "../data/data.h"
#include "node.h"
#include "functions.h"
#include "nodemap.h"
#include "../../thirdparty/tree.hh"

using std::string;
using Brush::Data::Dataset;
using Brush::Node;

/**
  * @brief tree node specialization for Node.
  * 
*/
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

        template<typename T>
        auto fit(const Dataset& d); 

        template<typename T> 
        auto predict(const Dataset& d, const float** weights=nullptr); 

        template<typename T, typename W>
        auto predict(const Dataset& d, const W** weights); 

		string get_model(bool pretty=false) const;
		string get_tree_model(bool pretty=false, string offset="") const;
}; 
using TreeNode = class tree_node_<Node>; 

//////////////////////////////////////////////////////////////////////////////////
// fit, eval, predict

#include "dispatch_table.h"

template<typename T>
auto TreeNode::fit(const Dataset& d)
{ 
    auto F = dtable_fit.template Get<T>(data.node_type, data.sig_hash);
    return F(d, (*this));
};

template<typename T> 
auto TreeNode::predict(const Dataset& d, const float** weights)
{ 
    auto F = dtable_predict.template Get<T>(data.node_type, data.sig_hash);
    return F(d, (*this), weights);
};

template<typename T, typename W> 
auto TreeNode::predict(const Dataset& d, const W** weights)
{ 
    auto F = dtable_predict.template Get<T>(data.node_type, data.sig_dual_hash);
    return F(d, (*this), weights);
};

// serialization functions
void to_json(json &j, const tree<Node> &t);
void from_json(const json &j, tree<Node> &t);
#endif
