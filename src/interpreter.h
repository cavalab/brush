/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"

namespace Brush{

template<typename T>
struct GetChildren
{
    using TreeNode = tree_node_<Node>;
    template <std::size_t N>
    using NthType = typename std::tuple_element<N, T>::type;

    T operator()(const Data& d, State (TreeNode::*fn)(const Data&))
    {
        // why not make get children return the tuple?
        // use get<NthType<i>> to get the type for it
        T child_outputs;

        TreeNode* sib = first_child;
        for (int i = 0; i < this->get_arg_count(); ++i)
        {
            /* std::get<i>(child_outputs) = std::get<NthType<i>>((sib->*fn)(d)); */
            child_outputs.at(i) = (sib->*fn)(d);
            sib = sib->next_sibling;
        }
        return child_outputs;

    }
}

auto get_children(const Data& d, auto (TreeNode::*fn)(const Data&))
{
    // why not make get children return the tuple?
    // use get<NthType<i>> to get the type for it
    auto signature = NodeSchema[data.node_type]["Signature"][data.ret_type]; 
    auto child_outputs = GetChildren<decltype(signature)>(d);

    TreeNode* sib = first_child;
    for (int i = 0; i < data.get_arg_count(); ++i)
    {
        // std::get<i>(child_outputs) = std::get<NthType<i>>((sib->*fn)(d));
        child_outputs.at(i) = (sib->*fn)(d);
        sib = sib->next_sibling;
    }
    /* return tupleize(child_outputs); */
    return child_outputs;
    
};;

TupleArgs get_children_fit(const Data& d)
{
    return get_children(d, &TreeNode::fit);
};

TupleArgs get_children_predict(const Data& d)
{
    return get_children(d, &TreeNode::predict);
};

}
#endif
