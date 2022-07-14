#include "tree_node.h"

namespace Brush {

DispatchTable<
              ArrayXf
              /* ArrayXb, */
              /* ArrayXi, */ 
              /* ArrayXf, */ 
              /* ArrayXXb, */
              /* ArrayXXi, */ 
              /* ArrayXXf, */ 
              /* TimeSeriesb, */
              /* TimeSeriesi, */
              /* TimeSeriesf */
             > dtable;
        /* template<typename T> */
        /* auto tree_node_<Node>::eval(const Data& d) */
        /* { */ 
        /*     auto F = DTable.TryGet<T>(n); */
        /*     return F(d, n); */
        /* }; */
        /* template<typename T> */
        /* auto tree_node_<Node>::fit(const Data& d){ State s; return std::get<T>(s);}; */
        /* template<typename T> */
        /* auto tree_node_<Node>::predict(const Data& d){ State s; return std::get<T>(s);}; */

string TreeNode::get_model(bool pretty) const 
{ 
    string child_outputs = "";
    auto sib = first_child;
    for(int i = 0; i < this->n.get_arg_count(); ++i)
    {
        child_outputs += sib->get_model(pretty);
        sib = sib->next_sibling;
        if (sib != nullptr)
            child_outputs += ",";
    }
    /* if (pretty) */
    /*     return this->n.op_name + "(" + child_outputs + ")"; */
    /* else */
    return this->n.name + "(" + child_outputs + ")";
};

string TreeNode::get_tree_model(bool pretty, string offset) const 
{ 
    // cout << "TypedNodeBase::get_tree_model. ";
    // cout << "first_child: " << first_child << endl;
    // cout << "last_child: " << last_child << endl;
    string new_offset = "  ";

    string  child_outputs = "\n";

    auto sib = first_child;
    for(int i = 0; i < this->n.get_arg_count(); ++i)
    {
        child_outputs += offset + "|-";
        string s = sib->get_tree_model(pretty, offset+new_offset);
        sib = sib->next_sibling;
        if (sib != nullptr)
            ReplaceStringInPlace(s, "\n"+offset, "\n"+offset+"|") ;
        child_outputs += s;
        if (sib != nullptr)
            child_outputs += "\n";
    }
    /* if (pretty) */
    /*     return this->op_name + child_outputs; */
    /* else */
        return this->n.name + child_outputs;
};
}
