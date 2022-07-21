#include "tree_node.h"

namespace Brush {

/* const DispatchTable< */
/*               ArrayXf */
/*               /1* ArrayXb, *1/ */
/*               /1* ArrayXi, *1/ */ 
/*               /1* ArrayXf, *1/ */ 
/*               /1* ArrayXXb, *1/ */
/*               /1* ArrayXXi, *1/ */ 
/*               /1* ArrayXXf, *1/ */ 
/*               /1* TimeSeriesb, *1/ */
/*               /1* TimeSeriesi, *1/ */
/*               /1* TimeSeriesf *1/ */
/*              > dtable; */
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
    if (n.get_arg_count()==0)
        return n.get_name();

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
