#include "tree_node.h"


string TreeNode::get_model(bool pretty) const 
{ 
    if (data.get_arg_count()==0)
        return data.get_name();

    string child_outputs = "";
    auto sib = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        child_outputs += sib->get_model(pretty);
        sib = sib->next_sibling;
        if (sib != nullptr)
            child_outputs += ",";
    }
    /* if (pretty) */
    /*     return data.op_name + "(" + child_outputs + ")"; */
    /* else */
    return data.get_name() + "(" + child_outputs + ")";
};

string TreeNode::get_tree_model(bool pretty, string offset) const 
{ 
    if (data.get_arg_count()==0)
        return data.get_name();

    string new_offset = "  ";
    string  child_outputs = "\n";

    auto sib = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        child_outputs += offset + "|-";
        string s = sib->get_tree_model(pretty, offset+new_offset);
        sib = sib->next_sibling;
        if (sib == nullptr)
            ReplaceStringInPlace(s, "\n"+offset, "\n"+offset+"|") ;
        child_outputs += s;
        if (sib != nullptr)
            child_outputs += "\n";
    }
    /* if (pretty) */
    /*     return op_name + child_outputs; */
    /* else */
    return data.get_name() + child_outputs;
};