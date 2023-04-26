#include "tree_node.h"


string TreeNode::get_model(bool pretty) const 
{ 
    if (data.get_arg_count()==0)
        return data.get_name();

    vector<string> child_outputs;
    auto sib = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        child_outputs.push_back(sib->get_model(pretty));
        sib = sib->next_sibling;
    }
    return data.get_model(child_outputs);
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
////////////////////////////////////////////////////////////////////////////////
// serialization for tree
void to_json(json &j, const tree<Node> &t)
{
    j.clear();
    // for (auto iter = t.begin(); iter!=t.end(); ++iter)
    for (const auto &el : t)
    {
        j.push_back(el);
    }
}

/// @brief load a tree from json. uses a stack. 
/// @param j json version of tree.
/// @param t tree into which data is loaded.
void from_json(const json &j, tree<Node> &t)
{
    vector<tree<Node>> stack; 
    for (int i = j.size(); i --> 0; )
    {
        auto node = j.at(i).get<Node>();
        tree<Node> subtree;
        auto root = subtree.insert(subtree.begin(), node);
        for (auto at : node.arg_types)
        {
            auto spot = subtree.append_child(root);
            auto arg = stack.back();
            subtree.move_ontop(spot, arg.begin());
            stack.pop_back();
        }
        stack.push_back(subtree);
    }
    t = stack.back();
}