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

unordered_map<NodeType, int> operator_complexities = {
    // Unary
    {NodeType::Abs     , 3},
    {NodeType::Acos    , 3},
    {NodeType::Asin    , 3},
    {NodeType::Atan    , 3},
    {NodeType::Cos     , 3},
    {NodeType::Cosh    , 3},
    {NodeType::Sin     , 3},
    {NodeType::Sinh    , 3},
    {NodeType::Tan     , 3},
    {NodeType::Tanh    , 3},
    {NodeType::Ceil    , 3},
    {NodeType::Floor   , 3},
    {NodeType::Exp     , 3},
    {NodeType::Log     , 3},
    {NodeType::Logabs  , 3},
    {NodeType::Log1p   , 3},
    {NodeType::Sqrt    , 3},
    {NodeType::Sqrtabs , 3},
    {NodeType::Square  , 3},
    {NodeType::Logistic, 3},

    // timing masks
    {NodeType::Before, 2},
    {NodeType::After , 2},
    {NodeType::During, 2},

    // Reducers
    {NodeType::Min   , 4},
    {NodeType::Max   , 4},
    {NodeType::Mean  , 4},
    {NodeType::Median, 4},
    {NodeType::Sum   , 4},
    {NodeType::Prod  , 4},

    // Transformers 
    {NodeType::Softmax, 4},

    // Binary
    {NodeType::Add, 1},
    {NodeType::Sub, 1},
    {NodeType::Mul, 1},
    {NodeType::Div, 1},
    {NodeType::Pow, 1},

    //split
    {NodeType::SplitBest, 2},
    {NodeType::SplitOn  , 2},

    // boolean
    {NodeType::And, 1},
    {NodeType::Or , 1},
    {NodeType::Not, 1},

    // leaves
    {NodeType::MeanLabel, 1},
    {NodeType::Constant , 1},
    {NodeType::Terminal , 2},
    {NodeType::ArgMax   , 2},
    {NodeType::Count    , 2},
    
    // custom
    {NodeType::CustomUnaryOp , 5},
    {NodeType::CustomBinaryOp, 5},
    {NodeType::CustomSplit   , 5}
};

int TreeNode::get_complexity() const 
{
    int node_complexity = operator_complexities.at(data.node_type);
    int children_complexity = 0;

    auto child = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        children_complexity += child->get_complexity();
        child = child->next_sibling;
    }
    
    return node_complexity*children_complexity;
}