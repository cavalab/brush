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
    {NodeType::Acos    , 5},
    {NodeType::Asin    , 5},
    {NodeType::Atan    , 5},
    {NodeType::Cos     , 5},
    {NodeType::Cosh    , 5},
    {NodeType::Sin     , 5},
    {NodeType::Sinh    , 5},
    {NodeType::Tan     , 5},
    {NodeType::Tanh    , 5},
    {NodeType::Ceil    , 4},
    {NodeType::Floor   , 4},
    {NodeType::Exp     , 4},
    {NodeType::Log     , 4},
    {NodeType::Logabs  , 12},
    {NodeType::Log1p   , 8},
    {NodeType::Sqrt    , 4},
    {NodeType::Sqrtabs , 4},
    {NodeType::Square  , 3},
    {NodeType::Logistic, 3},

    // timing masks
    {NodeType::Before, 3},
    {NodeType::After , 3},
    {NodeType::During, 3},

    // Reducers
    {NodeType::Min   , 3},
    {NodeType::Max   , 3},
    {NodeType::Mean  , 3},
    {NodeType::Median, 3},
    {NodeType::Sum   , 2},
    {NodeType::Prod  , 3},

    // Transformers 
    {NodeType::Softmax, 4},

    // Binary
    {NodeType::Add, 2},
    {NodeType::Sub, 2},
    {NodeType::Mul, 3},
    {NodeType::Div, 4},
    {NodeType::Pow, 5},

    //split
    {NodeType::SplitBest, 4},
    {NodeType::SplitOn  , 4},

    // boolean
    {NodeType::And, 2},
    {NodeType::Or , 2},
    {NodeType::Not, 2},

    // leaves
    {NodeType::MeanLabel, 1},
    {NodeType::Constant , 1},
    {NodeType::Terminal , 2},
    {NodeType::ArgMax   , 5},
    {NodeType::Count    , 3},
    
    // custom
    {NodeType::CustomUnaryOp , 5},
    {NodeType::CustomBinaryOp, 5},
    {NodeType::CustomSplit   , 5}
};

int TreeNode::get_complexity() const 
{
    int node_complexity = operator_complexities.at(data.node_type);
    int children_complexity_sum = 0; // acumulator for children complexities

    auto child = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        children_complexity_sum += child->get_complexity();
        child = child->next_sibling;
    }

    // avoid multiplication by zero if the node is a terminal
    children_complexity_sum = max(children_complexity_sum, 1);

    if (data.get_is_weighted()) // include the `w` and `*` if the node is weighted
        return operator_complexities.at(NodeType::Mul)*(
            operator_complexities.at(NodeType::Constant) + 
            node_complexity*(children_complexity_sum)
        );

    return node_complexity*(children_complexity_sum);
}