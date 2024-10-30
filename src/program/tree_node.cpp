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
        child_outputs += offset + "|- ";
        string s = sib->get_tree_model(pretty, offset+new_offset);
        sib = sib->next_sibling;
        // if (sib == nullptr)
            ReplaceStringInPlace(s, "\n"+offset, "\n"+offset+"|") ;
        child_outputs += s;
        if (sib != nullptr)
            child_outputs += "\n";
    }
    
    if (Is<NodeType::SplitBest>(data.node_type)){
        if (data.arg_types.at(0) == DataType::ArrayB)
            return fmt::format("If({})", data.get_feature()) + child_outputs;

        return fmt::format("If({}>{:.2f})", data.get_feature(), data.W) +
               child_outputs;
    }
    else if (Is<NodeType::SplitOn>(data.node_type)){
        if (data.arg_types.at(0) == DataType::ArrayB)
        {
            // booleans dont use thresholds (they are used directly as mask in split)
            return "If" + child_outputs;
        }
        // integers or floating points (they have a threshold)
        return fmt::format("If(>{:.2f})", data.W) + child_outputs;
    }
    else{
        return data.get_name() + child_outputs;
    }
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
    {NodeType::Abs     ,  4},
    {NodeType::Acos    ,  6},
    {NodeType::Asin    ,  6},
    {NodeType::Atan    ,  6},
    {NodeType::Cos     ,  6},
    {NodeType::Cosh    ,  6},
    {NodeType::Sin     ,  6},
    {NodeType::Sinh    ,  6},
    {NodeType::Tan     ,  6},
    {NodeType::Tanh    ,  6},
    {NodeType::Ceil    ,  5},
    {NodeType::Floor   ,  5},
    {NodeType::Exp     ,  5},
    {NodeType::Log     ,  5},
    {NodeType::Logabs  ,  10},
    {NodeType::Log1p   ,  9},
    {NodeType::Sqrt    ,  5},
    {NodeType::Sqrtabs ,  5},
    {NodeType::Square  ,  4},
    {NodeType::Logistic,  4},
    {NodeType::OffsetSum, 3},

    // timing masks
    {NodeType::Before, 4},
    {NodeType::After , 4},
    {NodeType::During, 4},

    // Reducers
    {NodeType::Min      , 4},
    {NodeType::Max      , 4},
    {NodeType::Mean     , 4},
    {NodeType::Median   , 4},
    {NodeType::Sum      , 4},
    {NodeType::Prod     , 4},

    // Transformers 
    {NodeType::Softmax, 5},

    // Binary
    {NodeType::Add, 3},
    {NodeType::Sub, 3},
    {NodeType::Mul, 4},
    {NodeType::Div, 5},
    {NodeType::Pow, 5},

    //split
    {NodeType::SplitBest, 4},
    {NodeType::SplitOn  , 4},

    // boolean
    {NodeType::And, 3},
    {NodeType::Or , 3},
    {NodeType::Not, 3},

    // leaves
    {NodeType::MeanLabel, 1},
    {NodeType::Constant , 2},
    {NodeType::Terminal , 3},
    {NodeType::ArgMax   , 5},
    {NodeType::Count    , 4},
    
    // custom
    {NodeType::CustomUnaryOp , 5},
    {NodeType::CustomBinaryOp, 5},
    {NodeType::CustomSplit   , 5}
};

int TreeNode::get_linear_complexity() const 
{
    int tree_complexity = operator_complexities.at(data.node_type);

    auto child = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        tree_complexity += child->get_linear_complexity();
        child = child->next_sibling;
    }

    // include the `w` and `*` if the node is weighted (and it is not a constant or mean label)
    if (data.get_is_weighted()
    &&  Isnt<NodeType::Constant, NodeType::MeanLabel>(data.node_type) )
    {
        // ignoring weight if it has the value of neutral element of operation
        if ((Is<NodeType::OffsetSum>(data.node_type) && data.W != 0.0)
        ||  (data.W != 1.0))
            return operator_complexities.at(NodeType::Mul) +
                operator_complexities.at(NodeType::Constant) + 
                tree_complexity;
    }
    
    return tree_complexity;
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

    // include the `w` and `*` if the node is weighted (and it is not a constant or mean label)
    if (data.get_is_weighted()
    &&  Isnt<NodeType::Constant, NodeType::MeanLabel>(data.node_type) )
    {
        // ignoring weight if it has the value of neutral element of operation
        if ((Is<NodeType::OffsetSum>(data.node_type) && data.W != 0.0)
        ||  (data.W != 1.0))
            return operator_complexities.at(NodeType::Mul)*(
                       operator_complexities.at(NodeType::Constant) + 
                       node_complexity*(children_complexity_sum)
                   );
    }

    return node_complexity*(children_complexity_sum);
};

int TreeNode::get_size(bool include_weight) const 
{
    int acc = 1; // the node operator or terminal

    // SplitBest has an optimizable decision tree consisting of 3 nodes
    // (terminal, arithmetic comparison, value) that needs to be taken
    // into account. Split on will have an random decision tree that can 
    // have different sizes, but will also have the arithmetic comparison
    // and a value.
    if (Is<NodeType::SplitBest>(data.node_type))
        acc += 3;
    else if (Is<NodeType::SplitOn>(data.node_type))
        acc += 2;

    if ( (include_weight && data.get_is_weighted()==true)
    &&   Isnt<NodeType::Constant, NodeType::MeanLabel>(data.node_type) )
        // Taking into account the weight and multiplication, if enabled.
        // weighted constants still count as 1 (simpler than constant terminals)
        acc += 2;

    auto child = first_child;
    for(int i = 0; i < data.get_arg_count(); ++i)
    {
        acc += child->get_size(include_weight);
        child = child->next_sibling;
    }

    return acc;
};
