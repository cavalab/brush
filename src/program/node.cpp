#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const NodeType& nt)
{
    os << NodeTypeName.at(nt);
    return os;
}

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.get_name();
    return os;
}

/// @brief get the name of the node. 
/// @return name 
auto Node::get_name() const noexcept -> std::string 
{ 
        
        if (Is<NodeType::Terminal>(node_type))
            return feature;
        else if (Is<NodeType::Constant>(node_type))
        {
            return fmt::format("{:.3f}",W.at(0));
        }
        else if (Is<NodeType::SplitBest>(node_type))
            return fmt::format("SplitBest[{}>{:.3f}]", feature, W.at(0)); 
        else if (Is<NodeType::SplitBest>(node_type))
            return fmt::format("SplitOn[{:.3f}]", W.at(0)); 
        else if (Is<NodeType::SplitOn>(node_type))
        {
            fmt::print("SplitOn[{}]", arg_types); 
            return fmt::format("SplitOn[{}]", arg_types); 
        }
        else
            return name;
}



}