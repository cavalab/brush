#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.name;
    return os;
}

auto Node::get_name() const noexcept -> std::string const& 
{ 
        if (Is<NodeType::Terminal>())
            return feature;
        else if (Is<NodeType::SplitBest>())
            return std::format("Split( {} > {:.3f})", feature, threshold); 
        else
            return name;
}
}
