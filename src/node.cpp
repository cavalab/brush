#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.name;
    return os;
};
auto Node::get_name() const noexcept -> std::string const& { return name; };
}
