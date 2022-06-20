#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.name;
    return os;
};
}
