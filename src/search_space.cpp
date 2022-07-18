#include "search_space.h"

namespace Brush{

NodeVector generate_terminals(const Data& d)
{
    NodeVector terminals;
    int i = 0;
    for ( const auto &[name, value]: d.features ) 
    {
        cout << "name: " << name << endl;
        cout << "get data type\n";
        auto data_type = d.data_types.at(i);
        cout << "push back terminal\n";
        // note: structured bindings cannot be captured by lambdas until C++20
        /* std::cout << "generating terminal " << key << "of type " << value << endl; */
        // terminals.push_back(make_shared<Terminal>(var, d[var]));

        terminals.push_back(Node(NodeType::Terminal, data_type, name));
        ++i;
    };
    return terminals;
};

} //Brush
