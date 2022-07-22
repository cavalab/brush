#include "search_space.h"

namespace Brush{

vector<Node> generate_terminals(const Data& d)
{
    vector<Node> terminals;
    int i = 0;
    for ( const auto &[name, value]: d.features ) 
    {
        auto data_type = d.feature_types.at(i);
        fmt::print("generating Terminal name: {}, data_type: {}",name, data_type);
        terminals.push_back(Node(NodeType::Terminal, data_type, name));
        ++i;
    };
    return terminals;
};

std::unordered_map<std::size_t, std::string> ArgsName; 

void SearchSpace::init(const Data& d, const unordered_map<string,float>& user_ops)
{
    fmt::print("constructing search space...\n");

    this->node_map.clear();
    this->weight_map.clear();
    this->terminal_map.clear();
    this->terminal_types.clear();
    this->terminal_weights.clear();

    bool use_all = user_ops.size() == 0;
    vector<string> op_names;
    for (const auto& [op, weight] : user_ops)
        op_names.push_back(op);


    // create nodes based on data types 
    terminal_types = d.unique_data_types;

    vector<Node> terminals = generate_terminals(d);
    
    fmt::print("generate nodemap\n");
    GenerateNodeMap(user_ops, d.unique_data_types, 
                    std::make_index_sequence<NodeTypes::OpCount>());
    // map terminals
    fmt::print("looping through terminals...\n");
    for (const auto& term : terminals)
    {
        fmt::print("adding {} to search space...\n", term.get_name());
        if (terminal_map.find(term.ret_type) == terminal_map.end())
            terminal_map[term.ret_type] = vector<Node>();
        fmt::print("terminal ret_type: {}\n", DataTypeName[term.ret_type]);
        terminal_map[term.ret_type].push_back(term);
        terminal_weights[term.ret_type].push_back(1.0);
    }

    fmt::print("{}\n", *this);

};

/// queue for make program
template<typename T>
T RandomDequeue(std::vector<T>& Q)
{
    int loc = r.rnd_int(0, Q.size()-1);
    std::swap(Q[loc], Q[Q.size()-1]);
    T val = Q.back();
    Q.pop_back();
    return val;
}
// constructs a tree using functions, terminals, and settings
template<typename T>
Program<T> SearchSpace::make_program(int max_d, int max_breadth, int max_size)
{
    /*
    * implementation of PTC2 for strongly typed GP from Luke et al. 
    * "Two fast tree-creation algorithms for genetic programming"
    *  
    */
    if (max_d == 0)
        max_d = r.rnd_int(1, params.max_depth);
    if (max_breadth == 0)
        max_breadth = r.rnd_int(1, params.max_breadth);
    if (max_size == 0)
        max_size = r.rnd_int(1, params.max_size);
    DataType root_type = DataTypeEnum<T>::value;

    auto prg = tree<Node>();

    fmt::print("building program with max size {}, max depth {}",max_size,max_d); 

    // Queue of nodes that need children
    vector<tuple<TreeIter, DataType, int>> queue; 

    if (max_size == 1)
    {
        auto root = prg.insert(prg.begin(), get_terminal(root_type));
    }
    else
    {
        cout << "getting op of type " << DataTypeName[root_type] << endl;
        auto n = get_op(root_type);
        cout << "chose " << n.name << endl;
        // auto spot = prg.set_head(n);
        cout << "inserting...\n";
        auto spot = prg.insert(prg.begin(), n);
        // node depth
        int d = 1;
        // current tree size
        int s = 1;
        //For each argument position a of n, Enqueue(a; g) 
        for (auto a : n.arg_types)
        { 
            cout << "queing a node of type " << DataTypeName[a] << endl;
            queue.push_back(make_tuple(spot, a, d));
        }

        cout << "queue size: " << queue.size() << endl; 
        cout << "entering first while loop...\n";
        while (queue.size() + s < max_size && queue.size() > 0) 
        {
            cout << "queue size: " << queue.size() << endl; 
            auto [qspot, t, d] = RandomDequeue(queue);

            cout << "current depth: " << d << endl;
            if (d == max_d)
            {
                cout << "getting " << DataTypeName[t] << " terminal\n"; 
                prg.append_child(qspot, get_terminal(t));
            }
            else
            {
                //choose a nonterminal of matching type
                cout << "getting op of type " << DataTypeName[t] << endl;
                auto n = get_op(t);
                cout << "chose " << n.name << endl;
                TreeIter new_spot = prg.append_child(qspot, n);
                // For each arg of n, add to queue
                for (auto a : n.arg_types)
                {
                    cout << "queing a node of type " << DataTypeName[a] << endl;
                    queue.push_back(make_tuple(new_spot, a, d+1));
                }
            }
            ++s;
            cout << "current tree size: " << s << endl;
        } 
        cout << "entering second while loop...\n";
        while (queue.size() > 0)
        {
            if (queue.size() == 0)
                break;

            cout << "queue size: " << queue.size() << endl; 

            auto [qspot, t, d] = RandomDequeue(queue);

            cout << "getting " << DataTypeName[t] << " terminal\n"; 
            prg.append_child(qspot, get_terminal(t));

        }
    }
    cout << "final tree:\n" 
        << prg.begin().node->get_model() << "\n"
        << prg.begin().node->get_tree_model(true) << endl;
         /* << prg.get_model() << "\n" */ 
         /* << prg.get_model(true) << endl; // pretty */

    return Program<T>(*this,prg);
};
} //Brush

