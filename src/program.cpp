/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/

#include "program.h"



namespace Brush 
{

// constructs a tree using functions, terminals, and settings
tree<Node> make_program(SearchSpace& SS, type_index root_type, 
                             int max_d, int max_breadth, int max_size)
{
    /*
    * implementation of PTC2 for strongly typed GP from Luke et al. 
    * "Two fast tree-creation algorithms for genetic programming"
    *  
    */
    cout << "============================================\n";
    auto prg = tree<Node>();

    cout << "building program with max size " << max_size 
            << ", max_depth: " << max_d << endl;


    // Queue of nodes that need children
    vector<tuple<Iter, type_index, int>> queue; 

    if (max_size == 1)
    {
        // auto root = prg.set_head(SS.get_terminal(root_type));
        auto root = prg.insert(prg.begin(), SS.get_terminal(root_type));
    }
    else
    {
        cout << "getting op of type " << type_names[root_type] << endl;
        auto n = SS.get_op(root_type);
        cout << "chose " << n->name << endl;
        // auto spot = prg.set_head(n);
        auto spot = prg.insert(prg.begin(), n);
        // node depth
        int d = 1;
        // current tree size
        int s = 1;
        //For each argument position a of n, Enqueue(a; g) 
        for (auto a : n->arg_types())
        { 
            cout << "queing a node of type " << type_names[a] << endl;
            queue.push_back(make_tuple(spot, a, d));
        }

        cout << "entering first while loop...\n";
        while (queue.size() + s < max_size && queue.size() > 0) 
        {
            cout << "queue size: " << queue.size() << endl; 
            auto [qspot, t, d] = RandomDequeue(queue);

            cout << "d: " << d << endl;
            if (d == max_d)
            {
                cout << "getting " << type_names[t] << " terminal\n"; 
                prg.append_child(qspot, SS.get_terminal(t));
            }
            else
            {
                //choose a nonterminal of matching type
                cout << "getting op of type " << type_names[t] << endl;
                auto n = SS.get_op(t);
                cout << "chose " << n->name << endl;
                Iter new_spot = prg.append_child(qspot, n);
                // For each arg of n, add to queue
                for (auto a : n->arg_types())
                {
                    cout << "queing a node of type " << type_names[a] << endl;
                    queue.push_back(make_tuple(new_spot, a, d+1));
                }
            }
            ++s;
            cout << "s: " << s << endl;
        } 
        cout << "entering second while loop...\n";
        while (queue.size() > 0)
        {
            if (queue.size() == 0)
                break;

            cout << "queue size: " << queue.size() << endl; 

            auto [qspot, t, d] = RandomDequeue(queue);

            cout << "getting " << type_names[t] << " terminal\n"; 
            prg.append_child(qspot, SS.get_terminal(t));

        }
    }
    cout << "final program:\n" 
         << prg.get_model() << "\n" 
         << prg.get_model(true) << endl; // pretty

    return prg;
};




} // Brush

