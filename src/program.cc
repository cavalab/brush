/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/

#include "program.h"



namespace Brush 
{

// constructs a tree using functions, terminals, and settings
tree<NodeBase*> make_program(type_index root_type, 
                             int max_d, int max_breadth, int max_size)
{
    /*
    * implementation of PTC2 for strongly typed GP from Luke et al. 
    * "Two fast tree-creation algorithms for genetic programming"
    *  
    */
    cout << "============================================\n";
    auto prg = tree<NodeBase*>();

    cout << "building program with max size " << max_size 
            << ", max_depth: " << max_d << endl;


    // Queue of nodes that need children
    vector<tuple<Iter, type_index, int>> queue; 

    if (max_size == 1)
    {
        auto root = prg.insert(prg.begin(), SS.get_terminal(root_type));
    }
    else
    {
        cout << "getting op of type " << type_names[root_type] << endl;
        auto n = SS.get_op(root_type);
        cout << "chose " << n->name << endl;
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
    cout << "final program: " << prg.head->first_child->get_model() << endl;
    return prg;
};

/// point mutation: replace node with same typed node
tree<NodeBase*> point_mutation(tree<NodeBase*>& prg, Iter spot)
{
    auto newNode = SS.get_node_like(spot.node->data); 
    prg.replace(spot, newNode);
    return prg;
}
/// insert a node with spot as a child
tree<NodeBase*> insert_mutation(tree<NodeBase*>& prg, Iter spot)
{
    auto spot_type = spot.node->data->ret_type();
    auto n = SS.get_op_with_arg(spot_type, spot_type); 
    // make node n wrap the subtree at the chosen spot
    auto parent_node = prg.wrap(spot, n);

    // now fill the arguments of n appropriately
    bool spot_filled = false;
    for (auto a: n->arg_types())
    {
        
        if (spot_filled)
        {
            // if spot is in its child position, append children
            prg.append_child(parent_node, SS.get_terminal(a));
        }
        // if types match, treat this spot as filled by the spot node 
        else if (a == spot_type)
            spot_filled = true;
        // otherwise, add siblings before spot node
        else
            prg.insert(spot, SS.get_terminal(a));

    } 
    return prg;
}
/// delete subtree and replace it with a terminal of the same return type
tree<NodeBase*> delete_mutation(tree<NodeBase*>& prg, Iter spot)
{
    auto terminal = SS.get_terminal(spot.node->data->ret_type()); 
    prg.erase_children(spot); 
    prg.replace(spot, terminal);
    return prg;
}



} // Brush

