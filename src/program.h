/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef PROGRAM_H
#define PROGRAM_H
//external includes

//
#include <iostream>

#include <string>
// internal includes
#include "tree.h"

#include "data.h"
#include "state.h"
#include "node.h"
#include "nodemap.h"

using std::cout;
using std::string;
using Brush::State;
using Brush::Dat::Data;
using Brush::SearchSpace;

namespace Brush {


/* template<typename T> class Program; */

template<typename T>
T RandomDequeue(std::vector<T>& Q)
{
    int loc = r.rnd_int(0, Q.size()-1);
    std::swap(Q[loc], Q[Q.size()-1]);
    T val = Q.back();
    Q.pop_back();
    return val;
}

template<typename T> class Program //: public tree<NodeBase*>
{
    typedef tree<NodeBase*>::pre_order_iterator Iter; 

    public:
      
        
        Program<T> mutate(){}; 
        Program<T> cross(Program<T>& other){}; 
       
        T fit(const Data& d)
		{
            Iter start = prg.begin(); 
            State out = start.node->fit(d);
			return std::get<T>(out);
		};

        T predict(const Data& d)
		{
            Iter start = prg.begin(); 
            State out = start.node->predict(d);
            cout << "Program::predict returning\n";
			return std::get<T>(out);
		};
        void grad_descent(const ArrayXf& gradient, const Data& d)
		{
            Iter start = prg.begin(); 
            start.node->grad_descent(gradient, d);
		};

        string get_model()
        {
            Iter start = prg.begin(); 
            return start.node->get_model();
        }


        /// the underlying program
        tree<NodeBase*> prg; 

        // constructs a tree using functions, terminals, and settings
        void make_program(int max_d, int max_size=0)
        {
            /*
            * implementation of PTC2 for strongly typed GP from Luke et al. 
            * "Two fast tree-creation algorithms for genetic programming"
            *  
            */
            cout << "============================================\n";
            prg.clear();
            if (max_size == 0)
                max_size = r.rnd_int(1, pow(2, max_d));

            cout << "building program with max size " << max_size 
                 << ", max_depth: " << max_d << endl;


            // Queue of nodes that need children
            vector<tuple<Iter, type_index, int>> queue; 

            if (max_size == 1)
            {
                auto root = prg.insert(prg.begin(), SS.get_terminal(typeid(T)));
            }
            else
            {
                // cout << "getting op of type " << type_names[typeid(T)] << endl;
                auto n = SS.get_op(typeid(T));
                // cout << "chose " << n->name << endl;
                auto spot = prg.insert(prg.begin(), n);
                // node depth
                int d = 1;
                // current tree size
                int s = 1;
                //For each argument position a of n, Enqueue(a; g) 
                for (auto a : n->arg_types())
                { 
                    // cout << "queing a node of type " << type_names[a] << endl;
                    queue.push_back(make_tuple(spot, a, d));
                }

                // cout << "entering first while loop...\n";
                while (queue.size() + s < max_size && queue.size() > 0) 
                {
                    // cout << "queue size: " << queue.size() << endl; 
                    auto [qspot, t, d] = RandomDequeue(queue);

                    // cout << "d: " << d << endl;
                    if (d == max_d)
                    {
                        // cout << "getting " << type_names[t] << " terminal\n"; 
                        prg.append_child(qspot, SS.get_terminal(t));
                    }
                    else
                    {
                        //choose a nonterminal of matching type
                        // cout << "getting op of type " << type_names[t] << endl;
                        auto n = SS.get_op(t);
                        // cout << "chose " << n->name << endl;
                        Iter new_spot = prg.append_child(qspot, n);
                        // For each arg of n, add to queue
                        for (auto a : n->arg_types())
                        {
                            // cout << "queing a node of type " << type_names[a] << endl;
                            queue.push_back(make_tuple(new_spot, a, d+1));
                        }
                    }
                    ++s;
                    // cout << "s: " << s << endl;
                } 
                // cout << "entering second while loop...\n";
                while (queue.size() > 0)
                {
                    if (queue.size() == 0)
                        break;

                    // cout << "queue size: " << queue.size() << endl; 

                    auto [qspot, t, d] = RandomDequeue(queue);

                    // cout << "getting " << type_names[t] << " terminal\n"; 
                    prg.append_child(qspot, SS.get_terminal(t));

                }
            }
            cout << "final program: " << this->get_model() << endl;
        };

};
}
#endif
