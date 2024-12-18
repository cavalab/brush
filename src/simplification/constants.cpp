
#include "constants.h"

namespace Brush { namespace Simpl{
    
    Constants_simplifier* Constants_simplifier::instance = NULL;
    
    Constants_simplifier::Constants_simplifier()
    {                   
    
    }

    Constants_simplifier* Constants_simplifier::initSimplifier()
    {
        // creates the static random generator by calling the constructor
        if (!instance)
        {
            instance = new Constants_simplifier();
        }

        return instance;
    }

    template <ProgramType PT>
    Program<PT> Constants_simplifier::simplify_tree(
        Program<PT>& program, const SearchSpace &ss, const Dataset &d)
    {
        VectorXf original_pred;

        // ss.print();
        
        // TODO: wrap this ifs below into a single function and call it istead (change it here and below)
        if constexpr (PT==ProgramType::Regressor || PT==ProgramType::BinaryClassifier)
            original_pred = (*program.Tree.begin().node).template predict<ArrayXf>(d);
        else if constexpr (PT==ProgramType::MulticlassClassifier)
        {
            ArrayXXf out = (*program.Tree.begin().node).template predict<ArrayXXf>(d);
            auto argmax = Function<NodeType::ArgMax>{};
            original_pred = ArrayXf(argmax(out).template cast<float>());
        }
        else
        {
            HANDLE_ERROR_THROW("No predict available for the class.");
        }

        // create a copy of the tree
        Program<PT> simplified_program(program);
        
        // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
        vector<TreeIter> queue; 
        queue.push_back(simplified_program.Tree.begin());
        
        while(queue.size()>0)
        {
            // cout << "starting another interaction" << endl;

            TreeIter spot = queue.back();
            queue.pop_back();
        
            Node n = spot.node->data;
            // cout << spot.number_of_children() << endl;

            // for (size_t i=0; i<spot.number_of_children(); ++i)
            //     cout << i;
            // cout << endl;

            if (n.name == "Constant")
                continue;

            // skip it if fixed, and add its children to the queue
            if (n.get_prob_change()<=0)
            {        
                // cout << "fixed. skipping" << endl;
                for (size_t i=0; i<spot.number_of_children(); ++i)
                    queue.push_back(simplified_program.Tree.child(spot, i));
                    
                continue;
            }

            // get constant equivalent to its argtype (all data types should have
            // a constant defined in the search space for its given type). It will be
            // the last node of the terminal map for the given type
            Node cte = ss.terminal_map.at(n.ret_type).at(
                ss.terminal_map.at(n.ret_type).size()-1);

            // TODO: figure out a better way of swaping these nodes and subtrees
            // replace program tree with a terminal and store a backup
            tree<Node> copy(spot);
            simplified_program.Tree.replace(spot, cte);

            // get new_pred with predictions after simplification
            VectorXf new_pred;
            if constexpr (PT==ProgramType::Regressor || PT==ProgramType::BinaryClassifier)
                new_pred = (*simplified_program.Tree.begin().node).template predict<ArrayXf>(d);
            else if constexpr (PT==ProgramType::MulticlassClassifier)
            {
                ArrayXXf out = (*simplified_program.Tree.begin().node).template predict<ArrayXXf>(d);
                auto argmax = Function<NodeType::ArgMax>{};
                new_pred = ArrayXf(argmax(out).template cast<float>());
            }

            // check for significant changes in predictions
            if (!original_pred.isApprox(new_pred, 1e-8))
            {
                // for (size_t i=0; i<spot.number_of_children(); ++i)
                //     cout << i;
                // cout << endl;

                // rollback and add its children to the queue
                simplified_program.Tree.move_ontop(spot, copy.begin());

                // cout << "Moved. checking" << endl;
                    
                // for (size_t i=0; i<copy.begin().number_of_children(); ++i)
                //     cout << i;
                // cout << endl;

                for (size_t i=0; i<copy.begin().number_of_children(); ++i)
                    queue.push_back(copy.child(copy.begin(), i));

            }   
            else{            
                simplified_program.Tree.erase_children(spot); 
            }
            // else keep changes and do not append its children (since it does
            //not have them anymore)  
        }

        // cout << "finished. modifing reference" << endl;
        // replace program's tree
        program.Tree = simplified_program.Tree;
        
        // cout << "exiting..." << endl;

        return simplified_program;
    }

    void Constants_simplifier::destroy()
    {
        if (instance)
            delete instance;
            
        instance = NULL;
    }
    
    Constants_simplifier::~Constants_simplifier() {}
} }
