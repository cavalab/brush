
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
        // create a copy of the tree
        Program<PT> simplified_program(program);
        
        // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
        TreeIter spot = simplified_program.Tree.begin();
        while(spot != simplified_program.Tree.end())
        {
            Node n = spot.node->data;

            if (Isnt<NodeType::Terminal, NodeType::Constant, NodeType::MeanLabel>(n.node_type)
            &&  n.get_prob_change()>0)
            {        
                // get new_pred with predictions after simplification
                VectorXf branch_pred;
                if constexpr (PT==ProgramType::Regressor || PT==ProgramType::BinaryClassifier)
                {
                    branch_pred = (*spot.node).template predict<ArrayXf>(d);
                }
                else if constexpr (PT==ProgramType::MulticlassClassifier)
                {
                    ArrayXXf out = (*spot.node).template predict<ArrayXXf>(d);
                    auto argmax = Function<NodeType::ArgMax>{};
                    branch_pred = ArrayXf(argmax(out).template cast<float>());
                }
                else
                {
                    HANDLE_ERROR_THROW("No predict available for the class.");
                }

                if (variance(branch_pred) < 1e-4) // TODO: calculate threshold based on data
                {
                    // get constant equivalent to its argtype (all data types should have
                    // a constant defined in the search space for its given type). It will be
                    // the last node of the terminal map for the given type
                    Node cte = ss.terminal_map.at(n.ret_type).at(
                        ss.terminal_map.at(n.ret_type).size()-1);

                    cte.W = branch_pred.mean();
                    simplified_program.Tree.erase_children(spot); 
                    spot = simplified_program.Tree.replace(spot, cte);
                }
            }
            ++spot;
        }

        program.Tree = simplified_program.Tree;

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
