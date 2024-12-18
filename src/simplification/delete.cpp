
#include "delete.h"

namespace Brush { namespace Simpl{
    
    Delete_simplifier* Delete_simplifier::instance = NULL;
    
    Delete_simplifier::Delete_simplifier()
    {                   
    
    }

    Delete_simplifier* Delete_simplifier::initSimplifier()
    {
        // creates the static random generator by calling the constructor
        if (!instance)
        {
            instance = new Delete_simplifier();
        }

        return instance;
    }

    template <ProgramType PT>
    Program<PT> Delete_simplifier::simplify_tree(
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
        
        // iterate over the tree, trying to simulate a delete mutation, and keeping the change if the pred does not change.

        int iter = 0;
        while(++iter < 100)
        {
            // cout << "starting another interaction" << endl;
            TreeIter spot = r.select_randomly(
                    simplified_program.Tree.begin(), simplified_program.Tree.end() );

            Node n = spot.node->data;
            
            // skip it if fixed, and add its children to the queue
            if (n.get_prob_change()<=0)
                continue;

            tree<Node> backup(spot);

            auto opt = ss.sample_terminal(n.ret_type, true);
            if (!opt) // there is no terminal with compatible arguments
                continue;

            simplified_program.Tree.replace(spot, opt.value());

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
                // rollback and add its children to the queue
                simplified_program.Tree.move_ontop(spot, backup.begin());
            }   
            else{ // clean up the children
                simplified_program.Tree.erase_children(spot); 
            }
        }

        program.Tree = simplified_program.Tree;
    
        return simplified_program;
    }

    void Delete_simplifier::destroy()
    {
        if (instance)
            delete instance;
            
        instance = NULL;
    }
    
    Delete_simplifier::~Delete_simplifier() {}
} }
