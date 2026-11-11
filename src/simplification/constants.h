#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../init.h"
#include "../types.h"
#include "../program/program.h"
#include "../vary/search_space.h"
#include "../util/utils.h"
#include "record.h"

using namespace std;
using Brush::Node;
using Brush::DataType;

namespace Brush { namespace Simpl{
    class Constants_simplifier
    {
        public:
            /// @param records if not null, every replacement is appended to it
            template <ProgramType P>
            Program<P> simplify_tree(
                Program<P>& program, const SearchSpace &ss, const Dataset &d,
                SimplificationRecords* records = nullptr)
            {
                using RetType =
                typename std::conditional_t<P == PT::Regressor, ArrayXf,
                            std::conditional_t<P == PT::Representer, ArrayXXf, ArrayXf
                >>;

                // create a copy of the tree
                Program<P> simplified_program(program);
                
                // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
                TreeIter spot = simplified_program.Tree.begin();
                while(spot != simplified_program.Tree.end())
                {
                    Node n = spot.node->data;

                    // This is avoiding using booleans.
                    // non-wheightable nodes are not simplified. TODO: revisit this and see if they should (then implement it)
                    if (Isnt<NodeType::Terminal, NodeType::Constant, NodeType::MeanLabel>(n.node_type)
                    &&  n.get_prob_change()>0
                    &&  IsWeighable(n.ret_type)
                    )
                    {       
                        // TODO: check if holds alternative and use this information, instead of making it templated. Also, return void.
                        // get new_pred with predictions after simplification
                        VectorXf branch_pred;
                        if constexpr (P==ProgramType::Regressor || P==ProgramType::BinaryClassifier)
                        {
                            RetType pred = (*spot.node).predict<RetType>(d);
                            branch_pred = pred.template cast<float>();
                        }
                        else if constexpr (P==ProgramType::MulticlassClassifier)
                        {
                            // A multiclass tree contains ArrayF logit branches
                            // below its MatrixF Softmax root.  Evaluate each
                            // node using its own return type; treating a logit
                            // branch as MatrixF requests an invalid dispatch
                            // callable (for example Sum(MatrixF)).
                            if (n.ret_type == DataType::MatrixF)
                            {
                                ArrayXXf out = (*spot.node).template predict<ArrayXXf>(d);
                                auto argmax = Function<NodeType::ArgMax>{};
                                branch_pred = ArrayXf(argmax(out).template cast<float>());
                            }
                            else
                            {
                                branch_pred = (*spot.node).template predict<ArrayXf>(d);
                            }
                        }
                        else
                        {
                            HANDLE_ERROR_THROW("No predict available for the class.");
                        }

                        if (variance(branch_pred) < 1e-6)
                        {
                            // get constant equivalent to its argtype (all data types should have
                            // a constant defined in the search space for its given type). It will be
                            // the last node of the terminal map for the given type
                            Node cte = ss.terminal_map.at(n.ret_type).at(
                                ss.terminal_map.at(n.ret_type).size()-1);

                            cte.W = branch_pred.mean();

                            string original;
                            if (records)
                                original = spot.node->get_model();

                            simplified_program.Tree.erase_children(spot); 
                            spot = simplified_program.Tree.replace(spot, cte);

                            if (records)
                                records->push_back({0, 0, "constants",
                                                    DataTypeName.at(n.ret_type),
                                                    original, spot.node->get_model(), 0.0f});
                        }
                    }
                    ++spot;
                }
                program.Tree = simplified_program.Tree;
                return simplified_program;
            }

            Constants_simplifier();
            ~Constants_simplifier();
        private:

    };
} // Simply
} // Brush

#endif
