#ifndef OPERATOR_H
#define OPERATOR_H
#include "init.h"
#include "data/data.h"

namespace Brush {

    typedef class tree_node_<Node> TreeNode; 

    struct BaseOperator
    {
        virtual State fit(const Data&, TreeNode&) = 0;
        virtual State predict(const Data&, TreeNode&) = 0;
    };

    template <class T1, class ...T>
    struct first
    {
        typedef T1 type;
    };

    template<NodeType Type, typename R, typename ...Args>
    struct Operator 
    {
        using RetType = R;
        using TupleArgs = std::tuple<Args...>;
        static constexpr std::size_t ArgCount = sizeof...(Args);
        template <std::size_t N>
        using NthType = typename std::tuple_element<N, TupleArgs>::type;
        using FirstArg = first<...Args>::type; 
        using ArrayArgs = std::array<FirstArg,ArgCount>;
        
        Function<Type> F{}; 

        //////////////////////////////////////////////////////////////////////////////////
        /// Apply weights
        template<typename T>
        void apply_weights(T& inputs, const Node& n)
        {
            cout << "applying weights to " << n.name << " operator\n";
            std::transform(
                        inputs.begin(), 
                        inputs.end(),
                        n.W.begin(),
                        inputs.begin(), 
                        std::multiplies<>()
                        );
        }
        //////////////////////////////////////////////////////////////////////////////////
        /// Utilities to grab child outputs.
        template<typename T> T get_kids(Data&, TreeNode&, State (TreeNode::*fn)(const Data&));

        template<>
        ArrayArgs get_kids<ArrayArgs>(const Data& d, TreeNode& tn, 
                                          State (TreeNode::*fn)(const Data&))
        {
            ArrayArgs child_outputs;

            TreeNode* sib = tn->first_child;
            for (int i = 0; i < this->get_arg_count(); ++i)
            {
                child_outputs.at(i) = std::get<FirstArg>((sib->*fn)(d));
                sib = sib->next_sibling;
            }
            return child_outputs;
        };
        template<>
        TupleArgs get_kids<TupleArgs>(const Data& d, TreeNode& tn,
                                          State (TreeNode::*fn)(const Data&))
        {
            TupleArgs child_outputs;

            TreeNode* sib = tn->first_child;
            for (int i = 0; i < ArgCount; ++i)
            {
                std::get<i>(child_outputs) = std::get<NthType<i>>((sib->*fn)(d));
                sib = sib->next_sibling;
            }
            return child_outputs;
        };

        template<typename T>
        auto get_kids_fit(const Data& d, TreeNode& tn)
        {
            return get_kids<T>(d, n, &TreeNode::fit);
        };

        template<typename T>
        auto get_kids_predict(const Data& d, TreeNode& tn)
        {
            return get_kids(d, n, &TreeNode::predict);
        };

        ////////////////////////////////////////////////////////////////////////////////
        // fit and predict
        RetType fit(const Data& d, TreeNode& tn) override
	    {
            if (n.is_weighted)
                auto inputs = get_kids_fit<ArrayArgs>(d, tn);
                this->apply_weights(inputs, tn->n);
                return std::apply(F, inputs);
            else
                auto inputs = get_kids_fit<TupleArgs>(d, tn);
                return std::apply(F, inputs);
        };

        RetType predict(const Data& d, TreeNode& tn) override
	    {
            if (n.is_weighted)
                auto inputs = get_kids_predict<ArrayArgs>(d, tn);
                this->apply_weights(inputs, tn->n);
                return std::apply(F, inputs);
            else
                auto inputs = get_kids_predict<TupleArgs>(d, tn);
                return std::apply(F, inputs);
        };
    };

}

#endif
