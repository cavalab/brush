#ifndef OPERATOR_H
#define OPERATOR_H
#include "init.h"
#include "data/data.h"
#include "nodemap.h"
#include "tree.h"
#include "tree_node.h"
#include <type_traits>
using namespace std;

namespace Brush {

    /* template<typename T> class tree_node_; */
    /* using TreeNode = typename class tree_node_<Node>; */ 
    /* using TreeNode = template<class T> class tree_node_; */ 

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

    template<NodeType NT, SigType S> //, typename ...Args>
    /* template<typename RetType> */
    struct Operator 
    {
        // TODO
        // read in ...Args types from a helper class 
        /* static constexpr ExecType exec_type = ExecValue<NT>::value; */
        // TODO: make arg types conditional on signature
        using Args = typename Signature<S>::ArgTypes;
        using RetType = typename Signature<S>::RetType;
        static constexpr size_t ArgCount = Signature<S>::ArgCount;
        /* using Args = typename ArgTypes<exec_type,RetType>::type; //std::tuple<Args...>; */
        //std::conditional_v<is_array_v<Args>, Args::size, std::tuple_size_v<Args>>;

        /* static constexpr size_t ArgCount = sizeof(...Args) */ 
        //templatize this
        /* if constexpr (is_array_v<Args>) */
        /* { */
        /* } */
        /* else */
        /*     static constexpr size_t ArgCount = std::tuple_size_v<Args>; */

        /* using FirstArg = first<...Args>::type; */ 
        /* using ArrayArgs = std::array<FirstArg,ArgCount>; */
        // enable if tuple
        template <std::size_t N>
        using NthType = conditional_t<!is_array_v<Args>,typename std::tuple_element<N, Args>::type,void>;
        
        Function<NT> F{}; 

        Operator() = default;
        /* Operator(NT node_type, RetType y, Args... args){}; */
        //////////////////////////////////////////////////////////////////////////////////
        /// Apply weights
        template<typename T>
        enable_if_t<is_array_v<T>> 
        apply_weights(T& inputs, const Node& n)
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
        template<typename T> T get_kids(const Data&, TreeNode&, bool fit);

        // get a std::array of kids
        template<typename T>
        enable_if_t<is_array_v<T>, T> 
        get_kids(const Data& d, TreeNode& tn, bool fit) 
        {
            T child_outputs;
            using arg_type = typename T::value_type;

            TreeNode* sib = tn.first_child;
            for (int i = 0; i < this->get_arg_count(); ++i)
            {
                child_outputs.at(i) = fit? sib->fit<arg_type>(d) : sib->predict<arg_type>(d);
                sib = sib->next_sibling;
            }
            return child_outputs;
        };

        /* template<size_t... Is> */
		/* auto tupleize(const array<State, ArgCount>& in, std::index_sequence<Is...>) */
		/* { */
        /*     return std::make_tuple(std::get<NthType<Is>>(in.at(Is))...); */
		/* }; */

		/* auto tupleize(const array<State, ArgCount>& in) */
        /* { */
        /*     return this->tupleize(in, std::make_index_sequence<ArgCount>{}); */
        /* }; */

        // get a std::tuple of kids
        template<int I>
        auto get_kid(TreeNode* first)
        {
            auto sib = first; 
            for (int i = 0 ; i < I; ++i)
                sib = sib->next_sibling;
            return sib;
        };

        template<typename T, size_t ...Is> 
        auto get_kids(const Data& d, TreeNode& tn, bool fit, std::index_sequence<Is...>)
        {
            /* static constexpr auto f = [&]( */
            return std::make_tuple(
                    fit ?  get_kid<Is>(tn.first_child)->fit<arg>(d) : 
                           get_kid<Is>(tn.first_child)->predict<arg>(d)
                        ...);
        };

        template<typename T>
        enable_if_t<!is_array_v<T>, T> 
        get_kids(const Data& d, TreeNode& tn, bool fit)
        {
            return get_kids(d, tn, fit, std::make_index_sequence<ArgCount>{});
            /* T child_outputs; */

            /* TreeNode* sib = tn.first_child; */
            /* for (int i = 0; i < ArgCount; ++i) */
            /* { */
            /*     constexpr size_t j = i; */
            /*     using arg_type = NthType<j>; */
            /*     child_outputs.at(i) = fit? sib->fit<arg_type>(d) : sib->predict<arg_type>(d); */
            /*     sib = sib->next_sibling; */
            /* } */
            /* return child_outputs; */
            /* array<State, ArgCount> child_outputs; */

            /* TreeNode* sib = first_child; */
            /* for (int i = 0; i < ArgCount; ++i) */
            /* { */
            /*     cout << i << endl; */
            /*     child_outputs.at(i) = (sib->*fn)(d); */
            /*     sib = sib->next_sibling; */
            /* } */
            /* return tupleize(child_outputs); */
        };

        template<typename T>
        auto get_kids_fit(const Data& d, TreeNode& tn)
        {
            return get_kids<T>(d, tn, true);
        };

        template<typename T>
        auto get_kids_predict(const Data& d, TreeNode& tn)
        {
            return get_kids(d, tn, false);
        };

        ///////////////////////////////////////////////////////////////////////////
        // fit and predict
        RetType eval(const Data& d, TreeNode& tn) override
	    {
            auto inputs = get_kids_fit<Args>(d, tn);
            if (tn.n.is_weighted)
                this->apply_weights(inputs, tn.n);
            return std::apply(F, inputs);
        };

        RetType fit(const Data& d, TreeNode& tn) override
	    {
            return eval(d,tn);
            /* auto inputs = get_kids_fit<Args>(d, tn); */
            /* if (n.is_weighted) */
            /*     this->apply_weights(inputs, tn->n); */
            /* return std::apply(F, inputs); */
            /* if (n.is_weighted) */
            /*     auto inputs = get_kids_fit<Args>(d, tn); */
            /*     this->apply_weights(inputs, tn->n); */
            /*     return std::apply(F, inputs); */
            /* else */
            /*     auto inputs = get_kids_fit<TupleArgs>(d, tn); */
            /*     return std::apply(F, inputs); */
        };

        RetType predict(const Data& d, TreeNode& tn) override
	    {
            return eval(d,tn);
            /* auto inputs = get_kids_predict<Args>(d, tn); */
            /* if (n.is_weighted) */
            /*     this->apply_weights(inputs, tn->n); */
            /* return std::apply(F, inputs); */
            /* if (n.is_weighted) */
            /*     auto inputs = get_kids_predict<Args>(d, tn); */
            /*     this->apply_weights(inputs, tn->n); */
            /*     return std::apply(F, inputs); */
            /* else */
            /*     auto inputs = get_kids_predict<TupleArgs>(d, tn); */
            /*     return std::apply(F, inputs); */
        };
    };
    ////////////////////////////////////////////////////////////////////////////
    // fit and predict Dispatch functions
    template<NodeType NT, SigType S> //, typename ...Args>
    auto DispatchFit(const Data& d, TreeNode& tn) 
    {
        const auto op = Operator<NT,S>{};
        return op.fit(d, tn);
    };

    template<NodeType NT, SigType S>
    auto DispatchPredict(const Data& d, TreeNode& tn) 

    {
        const auto op = Operator<NT,S>{};
        return op.predict(d, tn);
    };

}

#endif
