/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef TEMPLATE_NODES_H
#define TEMPLATE_NODES_H
#include <typeinfo>
#include <functional>
#include "tree.h"
#include "util/tuples.h"

using namespace std;
using std::cout; 

/* TODO:
 * - fit and predict
 *   - get nodes working in a tree
 *   - proper moving functions eg swap, etc. in base class
 */

class NodeBase {
	public:
        typedef tree_node_<NodeBase*> TreeNode;  
		string name;
        virtual State fit(const Data&, TreeNode*, TreeNode*) = 0;
};

typedef tree_node_<NodeBase*> TreeNode;  

/* class NodeBaseBase */
/* { */
/*     virtual ~NodeBaseBase() = 0; */
/* }; */
/// Basic specialization for edge types (output and input types)
template<typename R, typename... Args>
class TypedNodeBase : public NodeBase
{
    public:
        using RetType = R;
        using ArgTypes = std::tuple<Args...>;
        template <std::size_t N>
        using NthArg = std::tuple_element<N, ArgTypes>;
        static constexpr std::size_t ArgCount = sizeof...(Args);
        /* const vector<std::size_t> arg_index(ArgCount); */
        /* arg_index = std::iota(arg_index.begin(),arg_index.end(),0); */
    
};


template<typename F> class Node; // : public NodeBase;

/* template <size_t I=0, typename... Ts> */
/* constexpr void assign_inputs(tuple<Ts...> inputs, */ 
template<typename R, typename... Args>
class Node<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        /* using ArgCount = typename base::ArgCount; */
        using ArgTypes = typename base::ArgTypes;
        /* using ArgTypesTypes = std::tuple<typename Args::value_type...>; */
        template <std::size_t N>
        using NthType = typename std::tuple_element<N, ArgTypes>::type;

        /* using NthArg = base::NthArg; */
        /* template<std::size_t N> */
        /* const auto ArgIdx = std::make_integer_sequence<std::size_t, N>; */
        /* const auto ArgIdx = std::make_index_sequence<sizeof...(Args)>{}; */
		/* static constexpr std::array<int, base::ArgCount> ArgIdx = \ */
		/* 	f_them_all(ArgTypes); */

        Function op; 

		template<size_t... Is>
		ArgTypes set_inputs(const array<State, base::ArgCount>& in, 
					std::index_sequence<Is...>)
		{ 
            return std::make_tuple(std::get<NthType<Is>>(in.at(Is))...);
		};
        /* template<size_t I, typename T> */
        /*     void set_item( */
        /* get<I>(t) = get<T>(in.at(I)); */

        Node(const Function& x, string name) 
        {
            this->op = x;
			this->name = name;
            cout << "RetType: " << typename base::RetType() << endl;
            cout << "ArgCount: " << base::ArgCount << endl;
        };

        State fit(const Data& d, TreeNode* child1=0, TreeNode* child2=0)
	    {
            cout << "fitting " << this->name << endl;
            cout << "child1: " << child1 << endl;
            cout << "child2: " << child2 << endl;
            ArgTypes inputs; //(ArgCount);
            array<State, base::ArgCount> child_outputs;

            cout << "gathering inputs..." << endl;
            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                cout << "sibling: " << sib << endl;
                cout << "sibling name: " << sib->data->name << endl;
                child_outputs.at(i) = sib->fit(d);
                sib = sib->next_sibling;
            }
            inputs = set_inputs(child_outputs, 
                                std::make_index_sequence<sizeof...(Args)>{}
                                );

            State out = std::apply(this->op, inputs);
            cout << "returning " << std::get<R>(out) << endl;
 			return std::apply(this->op, inputs);
        }
};

template<typename R>
class Node: public TypedNodeBase<R>
{
    public:
        using base = TypedNodeBase<R>;
        string variable_name;
		unsigned int loc;

        Node(string name, int loc) 
        {
            this->variable_name = name;
			this->name = this->variable_name;
			this->loc = loc;
            /* this->op = x(); */
            /* ArgTypes types; */ 
            /* cout << "function: " << x << endl; */
            cout << "RetType: " << typename base::RetType() << endl;
            cout << "ArgCount: " << base::ArgCount << endl;
            /* cout << "ArgTypes: " << ArgTypes(); */
            /* for (auto at : types) cout << at; */
        };

        State fit(const Data& d, 
						   TreeNode* child1, 
						   TreeNode* child2)
        {
            //TODO: this needs to be specialized for different terminal types
            //that deal directly with data.
			/* State out; */
			/* std::get<R>(out) = d.X.row(this->loc); */ 
            return 0;
            /* return d.X.row(this->loc); */
        }
};

template<>
class Node<ArrayXf>: public TypedNodeBase<ArrayXf>
{
    public:
        string variable_name;
		unsigned int loc;

        Node(string name, int loc) 
        {
            this->variable_name = name;
			this->name = this->variable_name;
			this->loc = loc;
        };
        State fit(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0)
	    {
			/* State out; */
			/* std::get<R>(out) = d.X.row(this->loc); */ 
            cout << "returning " << d.X.row(this->loc).transpose() << endl;
            return ArrayXf(d.X.row(this->loc));
        };
};

// specialization for commutative and associate binary operators
/* template<typename R, typename Arg> */
/* class Node<R(*)(Args...)> : NodeBase<R, Args...> */
/* // this class should work for associate and commutative operators: */
/* // plus, multiply, AND, OR */
/* class Node<R(*)(Arg, Arg)> : NodeBase<R, Arg, Arg> */
/* { */
/* public: */
/* 	using base = NodeBase<R, Arg, Arg>; */
/* 	using Function = R(*)(Arg, Arg); */
/* 	/1* template<std::size_t N> *1/ */
/* 	/1* using NthArg = std::tuple_element_t<N, ArgTypes>; *1/ */
/* 	/1* using FirstArg = NthArg<0>; *1/ */
/* 	/1* using LastArg = NthArg<base::ArgCount - 1>; *1/ */

/*     Function op; */
/*     /1* Node(Function x) *1/ */
/*     Node() */
/*     { */
/*         cout << "RetType: " << this->RetType; */
/*         cout << "ArgCount: " << this->ArgCount; */
/*         cout << "ArgTypes: "; */
/*         for (auto at : this->ArgTypes) cout << at; */
/*         cout << endl; */
/*     }; */

	/* State fit(const Data& d, */ 
	/* 					   TreeNode* child1=0, */ 
	/* 					   TreeNode* child2=0) */
	/* { */
	/* 	tree_node_<Node*>* sib = child1; */
	/* 	vector<Arg> inputs; //(ArgCount); */
        /* for (int i = 0; i < ArgCount; ++i) */
        /* { */
            /* inputs.push_back( sib->fit(d).get<Arg>() ); */
            /* sib = sib->next_sibling; */
        /* } */
/* 	/1* 	R output = transform_reduce( inputs.begin(), inputs.end(), *1/ */
/*                                      /1* W.begin(), 0, *1/ */ 
/*                                      /1* Function, *1/ */
/*                                      /1* std::multiplies<>() *1/ */
/*                                      /1* ) ; *1/ */
/* 	/1* 	// get inputs *1/ */
/* 	/1* 	State s; *1/ */ 
/* 	/1* 	s.set<R>(output); *1/ */
/* 	/1* 	return  s; *1/ */
		
		

/* 	/1* } *1/ */

/* /1* 	/2* State predict(const Data& d, *2/ *1/ */ 
/* /1* 	/2* 					   TreeNode* child1=0, *2/ *1/ */ 
/* /1* 	/2* 					   TreeNode* child2=0) = 0; *2/ *1/ */
/* }; */

// terminals
/* template<typename R> */
/* class Node<R(*)()> : NodeBase<R> */
/* { */
/* public: */
/* 	using Pointer = R(*)(); */
	
	/* virtual State fit(const Data& d, */ 
	/* 					   tree_node_<Node*>* child1=0, */ 
	/* 					   tree_node_<Node*>* child2=0) = 0; */

	/* virtual State predict(const Data& d, */ 
	/* 					   tree_node_<Node*>* child1=0, */ 
	/* 					   tree_node_<Node*>* child2=0) = 0; */
	
/* }; */


// specialization for non-commutative and non-associate binary operators
/* template<typename R, typename Arg> */
/* /1* class Node<R(*)(Args...)> : NodeBase<R, Args...> *1/ */
/* class Node<R(*)(Arg)> : NodeBase<R, Arg> */
/* // this class should work for any non-associative/commutative operators */
/* { */
/* public: */
/* 	using base = NodeBase<R, Arg, Arg>; */
/* 	using Function = R(*)(Arg, Arg); */
/* 	/1* template<std::size_t N> *1/ */
/* 	/1* using NthArg = std::tuple_element_t<N, ArgTypes>; *1/ */
/* 	/1* using FirstArg = NthArg<0>; *1/ */
/* 	/1* using LastArg = NthArg<base::ArgCount - 1>; *1/ */

/* 	/1* State fit(const Data& d, *1/ */ 
/* 	/1* 					   TreeNode* child1=0, *1/ */ 
/* 	/1* 					   TreeNode* child2=0) *1/ */
/* 	/1* { *1/ */
/* 	/1* 	vector<State> inputs; //(ArgCount); *1/ */
/* 	/1* 	tree_node_<Node*>* sib = child1; *1/ */
/*         /1* for (int i = 0; i < ArgCount; ++i) *1/ */
/*         /1* { *1/ */
/*             /1* inputs.push_back( sib->fit(d) ); *1/ */
/*             /1* sib = sib->next_sibling; *1/ */
/*         /1* } *1/ */

/*         /1* State output; *1/ */ 
/*         /1* output.set<R>( *1/ */
/*                 /1* Function( inputs.at(0).get<Arg>(), inputs.at(1).get<Arg>()); *1/ */
/*                 /1* ); *1/ */

/* 	/1* 	return  output; *1/ */
		
		

/* 	/1* } *1/ */

/* 	/1* State predict(const Data& d, *1/ */ 
/* 	/1* 					   TreeNode* child1=0, *1/ */ 
/* 	/1* 					   TreeNode* child2=0) = 0; *1/ */
/* }; */

/* // specialization for non-commutative and non-associate unary operators */
/* template<typename R, typename Arg> */
/* /1* class Node<R(*)(Args...)> : NodeBase<R, Args...> *1/ */
/* class Node<R(*)(Arg)> : NodeBase<R, Arg> */
/* { */
/* public: */
/* 	using base = NodeBase<R, Arg>; */
/* 	using Function = R(*)(Arg); */
/* 	template<std::size_t N> */
/* 	using NthArg = std::tuple_element_t<N, ArgTypes>; */
/* 	using FirstArg = NthArg<0>; */
/* 	using LastArg = NthArg<base::ArgCount - 1>; */

/* 	/1* State fit(const Data& d, *1/ */ 
/* 	/1* 					   TreeNode* child1=0, *1/ */ 
/* 	/1* 					   TreeNode* child2=0) *1/ */
/* 	/1* { *1/ */

/*         /1* State output; *1/ */ 
/*         /1* output.set<R>( *1/ */
/*                 /1* Function( this->W.at(0) * child1->fit(d).get<Arg>() ); *1/ */
/*                 /1* ); *1/ */

/* 	/1* 	return  output; *1/ */

/* 	/1* } *1/ */

/* 	/1* State predict(const Data& d, *1/ */ 
/* 	/1* 					   TreeNode* child1=0, *1/ */ 
/* 	/1* 					   TreeNode* child2=0) = 0; *1/ */
/* }; */

#endif
