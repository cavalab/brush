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
		string name;
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
        /* template <std::size_t N> */
        /* using NthArg = std::tuple_element<N, ArgTypes>; */
        static constexpr std::size_t ArgCount = sizeof...(Args);
    
};


template<typename F> class Node; // : public NodeBase;

template<typename R, typename... Args>
class Node<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        /* using ArgCount = typename base::ArgCount; */
        using ArgTypes = typename base::ArgTypes;

        using NthArg = std::tuple_element<base::ArgCount, ArgTypes>;

        Function op; 

        Node(Function& x, string name) 
        {
            this->op = x;
			this->name = name;
            /* this->op = x(); */
            /* ArgTypes types; */ 
            /* cout << "function: " << x << endl; */
            cout << "RetType: " << typename base::RetType() << endl;
            cout << "ArgCount: " << base::ArgCount << endl;
            /* cout << "ArgTypes: " << ArgTypes(); */
            /* for (auto at : types) cout << at; */
        };

        State fit(const Data& d, TreeNode* child1=0, TreeNode* child2=0)
	    {
            ArgTypes inputs; //(ArgCount);

            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                /* std::get<i>(inputs) =  std::get<base::NthArg<i>>(sib->fit(d)) ; */
                tuple_index(inputs, i) =  tuple_index(sib->fit(d), i) ;
                sib = sib->next_sibling;
            }
            State out;
            /* std::get<R>(out) = this->op( */
            /*         std::get<base::ArgCount>(std::forward<ArgTypes>(inputs))... */
            /*             ); */
 			std::get<R>(out) = std::apply([this](auto &&... args) 
										  { this->op(args...); }, 
								  		  inputs);
            return out;
        }
};

template<typename R>
class Node: public TypedNodeBase<R>
{
    public:
        using base = TypedNodeBase<R>;
        string variable_name;

        Node(string name) 
        {
            this->variable_name = name;
			this->name = this->variable_name;
            /* this->op = x(); */
            /* ArgTypes types; */ 
            /* cout << "function: " << x << endl; */
            cout << "RetType: " << typename base::RetType() << endl;
            cout << "ArgCount: " << base::ArgCount << endl;
            /* cout << "ArgTypes: " << ArgTypes(); */
            /* for (auto at : types) cout << at; */
        };

        State fit(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0)
	    {
            
        }
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
