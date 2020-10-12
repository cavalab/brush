/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_H
#define NODE_H
//external includes
//
#include <iostream>
#include <string>
// internal includes
#include "../tree.h"
#include "../data.h"
#include "../state.h"
using std::cout;
using std::string;

namespace BR {


/* template<typename Tout> */ 
/* class Terminal : public Node {}; */

/* template< */
/*     typename Tout, // the output type of the node */
/*     typename Tin,  // the input type of the node */
/*     typename Tin2 = Tin // an optional second input type for the node for 2+ary nodes */
/*     // TODO: use ellipses? https://www.learncpp.com/cpp-tutorial/714-ellipsis-and-why-to-avoid-them/ */
/*     > class Op : public Node */
/* { */
/*     public: */
/*         // function pointer */ 
/* }; */


// Node is a template class that takes a single type parameter, which is a function pointer
template<typename F> class Node;

/* typedef tree_node_<Node*> TreeNode; */  

class NodeBaseBase
{
    virtual ~NodeBaseBase() = 0;
};
// Now we define specializations for the function pointers
template<typename R, typename... Args>
class NodeBase : public NodeBaseBase
{
    using RetType = R;
	using Function = R(*)(Args...);
    using ArgTypes = std::tuple<Args...>;
    static constexpr std::size_t ArgCount = sizeof...(Args);
    
	public:

		// name of the node
		string name;
		// output type
		string otype;
		// arity
		map<string,int> arity;
		// complexity
		int complexity;
		// sample probability of this node
		float probability;

		NodeBase(string name)
		{
			this->name = name;
			probability = 1.0;
		}
		Node() = default;
		~Node() = default;
		Node(const Node&) = default;
		//TODO: implement this
		Node& operator=(Node && other) = default;

		//TODO: revisit this
		/* bool operator==(Node && other){return false;}; */
		bool operator==(const Node & other){return this->name==other.name;};

		// note this can be called in derived classes via Node::swap(b)
		void swap(Node& b)
		{
			using std::swap;
			swap(this->name,b.name);
			swap(this->probability,b.probability);
		};

		virtual State fit(const Data& d, 
							   tree_node_<Node*>* child1=0, 
							   tree_node_<Node*>* child2=0) = 0;

		virtual State predict(const Data& d, 
							   tree_node_<Node*>* child1=0, 
							   tree_node_<Node*>* child2=0) = 0;

		bool isNodeDx(){return false; };
};


// specialization for commutative and associate binary operators
template<typename R, typename Arg>
/* class Node<R(*)(Args...)> : NodeBase<R, Args...> */
// this class should work for associate and commutative operators:
// plus, multiply, AND, OR
class Node<R(*)(Arg, Arg)> : NodeBase<R, Arg, Arg>
{
public:
	using base = NodeBase<R, Arg, Arg>;
	using Function = R(*)(Arg, Arg);
	template<std::size_t N>
	using NthArg = std::tuple_element_t<N, ArgTypes>;
	using FirstArg = NthArg<0>;
	using LastArg = NthArg<base::ArgCount - 1>;

	State fit(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0)
	{
		tree_node_<Node*>* sib = child1;
		vector<Arg> inputs; //(ArgCount);
        for (int i = 0; i < ArgCount; ++i)
        {
            inputs.push_back( sib->fit(d).get<Arg>() );
            sib = sib->next_sibling;
        }
		R output = transform_reduce( inputs.begin(), inputs.end(),
                                     W.begin(), 0, 
                                     Function,
                                     std::multiplies<>()
                                     ) ;
		// get inputs
		State s; 
		s.set<R>(output);
		return  s;
		
		

	}

	State predict(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0) = 0;
};

// terminals
template<typename R>
class Node<R(*)()> : NodeBase<R>
{
public:
	using Pointer = R(*)();
	
	virtual State fit(const Data& d, 
						   tree_node_<Node*>* child1=0, 
						   tree_node_<Node*>* child2=0) = 0;

	virtual State predict(const Data& d, 
						   tree_node_<Node*>* child1=0, 
						   tree_node_<Node*>* child2=0) = 0;
	
};


// specialization for non-commutative and non-associate binary operators
template<typename R, typename Arg>
/* class Node<R(*)(Args...)> : NodeBase<R, Args...> */
class Node<R(*)(Arg, Arg)> : NodeBase<R, Arg, Arg>
// this class should work for any non-associative/commutative operators
{
public:
	using base = NodeBase<R, Arg, Arg>;
	using Function = R(*)(Arg, Arg);
	template<std::size_t N>
	using NthArg = std::tuple_element_t<N, ArgTypes>;
	using FirstArg = NthArg<0>;
	using LastArg = NthArg<base::ArgCount - 1>;

	State fit(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0)
	{
		vector<State> inputs; //(ArgCount);
		tree_node_<Node*>* sib = child1;
        for (int i = 0; i < ArgCount; ++i)
        {
            inputs.push_back( sib->fit(d) );
            sib = sib->next_sibling;
        }

        State output; 
        output.set<R>(
                Function( inputs.at(0).get<Arg>(), inputs.at(1).get<Arg>());
                );

		return  output;
		
		

	}

	State predict(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0) = 0;
};

// specialization for non-commutative and non-associate unary operators
template<typename R, typename Arg>
/* class Node<R(*)(Args...)> : NodeBase<R, Args...> */
class Node<R(*)(Arg)> : NodeBase<R, Arg>
{
public:
	using base = NodeBase<R, Arg>;
	using Function = R(*)(Arg);
	template<std::size_t N>
	using NthArg = std::tuple_element_t<N, ArgTypes>;
	using FirstArg = NthArg<0>;
	using LastArg = NthArg<base::ArgCount - 1>;

	State fit(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0)
	{

        State output; 
        output.set<R>(
                Function( this->W.at(0) * child1->fit(d).get<Arg>() );
                );

		return  output;

	}

	State predict(const Data& d, 
						   TreeNode* child1=0, 
						   TreeNode* child2=0) = 0;
};
}
#endif
