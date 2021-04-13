/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef BASE_H
#define BASE_H
#include <typeinfo>
#include <functional>
#include "../tree.h"
#include "../util/utils.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <typeinfo>
using Eigen::ArrayXf;
using Eigen::VectorXf;
using namespace std;

/* TODO:
 *   - proper moving functions eg swap, etc. in base class
 */
/* namespace Brush{ */
using namespace Brush::Util;

namespace Brush {
namespace nodes {

// defined nodes:
template<typename F> class Node; 

class NodeBase {
	public:

        typedef tree_node_<NodeBase*> TreeNode;  
        /// full name of the node, with types
		string name;
        // name of the operator
		string op_name;
        // pretty name of the op, for printing equations
		string op_name_pretty;
        // whether to center the operator in pretty printing
        bool center_op;
        // chance of node being selected for variation
        float prob_change; 

        virtual std::type_index ret_type() const = 0; 
        virtual std::type_index args_type() const = 0; 
        virtual vector<std::type_index> arg_types() const = 0; 
        virtual size_t arg_count() const = 0;
        virtual State fit(const Data&, TreeNode*&, TreeNode*&) = 0;
        virtual State predict(const Data&, TreeNode*&, TreeNode*&) = 0;
        virtual void grad_descent(const ArrayXf&, const Data&, 
                                   TreeNode*&, TreeNode*&) = 0;
        virtual string get_model(TreeNode*& child1, 
                                 TreeNode*& child2) const = 0; 
         
        // virtual bool is() = 0;
        /*TODO: 
         * implement clone, copy, swap and assignment operators
         */
};

typedef tree_node_<NodeBase*> TreeNode;  

/* Basic specialization for edge types (output and input types)
 * Defines some basic shared parameters for nodes.
 */
template<typename R, typename... Args>
class TypedNodeBase : public NodeBase
{
    public:
        using RetType = R;
        using TupleArgs = std::tuple<Args...>;
        static constexpr std::size_t ArgCount = sizeof...(Args);
        template <std::size_t N>
        using NthType = typename std::tuple_element<N, TupleArgs>::type;

        void grad_descent(const ArrayXf&, const Data&, 
                           TreeNode*&, TreeNode*&) override 
        {
            throw runtime_error("grad_descent not implemented for " + name);
        };

        TypedNodeBase(string n)
        {
            this->set_op_name(n);
            n += "<"+ type_names.at(this->ret_type()) + "(";
            auto ats = this->arg_types();
            for (int i = 0; i<ats.size(); ++i )
            {
                n += type_names.at(ats.at(i));
                if(i < ats.size()-1)  
                    n += ",";
            }
            n += ")>";
            this->set_name(n);
        };

        void set_name(string n){this->name = n;}
        void set_op_name(string n){this->op_name = n;}
        std::type_index ret_type() const override { return typeid(R); }; 
        std::type_index args_type() const override { return typeid(TupleArgs);}; 
        vector<std::type_index> arg_types() const override
        {
            return this->get_arg_types(make_index_sequence<ArgCount>());
        }; 
        size_t arg_count() const override {return ArgCount;};

    protected:

        void set_prob_change(float w){ this->prob_change = w;};

        string get_model(TreeNode*& child1, TreeNode*& child2) const override
        { 
            TreeNode* sib = child1;
            string  child_outputs = "";
            while (sib != child2)
            {
                child_outputs += sib->get_model();
                sib = sib->next_sibling;
                if (sib != child2-1)
                    child_outputs += ",";
            }
            return this->name + "(" + child_outputs + ")";
        };

        template<size_t... Is>
        vector<std::type_index> get_arg_types(std::index_sequence<Is...>) const
        {
            return vector<type_index>{typeid(NthType<Is>)...};
        }
        template<size_t... Is>
		TupleArgs tupleize(const array<State, ArgCount>& in, 
        				   std::index_sequence<Is...>)
		{ 
            return std::make_tuple(std::get<NthType<Is>>(in.at(Is))...);
		};

		TupleArgs tupleize(const array<State, ArgCount>& in)
        {
            return this->tupleize(in, 
                                  std::make_index_sequence<ArgCount>{});
        };

        /// Utility to grab child outputs. 
        array<State, ArgCount> get_children(const Data& d,
                                            TreeNode*& child1, 
                                            TreeNode*& child2, 
                                            State (TreeNode::*fn)(const Data&)
                                            )
        {
            array<State, ArgCount> child_outputs;

            TreeNode* sib = child1;
            for (int i = 0; i < ArgCount; ++i)
            {
                cout << i << endl;
                child_outputs.at(i) = (sib->*fn)(d);
                sib = sib->next_sibling;
            }
            return child_outputs;
            
        };

        array<State, ArgCount> get_children_fit(const Data& d, 
                                                TreeNode*& child1, 
                                                TreeNode*& child2)
        {
            return get_children(d, child1, child2, &TreeNode::fit);
        }
        array<State, ArgCount> get_children_predict(const Data& d, 
                                                    TreeNode*& child1, 
                                                    TreeNode*& child2)
        {
            return get_children(d, child1, child2, &TreeNode::predict);
        }

        /// Utility to grab child outputs for variable arity nodes.
        vector<State> get_variable_children(const Data& d,
                                            TreeNode*& child1, 
                                            TreeNode*& child2, 
                                            State (TreeNode::*fn)(const Data&)
                                            )
        {
            vector<State> child_outputs;

            TreeNode* sib = child1;
            while (sib != child2)
            {
                child_outputs.push_back((sib->*fn)(d));
                sib = sib->next_sibling;
            }
            return child_outputs;
        };

        vector<State> get_variable_children_fit(const Data& d, 
                                                TreeNode*& child1, 
                                                TreeNode*& child2)
        {
            return get_variable_children(d, child1, child2, &TreeNode::fit);
        }
        vector<State> get_variable_children_predict(const Data& d, 
                                                    TreeNode*& child1, 
                                                    TreeNode*& child2)
        {
            return get_variable_children(d, child1, child2, &TreeNode::predict);
        }
};



/* Specialization of Node for simple functions 
 * (no weights, no gradients, really quite sad, pitiful creatures) 
 * 
 * */
template<typename R, typename... Args>
class Node<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        using TupleArgs = typename base::TupleArgs;

        /// the function applied to data
        Function op; 

        Node(string name, const Function& x) : base(name), op(x)
        {
            this->set_name("Node(" +this->name + ")");
        };

        State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override 
	    {
            TupleArgs inputs = base::tupleize(
                base::get_children_fit(d, child1, child2));

 			return std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode*& child1, 
                TreeNode*& child2) override
	    {
            auto child_outputs = base::get_children_predict(d, child1, child2);
            TupleArgs inputs = base::tupleize(child_outputs);

 			return std::apply(this->op, inputs);
        };
};

} // nodes
} // Brush

#endif
