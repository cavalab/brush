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
#include <Eigen/Dense>
#include <stdexcept>
using Eigen::ArrayXf;
using namespace std;

/* TODO:
 * - fit and predict
 *   - get nodes working in a tree
 *   - proper moving functions eg swap, etc. in base class
 */

class NodeBase {
	public:
        typedef tree_node_<NodeBase*> TreeNode;  
		string name;
        virtual State fit(const Data&, TreeNode*&, TreeNode*&) = 0;
        virtual State predict(const Data&, TreeNode*, TreeNode*) = 0;
        virtual void grad_descent(const ArrayXf&, const Data&, 
                                   TreeNode*&, TreeNode*&) = 0;
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
		
        string name;

        void grad_descent(const ArrayXf&, const Data&, 
                           TreeNode*&, TreeNode*&) override 
        {
            throw runtime_error("grad_descent not implemented for " + name);
        };

    protected:
        template<size_t... Is>
		TupleArgs set_inputs(const array<State, ArgCount>& in, 
					std::index_sequence<Is...>)
		{ 
            return std::make_tuple(std::get<NthType<Is>>(in.at(Is))...);
		};
};

/* Declaration of Node as a templated class */
template<typename F> class Node; 
template<typename F> class WeightedNode; 
template<typename F> class SplitNode; 

/* specialization of Node for terminals */
template<typename R>
class Node: public TypedNodeBase<R>
{
    public:
        using base = TypedNodeBase<R>;
        string variable_name;
        string name;
		unsigned int loc;

        Node(string name, int loc) 
        {
            this->variable_name = name;
			this->name = this->variable_name;
			this->loc = loc;
            /* this->op = x(); */
            /* TupleArgs types; */ 
            /* cout << "function: " << x << endl; */
            cout << "RetType: " << typename base::RetType() << endl;
            cout << "ArgCount: " << base::ArgCount << endl;
            /* cout << "TupleArgs: " << TupleArgs(); */
            /* for (auto at : types) cout << at; */
        };

        State fit(const Data& d, TreeNode& child1=0, TreeNode& child2=0)
        {
            //TODO: this needs to be specialized for different terminal types
            //that deal directly with data.
			/* State out; */
			/* std::get<R>(out) = d.X.row(this->loc); */ 
            return 0;
            /* return d.X.row(this->loc); */
        }
        State predict(const Data&, TreeNode*, TreeNode*) override {};
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& child1, TreeNode*& child2) override {};
};

/* specialization of Node for Array terminals */
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
						   TreeNode*& child1, 
						   TreeNode*& child2) override
	    {
            cout << "returning " << d.X.row(this->loc) << endl;
            return ArrayXf(d.X.row(this->loc));
        };
        State predict(const Data& d, TreeNode* child1, 
                      TreeNode* child2) override 
        {
            return ArrayXf(d.X.row(this->loc));
        };
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& child1, TreeNode*& child2) override {};
};

/* Specialization of Node for simple functions (no weights) 
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

        Node(string name, const Function& x) 
        {
			this->name = name;
            this->op = x;
        };

        State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override 
	    {
            array<State, base::ArgCount> child_outputs;

            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                child_outputs.at(i) = sib->fit(d);
                sib = sib->next_sibling;
            }
            TupleArgs inputs = base::set_inputs(child_outputs, 
                                std::make_index_sequence<sizeof...(Args)>{}
                                );

 			return std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode* child1, 
                TreeNode* child2) override
	    {
            array<State, base::ArgCount> child_outputs;

            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                cout << "sibling: " << sib << endl;
                cout << "sibling name: " << sib->data->name << endl;
                child_outputs.at(i) = sib->predict(d);
                sib = sib->next_sibling;
            }
            TupleArgs inputs = base::set_inputs(child_outputs, 
                                std::make_index_sequence<sizeof...(Args)>{}
                                );

 			return std::apply(this->op, inputs);
        };



};

/* Node for weighted functions.
 * 
 * Restrictions: all argument datatypes and the return type must match.
 * */
template<typename R, typename... Args>
/* class WeightedNode<ArrayXf(Args...)> : public TypedNodeBase<R(Args...)> */
/* template<Array<typename T,-1,1>(Array<T,-1,1>... Args)> */
/* template<ArrayXf R, ArrayXf ... Args> */
/* template<> */
class WeightedNode<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        /* using Args = ... ArrayXf; */
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        using TupleArgs = typename base::TupleArgs;
        using ArrayArgs = std::array<R, base::ArgCount>;
        // weight types
        using WTypes = std::array<float, base::ArgCount>;
        // derivative function type
        using DxFunction = std::function<ArrayArgs(Args...)>;
        /// the function applied to data
        Function op; 
        /// the derivative of the function wrt each input
        DxFunction d_op; 

        /// the weights associated with each input
        WTypes W;
        /// the momentum of the weights associated with each input
        WTypes V;
        /// partial derivative w.r.t. the weights, used to update W
        ArrayArgs df_dW;
        /// partial derivative w.r.t. the inputs, to propagate the gradient
        ArrayArgs df_dX; 

        WeightedNode(string name, const Function& f, const DxFunction& df,
                     const vector<float>& Win = {})
        {
            this->op = f;
            this->d_op = df; 
			this->name = name;
            this->V.fill(0.0);
            if (Win.empty())
                this->W.fill(1.0);
            else
                std::move(Win.begin(), Win.begin()+base::ArgCount, 
                        this->W.begin());

            cout << name << " Win: ";
            for (auto w: Win)
                cout << w << " ";
            cout << endl;

            cout << name << " weights: ";
            for (auto w: this->W)
                cout << w << " ";
            cout << endl;
        };

        State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override 
	    {
            cout << "fitting " << this->name << endl;
            cout << "child1: " << &child1 << endl;
            cout << "child2: " << &child2 << endl;
            array<R, base::ArgCount> inputs;

            cout << "gathering inputs..." << endl;
            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                cout << "sibling: " << sib << endl;
                cout << "sibling name: " << sib->data->name << endl;
                inputs.at(i) = std::get<R>(sib->fit(d));
                sib = sib->next_sibling;
            }

            this->store_gradients(inputs);

            cout << "applying weights to " << this->name << " operator\n";
            std::transform(W.begin(), W.end(), inputs.cbegin(),
                           inputs.begin(), std::multiplies<>());

            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            cout << "returning " << std::get<R>(out) << endl;


 			return std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode* child1, 
                TreeNode* child2) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "child1: " << child1 << endl;
            cout << "child2: " << child2 << endl;
            array<R, base::ArgCount> inputs;

            cout << "gathering inputs..." << endl;
            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                cout << "sibling: " << sib << endl;
                cout << "sibling name: " << sib->data->name << endl;
                inputs.at(i) = std::get<R>(sib->predict(d));
                sib = sib->next_sibling;
            }

            cout << "applying weights to " << this->name << " operator\n";
            std::transform(W.begin(), W.end(), inputs.cbegin(),
                           inputs.begin(), std::multiplies<>());
            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            cout << "returning " << std::get<R>(out) << endl;
 			return std::apply(this->op, inputs);
        };

        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& child1, TreeNode*& child2) override
        {
            /* backpropagate the gradient * df_dX. 
             * update internal weights. 
             */
            cout << "gradient descent on " << this->name << endl;
            TreeNode* sib = child1;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                sib->grad_descent(gradient*this->df_dX.at(i), d);
                sib = sib->next_sibling;
            }

            this->update_weights(gradient);
        };

    private:

        void store_gradients(const ArrayArgs& inputs)
        {
            /* Here we store the derivatives of the output w.r.t. 
             * the inputs (df_dX, used to backpropagate the gradient) 
             * and the edge weights (df_dW, used to update these weights).
             *
             * it's important that argument inputs enters this 
             * function before scaling by the weights, W.
             */
            cout << "storing gradients for " << this->name << endl;
            ArrayArgs df_dIn = std::apply(this->d_op, inputs);

            std::transform(W.begin(), W.end(), df_dIn.begin(),
                           this->df_dX.begin(), std::multiplies<>());    

            std::transform(df_dIn.begin(), df_dIn.end(), inputs.begin(),
                           this->df_dW.begin(), std::multiplies<>());    
        };

        void update_weights(const ArrayXf& gradient)
        {
            /*! update weights via gradient descent + momentum
             * @param lr : learning rate
             * @param m : momentum
             * v(t+1) = m * v(t) - lr * gradient
             * w(t+1) = w(t) + v(t+1)
             *
             * TODO: move the optimizer-specific functionality of this method
             * to a separate class
             * */
            std::cout << "***************************\n";
            std::cout << "Updating " << this->name << "\n";

            // Update all weights
            std::cout << "Current gradient" << gradient.transpose() << "\n";
            cout << "Current weights: ";
            for (const auto& w: W) cout << w << " ";
            cout << endl;
            array<float, base::ArgCount> W_temp(W);
            array<float, base::ArgCount> V_temp(V);
            float lr = 0.25;
            float m = 0.1; 
            
            cout << "learning_rate is "<< lr <<"\n"; 
            // Have to use temporary weights so as not to compute updates with 
            // updated weights
            for (int i = 0; i < base::ArgCount; ++i) 
            {
                cout << "dL/dW[" << i << "]: " 
                    << (gradient*this->df_dW.at(i)).mean() << endl;
                std::cout << "V[i]: " << V[i] << "\n";
                V_temp[i] = (m * V.at(i) 
                             - lr * (gradient*this->df_dW.at(i)).mean() );
                std::cout << "V_temp: " << V_temp[i] << "\n";
            }
            for (int i = 0; i < W.size(); ++i)
            {
                if (std::isfinite(V_temp[i]) && !std::isnan(V_temp[i]))
                {
                    this->W[i] += V_temp[i];
                    this->V[i] = V_temp[i];
                }
            }

            std::cout << "Updated\n";
            cout << "Weights: ";
            for (const auto& w: W) cout << w << " ";
            std::cout << "\n***************************\n";
        };


};
/* Node for split functions 
 * 
 * */
/* template<typename R, typename... Args>*/
/* class SplitNode<R(Args...)> : public TypedNodeBase<R, Args...>*/
/* {*/
/*     public:*/
/*         using base = TypedNodeBase<R, Args...>;*/
/*         using Function = std::function<R(Args...)>;*/
/*         using TupleArgs = typename base::TupleArgs;*/

/*         /// the function applied to data*/
/*         Function op;*/ 
/*         /// the learned feature choice*/
/*         unsigned int loc;*/
/*         /// the learned threshold*/
/*         float threshold;*/

/*         Node(const Function& x, string name)*/ 
/*         {*/
/*             this->op = x;*/
/* 			this->name = name;*/
/*             this->V = {};*/
/*             this->W = { 1.0 };*/
/*         };*/

/*         State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override*/ 
/* 	    {*/

            /* 1) choose best feature
             * 2) choose best threshold of feature
             * 3) split data on feature at threshold
             * 4) evaluate child nodes on split data
             * 5) stitch child outputs together and return
             */

/*             cout << "fitting " << this->name << endl;*/
/*             cout << "child1: " << &child1 << endl;*/
/*             cout << "child2: " << &child2 << endl;*/
/*             array<State, base::ArgCount> child_outputs;*/

/*             cout << "gathering inputs..." << endl;*/
/*             TreeNode* sib = child1;*/
/*             for (int i = 0; i < base::ArgCount; ++i)*/
/*             {*/
/*                 cout << i << endl;*/
/*                 cout << "sibling: " << sib << endl;*/
/*                 cout << "sibling name: " << sib->data->name << endl;*/
/*                 child_outputs.at(i) = sib->fit(d);*/
/*                 sib = sib->next_sibling;*/
/*             }*/
/*             TupleArgs inputs = set_inputs(child_outputs,*/ 
/*                                 std::make_index_sequence<sizeof...(Args)>{}*/
/*                                 );*/

/*             cout << "applying " << this->name << " operator\n";*/
/*             State out = std::apply(this->op, inputs);*/
/*             cout << "returning " << std::get<R>(out) << endl;*/

/*             this->store_gradients(inputs);*/

/*  			return std::apply(this->op, inputs);*/
/*         };*/

/*         State predict(const Data& d, TreeNode* child1,*/ 
/*                 TreeNode* child2) override*/
/* 	    {*/
/*             cout << "predicting " << this->name << endl;*/
/*             cout << "child1: " << child1 << endl;*/
/*             cout << "child2: " << child2 << endl;*/
/*             array<State, base::ArgCount> child_outputs;*/

/*             cout << "gathering inputs..." << endl;*/
/*             TreeNode* sib = child1;*/
/*             for (int i = 0; i < base::ArgCount; ++i)*/
/*             {*/
/*                 cout << i << endl;*/
/*                 cout << "sibling: " << sib << endl;*/
/*                 cout << "sibling name: " << sib->data->name << endl;*/
/*                 child_outputs.at(i) = sib->predict(d);*/
/*                 sib = sib->next_sibling;*/
/*             }*/
/*             TupleArgs inputs = set_inputs(child_outputs,*/ 
/*                                 std::make_index_sequence<sizeof...(Args)>{}*/
/*                                 );*/

/*             cout << "applying " << this->name << " operator\n";*/
/*             State out = std::apply(this->op, inputs);*/
/*             cout << "returning " << std::get<R>(out) << endl;*/
/*  			return std::apply(this->op, inputs);*/
/*         };*/

/*         void grad_descent(const ArrayXf& gradient, TreeNode*& child1,*/ 
/*                                TreeNode*& child2) override*/
/*         {*/
/*          this->update_weights(d, gradient); */
/*          child1->backprop(d, gradient*ddx.at(0)); */
/*          child2->backprop(d, gradient*ddx.at(1)); */

/*         };*/

/*     private:*/
/* // TODO: move to shared parent class*/
/* 		template<size_t... Is>*/
/* 		TupleArgs set_inputs(const array<State, base::ArgCount>& in,*/ 
/* 					std::index_sequence<Is...>)*/
/* 		{*/ 
/*             return std::make_tuple(std::get<base::NthType<Is>>(in.at(Is))...);*/
/* 		};*/


/* };*/


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
/* 	/1* using NthArg = std::tuple_element_t<N, TupleArgs>; *1/ */
/* 	/1* using FirstArg = NthArg<0>; *1/ */
/* 	/1* using LastArg = NthArg<base::ArgCount - 1>; *1/ */

/*     Function op; */
/*     /1* Node(Function x) *1/ */
/*     Node() */
/*     { */
/*         cout << "RetType: " << this->RetType; */
/*         cout << "ArgCount: " << this->ArgCount; */
/*         cout << "TupleArgs: "; */
/*         for (auto at : this->TupleArgs) cout << at; */
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

#endif
