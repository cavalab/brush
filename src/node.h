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
using Eigen::ArrayXf;
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
        virtual State fit(const Data&, TreeNode*&, TreeNode*&) = 0;
        virtual State predict(const Data&, TreeNode*, TreeNode*) = 0;
        virtual State grad_descent(const ArrayXf&, TreeNode*&, TreeNode*&) = 0;
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
template<typename F> class NodeDx; 
template<typename F> class NodeSplit; 

/* specialization of Node for terminals */
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
        State grad_descent(const ArrayXf&, TreeNode*&, TreeNode*&) override {};
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
			/* State out; */
			/* std::get<R>(out) = d.X.row(this->loc); */ 
            cout << "returning " << d.X.row(this->loc).transpose() << endl;
            return ArrayXf(d.X.row(this->loc));
        };
        State predict(const Data& d, TreeNode* child1, 
                      TreeNode* child2) override 
        {
            return ArrayXf(d.X.row(this->loc));
        };
        State grad_descent(const ArrayXf& d, TreeNode*& child1, 
                           TreeNode*& child2) override {};
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


        State grad_descent(const ArrayXf&, TreeNode*&, TreeNode*&) override {};

};

/* Node for differentiable functions 
 * 
 * */
template<typename R, typename... Args>
/* class NodeDx<ArrayXf(Args...)> : public TypedNodeBase<R(Args...)> */
/* template<Array<typename T,-1,1>(Array<T,-1,1>... Args)> */
/* template<ArrayXf R, ArrayXf ... Args> */
/* template<> */
class NodeDx<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        /* using Args = ... ArrayXf; */
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        using TupleArgs = typename base::TupleArgs;
        // weight types
        using WTypes = std::array<float, base::ArgCount>;
        // derivative function type
        using DxFunction = std::function<WTypes(float...)>;
        /// the function applied to data
        Function op; 
        /// the derivative of the function wrt each input
        DxFunction d_op; 

        /// the weights associated with each input
        WTypes W;
        /// the momentum of the weights associated with each input
        WTypes V;


        NodeDx(string name, const Function& f, const DxFunction& df ) 
        {
            this->op = f;
            this->d_op = df; 
			this->name = name;
            this->V = {};
            this->W = { 1.0 };
        };

        State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override 
	    {
            cout << "fitting " << this->name << endl;
            cout << "child1: " << &child1 << endl;
            cout << "child2: " << &child2 << endl;
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
            TupleArgs inputs = base::set_inputs(child_outputs, 
                                std::make_index_sequence<sizeof...(Args)>{}
                                );

            cout << "applying weights to " << this->name << " operator\n";
            inputs = std::transform(inputs.begin(), inputs.end(),
                                    W.begin(), std::multiplies<>());

            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            cout << "returning " << std::get<R>(out) << endl;

            this->store_gradients(inputs);

 			return std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode* child1, 
                TreeNode* child2) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "child1: " << child1 << endl;
            cout << "child2: " << child2 << endl;
            array<State, base::ArgCount> child_outputs;

            cout << "gathering inputs..." << endl;
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

            cout << "applying weights to " << this->name << " operator\n";
            inputs = std::transform(inputs.begin(), inputs.end(),
                                    W.begin(), std::multiplies<>());
            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            cout << "returning " << std::get<R>(out) << endl;
 			return std::apply(this->op, inputs);
        };

        State grad_descent(const ArrayXf& gradient, TreeNode*& child1, 
                               TreeNode*& child2) override
        {
    /*     this->update_weights(d, gradient); */
    /*     child1->backprop(d, gradient*ddx.at(0)); */
    /*     child2->backprop(d, gradient*ddx.at(1)); */

        };

    private:


        void store_gradients(const typename base::TupleArgs& inputs)
        {
            /* Here we store the derivatives of the output w.r.t. 
             * the inputs (ddx, used to backpropagate the gradient) 
             * and the edge weights (ddw, used to update these weights).
             */
            // TODO: this isn't going to work generically with d_op... but
            // there should be an elegant way to provide a derivative function
            // that respects datatypes as well as weights types
            TupleArgs d_inputs = std::apply(this->d_op, 
                                    inputs);
            this->df_dw = std::apply(this->d_op, inputs);
            this->df_dx = std::apply(this->d_op, this->W);
/*         this->g.clear(); */
/*         ddw.push_back(x1); */
/*         ddw.push_back(x2); */
/*         //ddx0 */
/*         ddx.push_back(W.at(0)); */
/*         //ddx1 */
/*         ddx.push_back(W.at(1)); */
        };

        void update_weights(const ArrayXf& gradient)
        {
            /*! update weights via gradient descent + momentum
             * @param lr : learning rate
             * @param m : momentum
             * v(t+1) = a * v(t) - n * gradient
             * w(t+1) = w(t) + v(t+1)
             * */
            std::cout << "***************************\n";
            std::cout << "Updating " << this->name << "\n";

            // Update all weights
            std::cout << "Current gradient" << gradient << "\n";
            array<float, base::ArgCount> W_temp(W);
            array<float, base::ArgCount> V_temp(V);
            float lr = 0.1;
            float m = 0.1; 
            
            cout << "learning_rate is "<< lr <<"\n"; 
            // Have to use temporary weights so as not to compute updates with 
            // updated weights
            for (int i = 0; i < base::ArgCount; ++i) 
            {
                std::cout << "V[i]: " << V[i] << "\n";
                V_temp[i] = (m * V.at(i) 
                             - lr * (gradient*this->ddw.at(i)).mean() );
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
            std::cout << "***************************\n";
        };


};
/* Node for split functions 
 * 
 * */
/* template<typename R, typename... Args>*/
/* class NodeSplit<R(Args...)> : public TypedNodeBase<R, Args...>*/
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

/*         State grad_descent(const ArrayXf& gradient, TreeNode*& child1,*/ 
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
