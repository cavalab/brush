/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef DX_H
#define DX_H
#include "base.h"

namespace Brush {
namespace nodes {

// nodes herein defined: 
template<typename F> class DxNode; 
template<typename F> class ReduceDxNode; 
template<typename F> class WeightedDxNode; 
template<typename F> class TransformReduceDxNode; 
// template<typename F> class TransformDxNode; 

/* Node for differentiable functions.
 * 
 * Restrictions: all argument datatypes and the return type must match.
 * */
template<typename R, typename... Args>
class DxNode<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        /* using Args = ... ArrayXf; */
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<R(Args...)>;
        using TupleArgs = typename base::TupleArgs;
        using ArrayArgs = std::array<R, base::ArgCount>;
        // derivative function type
        using DxFunction = std::function<ArrayArgs(Args...)>;
        /// the function applied to data
        Function op; 
        /// the derivative of the function wrt each input
        DxFunction d_op; 
        /// partial derivative w.r.t. the inputs, to propagate the gradient
        ArrayArgs df_dX; 

        DxNode(string name, const Function& f, const DxFunction& df)
        : base(name), op(f), d_op(df) 
        {
            this->set_name("DxNode(" +this->name + ")");
        };

        State fit(const Data& d, TreeNode*& first_child, TreeNode*& last_child) override 
	    {
            cout << "fitting " << this->name << endl;
            cout << "first_child: " << &first_child << endl;
            cout << "last_child: " << &last_child << endl;
            auto child_outputs = base::get_children_fit(d, first_child, last_child);
            // TODO: helper fn to convert children to ArrayArgs
            ArrayArgs inputs; // = std::apply(std::get<R>, child_outputs);
            for (int i = 0; i < child_outputs.size(); ++i)
                inputs.at(i) = std::get<R>(child_outputs.at(i));

            this->store_gradients(inputs);

            // cout << "applying " << this->name << " operator\n";
            // State out = std::apply(this->op, inputs);
            // cout << "returning " << std::get<R>(out) << endl;

 			return std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode*& first_child, 
                TreeNode*& last_child) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "first_child: " << first_child << endl;
            cout << "last_child: " << last_child << endl;
            auto child_outputs = base::get_children_predict(d, first_child, last_child);
            // TODO: helper fn to convert children to ArrayArgs
            ArrayArgs inputs; 
            for (int i = 0; i < child_outputs.size(); ++i)
                inputs.at(i) = std::get<R>(child_outputs.at(i));

            // cout << "applying " << this->name << " operator\n";
            // State out = std::apply(this->op, inputs);
            // cout << "returning " << std::get<R>(out) << endl;
 			return std::apply(this->op, inputs);
        };

        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& first_child, TreeNode*& last_child) override
        {
            /* backpropagate the gradient * df_dX. 
             */
            cout << "gradient descent on " << this->name << endl;
            TreeNode* sib = first_child;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                // chain rule
                sib->grad_descent(gradient*this->df_dX.at(i), d);
                sib = sib->next_sibling;
            }
            this->set_prob_change(gradient.matrix().norm());
        };

    private:

        void store_gradients(const ArrayArgs& inputs)
        {
            /* Here we store the derivatives of the output w.r.t. 
             * the inputs (df_dX, used to backpropagate the gradient) 
             */
            cout << "storing gradients for " << this->name << endl;
            this->df_dX = std::apply(this->d_op, inputs);
        };

};

/* Node for weighted, differentiable functions.
 * 
 * Restrictions: all argument datatypes and the return type must match.
 * */
template<typename R, typename... Args>
class WeightedDxNode<R(Args...)> : public DxNode<R(Args...)>
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

        /// the weights associated with each input
        WTypes W;
        /// the momentum of the weights associated with each input
        WTypes V;
        /// partial derivative w.r.t. the weights, used to update W
        std::array<ArrayXf, base::ArgCount> df_dW;

        WeightedDxNode(string name, const Function& f, const DxFunction& df,
                     const vector<float>& Win = {})
        : DxNode<R(Args...)>(name, f, df) 
        {
            this->set_name("Weighted" +this->name );

            cout << "initializing node " << name << endl;
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

        State fit(const Data& d, TreeNode*& first_child, TreeNode*& last_child) override 
	    {
            cout << "fitting " << this->name << endl;
            cout << "first_child: " << &first_child << endl;
            cout << "last_child: " << &last_child << endl;
            auto child_outputs = base::get_children_fit(d, first_child, last_child);
            // TODO: helper fn to convert children to ArrayArgs
            ArrayArgs inputs; // = std::apply(std::get<R>, child_outputs);
            for (int i = 0; i < child_outputs.size(); ++i)
                inputs.at(i) = std::get<R>(child_outputs.at(i));

            this->store_gradients(inputs);

            cout << "applying weights to " << this->name << " operator\n";
            std::transform(inputs.begin(), inputs.end(), W.begin(),
                           inputs.begin(), std::multiplies<>());

            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            // cout << "returning " << std::get<R>(out) << endl;


 			return out; //std::apply(this->op, inputs);
        };

        State predict(const Data& d, TreeNode*& first_child, 
                TreeNode*& last_child) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "first_child: " << first_child << endl;
            cout << "last_child: " << last_child << endl;
            auto child_outputs = base::get_children_predict(d, first_child, last_child);
            // TODO: helper fn to convert children to ArrayArgs
            ArrayArgs inputs; 
            for (int i = 0; i < child_outputs.size(); ++i)
                inputs.at(i) = std::get<R>(child_outputs.at(i));

            cout << "applying weights to " << this->name << " operator\n";
            std::transform(inputs.begin(), inputs.end(), W.begin(),
                           inputs.begin(), std::multiplies<>());
            cout << "applying " << this->name << " operator\n";
            State out = std::apply(this->op, inputs);
            // cout << "returning " << std::get<R>(out) << endl;
 			return std::apply(this->op, inputs);
        };

        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& first_child, TreeNode*& last_child) override
        {
            /* backpropagate the gradient * df_dX. 
             * update internal weights. 
             */
            DxNode<R(Args...)>::grad_descent(gradient, d, first_child, 
                                             last_child);
            this->update_weights(gradient);

            this->set_prob_change(gradient.matrix().norm());
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

            // df_dX = W*df_dIn
            std::transform(W.begin(), W.end(), df_dIn.begin(),
                           this->df_dX.begin(), std::multiplies<>());    
            // df_dW = df_dIn*X
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
                    << (gradient*this->df_dW.at(i)).matrix().mean() << endl;
                std::cout << "V[i]: " << V[i] << "\n";
                V_temp[i] = (m * V.at(i) 
                             - lr * (gradient*this->df_dW.at(i)).matrix().mean() );
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


/* } // Brush */
/* Node for differentiable functions.
 * 
 * Restrictions: all argument datatypes and the return type must match.
 * */
template<typename R, typename Arg>
class ReduceDxNode<R(Arg,Arg)> : public TypedNodeBase<R, Arg>
{
    //TODO:
    // override arg_types to return correct arity, args_type to be different
    public:
        /* using Args = ... ArrayXf; */
        using base = TypedNodeBase<R, Arg>;
        using Function = std::function<R(Arg)>;
        // derivative function type
        using DxFunction = std::function<vector<Arg>(Arg)>;
        /// the function applied to data
        Function op; 
        /// the derivative of the function wrt each input
        DxFunction d_op; 
        /// partial derivative w.r.t. the inputs, to propagate the gradient
        vector<Arg> df_dX; 
        /// arity of the node
        int arity;

        /// override args_type 
        type_index args_type() const override {return typeid(vector<Arg>);}; 

        size_t arg_count() const override {return arity;};

        ReduceDxNode(string name, const Function& f, const DxFunction& df, 
                    int arity)
        : base(name), op(f), d_op(df), arity(arity)
        {
            this->set_name("ReduceDxNode(" +this->name + ")");
        };

        State fit(const Data& d, TreeNode*& first_child, TreeNode*& last_child) override 
	    {
            cout << "fitting " << this->name << endl;
            vector<Arg> inputs = this->get_children_variable_fit(d, first_child, 
                                                                 last_child);

            return std::reduce(inputs.begin(), inputs.end(), this->op);
        };

        State predict(const Data& d, TreeNode*& first_child, 
                      TreeNode*& last_child) override
	    {
            cout << "predicting " << this->name << endl;
            vector<Arg> inputs = base::get_variable_children_predict(d, 
                                    first_child, last_child);

            return std::reduce(inputs.begin(), inputs.end(), this->op);
        };

        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& first_child, TreeNode*& last_child) override
        {
            /* backpropagate the gradient * df_dX. 
             */
            cout << "gradient descent on " << this->name << endl;
            TreeNode* sib = first_child;
            int i = 0;
            // while (sib <= last_child)
            for(int i = 0; i < this->arg_count(); ++i)
            {
                sib->grad_descent(gradient*float(this->df_dX.at(i)), d);
                sib = sib->next_sibling;
                ++i;
            }
            this->set_prob_change(gradient.matrix().norm());
        };

    private:

        template<size_t... Is>
        vector<type_index> get_arg_types(index_sequence<Is...>) const 
        {
            cout << "calling reduce's get_arg_types\n";
            return vector<type_index>(this->arity, typeid(Arg));
        }

};

// template<typename R, typename Arg>
// class TransformReduceDxNode<R(Arg)> : public TypedNodeBase<R, Arg>
// {
//     public:
//         /* using Args = ... ArrayXf; */
//         using base = TypedNodeBase<R, Arg>;
//         using Function = std::function<R(Arg)>;
//         // derivative function type
//         using DxFunction = std::function<array<Arg,1>(Arg)>;
//         /// the function applied to data
//         Function op; 
//         Function reduce_op; 
//         /// the derivative of the function wrt each input
//         DxFunction d_op; 
//         /// partial derivative w.r.t. the inputs, to propagate the gradient
//         vector<Arg> df_dX; 
//         /// the weights associated with each input
//         vector<float> W;
//         /// the momentum of the weights associated with each input
//         vector<float> V;
//         /// partial derivative w.r.t. the weights, used to update W
//         std::array<ArrayXf, base::ArgCount> df_dW;
//         /// arity of the node
//         int arity;

//         /// override args_type 
//         type_index args_type() const override {return typeid(vector<Arg>);}; 

//         size_t arg_count() const override {return arity;};

//         TransformReduceDxNode(string name, const Function& f, const DxFunction& df, 
//                     int arity)
//         : base(name), op(f), d_op(df), arity(arity)
//         {
//             this->set_name("TransformReduceDxNode(" +this->name + ")");
//         };

//         State fit(const Data& d, TreeNode*& first_child, TreeNode*& last_child) override 
// 	    {
//             cout << "fitting " << this->name << endl;
//             vector<Arg> inputs = this->get_variable_children_fit(d, first_child, 
//                                                                  last_child);

//             this->store_gradients(inputs);
//             // calc W'X
//             Arg wx = std::transform_reduce(inputs.begin(), inputs.end(), 
//                                            W.begin(), Arg(0.0));

//             R output = this->op(wx);
//             return output;
//         };

//         State predict(const Data& d, TreeNode*& first_child, 
//                       TreeNode*& last_child) override
// 	    {
//             cout << "predicting " << this->name << endl;
//             vector<Arg> inputs = base::get_variable_children_predict(d, 
//                                     first_child, last_child);

//             Arg wx = std::transform_reduce(inputs.begin(), inputs.end(), 
//                                            W.begin(), Arg(0.0));

//             R output = this->op(wx);
//             return output;
//         };

        
//         void grad_descent(const ArrayXf& gradient, const Data& d, 
//                           TreeNode*& first_child, TreeNode*& last_child) override
//         {
//             /* backpropagate the gradient * df_dX. 
//              * update internal weights. 
//              */
//             ReduceDxNode<R(Arg)>::grad_descent(gradient, d, first_child, 
//                                              last_child);
//             this->update_weights(gradient);

//             this->set_prob_change(gradient.matrix().norm());
//         };
// ;

//     private:

//         template<size_t... Is>
//         vector<type_index> get_arg_types(index_sequence<Is...>) const 
//         {
//             cout << "calling reduce's get_arg_types\n";
//             return vector<type_index>(this->arity, typeid(Arg));
//         }

//         void store_gradients(const vector<Arg>& inputs)
//         {
//             /* f = op(w*x)
//              * df/dx = (df/dIn)(dIn/dx) = d_op(inputs)*W = df_dIn*W
//              * df/dw = (df/dIn)(dIn/dw) = d_op(inputs)*X = df_dIn*X
//              * Here we store the derivatives of the output w.r.t. 
//              * the inputs (df_dX, used to backpropagate the gradient) 
//              * and the edge weights (df_dW, used to update these weights).
//              * 
//              * it's important that argument inputs enters this 
//              * function before scaling by the weights, W.
//              */
//             cout << "storing gradients for " << this->name << endl;
//             vector<Arg> df_dIn; 
//             for (const auto& in : inputs)
//                 df_dIn.push_back(this->d_op({in})[0]);

//             // df_dX = (d_op/d_in)(d_in/d_W) = df_dIn*W
//             std::transform(W.begin(), W.end(), df_dIn.begin(),
//                            this->df_dX.begin(), std::multiplies<>());    

//             // df_dW = (d_op/d_in)(d_in/d_w) = df_dIn*inputs
//             std::transform(df_dIn.begin(), df_dIn.end(), inputs.begin(),
//                            this->df_dW.begin(), std::multiplies<>());    
//         };

//         void update_weights(const ArrayXf& gradient)
//         {
//             /*! update weights via gradient descent + momentum
//              * @param lr : learning rate
//              * @param m : momentum
//              * v(t+1) = m * v(t) - lr * gradient
//              * w(t+1) = w(t) + v(t+1)
//              *
//              * TODO: move the optimizer-specific functionality of this method
//              * to a separate class
//              * */
//             std::cout << "***************************\n";
//             std::cout << "Updating " << this->name << "\n";

//             // Update all weights
//             std::cout << "Current gradient" << gradient.transpose() << "\n";
//             cout << "Current weights: ";
//             for (const auto& w: W) cout << w << " ";
//             cout << endl;
//             auto W_temp(W);
//             auto V_temp(V);
//             float lr = 0.25;
//             float m = 0.1; 
            
//             cout << "learning_rate is "<< lr <<"\n"; 
//             // Have to use temporary weights so as not to compute updates with 
//             // updated weights
//             for (int i = 0; i < this->get_arity(); ++i) 
//             {
//                 cout << "dL/dW[" << i << "]: " 
//                     << (gradient*df_dW.at(i)).matrix().mean() << endl;
//                 std::cout << "V[i]: " << V[i] << "\n";
//                 V_temp[i] = (m * V.at(i) 
//                              - lr * (gradient*df_dW.at(i)).matrix().mean());
//                 std::cout << "V_temp: " << V_temp[i] << "\n";
//             }
//             for (int i = 0; i < W.size(); ++i)
//             {
//                 if (std::isfinite(V_temp[i]) && !std::isnan(V_temp[i]))
//                 {
//                     this->W[i] += V_temp[i];
//                     this->V[i] = V_temp[i];
//                 }
//             }

//             std::cout << "Updated\n";
//             cout << "Weights: ";
//             for (const auto& w: W) cout << w << " ";
//             std::cout << "\n***************************\n";
//         };
// };
} // nodes
} // Brush
#endif
