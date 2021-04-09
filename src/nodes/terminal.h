/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef TERMINAL_H
#define TERMINAL_H

namespace Brush {
namespace nodes {

// defined nodes:
template<typename F> class Terminal; 

/* specialization of Node for terminals */
template<typename R>
class Terminal: public TypedNodeBase<R>
{
    public:
        using base = TypedNodeBase<R>;
        string variable_name;

        Terminal(string variable_name, R& value) : base("terminal")
        {
            /* @param variable_name: name of the variable
            *  @param value: the value, to grab its reference type
            */
            this->variable_name = variable_name;
        };

        State fit(const Data& d, TreeNode*& child1=0, TreeNode*& child2=0) override
        {
            return this->predict(d, child1, child2);
        }
        State predict(const Data& d, TreeNode*& child1, 
                      TreeNode*& child2) override
        {
            State out = d[variable_name];
            return out;
        };
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& child1, TreeNode*& child2) override 
        {
            this->set_prob_change(gradient.matrix().norm());
        };

        string get_model(TreeNode*& child1=0, TreeNode*& child2=0) const override
        { 
            return this->variable_name;
        };
};

} // nodes
} // Brush
#endif