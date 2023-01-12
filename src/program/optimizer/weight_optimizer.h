/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Code below heavily inspired by heal-research/operon, Copyright 2019-2022 Heal Research
*/
#ifndef WEIGHT_OPTIMIZER_H
#define WEIGHT_OPTIMIZER_H

#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/tiny_solver.h>

#include "tiny_cost_function.h"

namespace Brush
{

struct OptimizerSummary {
    double InitialCost;
    double FinalCost;
    int Iterations;
    int FunctionEvaluations;
    int JacobianEvaluations;
    bool Success;
};

template<typename PT>
struct ResidualEvaluator {
    typedef float Scalar; 
    ResidualEvaluator(PT& program, Dataset const& dataset)
        : program_(program)
        , dataset_(dataset)
        , numParameters_(program.get_weights().size())
        , y_true_(dataset.y)
    {}

    template<typename T>
    auto operator()(Eigen::DenseBase<T>& parameters, Eigen::DenseBase<T>& residuals) const noexcept -> void
    {
        return (*this)(parameters.data(), residuals.data());
    }

    template <typename T>
    auto operator()(T const* parameters, T* residuals) const -> bool
    {
        using ArrayType = Array<T, Dynamic, 1>; // ColMajor?
        const T ** new_weights = &parameters; 

        ArrayType y_pred = GetProgram().template predict_with_weights<ArrayType>(
            GetDataset(), 
            new_weights
        );

        auto residualMap = ArrayType::Map(residuals, GetDataset().get_n_samples());

        residualMap = (y_pred - GetTarget()); 

        return true;
    }

    [[nodiscard]] auto NumParameters() const -> size_t { return numParameters_; }
    [[nodiscard]] auto NumResiduals() const -> size_t { return y_true_.get().size(); }
    inline auto GetProgram() const { return program_.get();};
    inline auto GetDataset() const { return dataset_.get();};
    inline auto GetTarget() const { return y_true_.get();};

private:
    std::reference_wrapper<PT> program_;
    std::reference_wrapper<Dataset const> dataset_;
    std::reference_wrapper<ArrayXf const> y_true_;
    size_t numParameters_; // cache the number of parameters in the tree
};

struct WeightOptimizer
{
    // put ceres stuff in here!
    template<typename PT>
    void update(PT& program, const Dataset& dataset)
    {
        // update weights to return new_weights using Non-linear least squares.
        // target: d.y
        // get a copy of the weights from the tree. 
        if (program.get_n_weights() == 0)
            return;
        fmt::print("number of weights: {}\n",program.get_n_weights());
        auto init_weights = program.get_weights();

        ceres::Solver::Summary summary;

        ResidualEvaluator<PT> re(program, dataset);
        Brush::TinyCostFunction<ResidualEvaluator<PT>, fJet, float, Eigen::ColMajor> cost_function(re);
        ceres::TinySolver<decltype(cost_function)> solver;

        typename decltype(solver)::Parameters parameters = program.get_weights(); 
        solver.Solve(cost_function, &parameters);
    
        fmt::print("Summary:\nInitial cost: {}\nFinal Cost: {}\nIterations: {}\n",
            solver.summary.initial_cost,
            solver.summary.final_cost,
            solver.summary.iterations
        );
        fmt::print("Initial weights: {}\nFinal weights: {}\n", 
            init_weights, 
            parameters
        );
        // summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        if (summary.final_cost < summary.initial_cost)
        {
            program.set_weights(parameters);
        }

////////////////////////////////////////////////////////////////////////////////
    }
};

}
#endif