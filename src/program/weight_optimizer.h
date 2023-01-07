#ifndef WEIGHT_OPTIMIZER_H
#define WEIGHT_OPTIMIZER_H
// #include "../init.h"
// #include "program.h"
// #include "../data/data.h"

#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>
#include <ceres/tiny_solver.h>

#include "optimizer/tiny_cost_function.hpp"
namespace Brush
{
    using Scalar = float;
    using Dual = ceres::Jet<Scalar, 1>;
// auto mean_squared_error(ArrayXf y, ArrayXf y_pred)
// {
//     return (y-y_pred).norm();
// }

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
    {
    }

    template<typename T>
    auto operator()(Eigen::DenseBase<T>& parameters, Eigen::DenseBase<T>& residuals) const noexcept -> void
    {
        return (*this)(parameters.data(), residuals.data());
    }

    template <typename T>
    auto operator()(T const* parameters, T* residuals) const -> bool
    {
        using ArrayType = Array<T, Dynamic, 1>; // ColMajor?
        // Map<ArrayType const> new_weights(parameters, numParameters_);
        // T const * p2 = parameters;
        const T ** new_weights = &parameters; 
        // auto new_weights = &parameters;

        // GetProgram().set_weights(new_weights);
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
        auto init_weights = program.get_weights();
        int n_weights = init_weights.size();
        
        // auto residual_functor = ResidualEvaluator<PT>(program, dataset);
        // auto cost_function = ceres::DynamicAutoDiffCostFunction<ResidualEvaluator<PT>>(
        //     &residual_functor
        // );
        // cost_function.AddParameterBlock(n_weights);
        // cost_function.SetNumResiduals(residual_functor.NumResiduals());

        // ceres::TinySolver<ceres::DynamicAutoDiffCostFunction<ResidualEvaluator<PT>>> solver;
        // solver.options.max_num_iterations = 1000;
        ceres::Solver::Summary summary;

        ResidualEvaluator<PT> re(program, dataset);
        Brush::TinyCostFunction<ResidualEvaluator<PT>, Dual, float, Eigen::ColMajor> cost_function(re);
        ceres::TinySolver<decltype(cost_function)> solver;
        // auto x0 = weights;
        // auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()); 
        if (n_weights > 0) {
            typename decltype(solver)::Parameters parameters = program.get_weights(); 
            solver.Solve(cost_function, &parameters);
            // m0 = p.cast<Operon::Scalar>();
        
            // std::cout << summary.BriefReport() << "\n";
            fmt::print("Summary:\nInitial cost: {}\nFinal Cost: {}\nIterations: {}\n",
                solver.summary.initial_cost,
                solver.summary.final_cost,
                solver.summary.iterations
            );
            fmt::print("Initial weights: {}\nFinal weights: {}", 
                init_weights, 
                parameters
            );
            // summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
            // return x0;
        }

////////////////////////////////////////////////////////////////////////////////
        // ceres::DynamicCostFunction* costFunction = nullptr;
        // ResidualEvaluator re(interpreter, tree, dataset, target, range);
        // TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::RowMajor> f(re);
        // costFunction = new Operon::DynamicCostFunction<decltype(f)>(f);

        // auto sz = static_cast<Eigen::Index>(x0.size());
        // Eigen::MatrixXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(x0.data(), sz).template cast<double>();
        // ceres::Problem problem;
        // problem.AddResidualBlock(costFunction, nullptr, params.data());

        // ceres::Solver::Options options;
        // options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
        // options.linear_solver_type = ceres::DENSE_QR;
        // options.minimizer_progress_to_stdout = report;
        // options.num_threads = 1;
        // options.logging_type = ceres::LoggingType::SILENT;

        // ceres::Solver::Summary s;
        // Solve(options, &problem, &s);
        // sum.InitialCost = s.initial_cost;
        // sum.FinalCost = s.final_cost;
        // sum.Iterations = static_cast<int>(s.iterations.size());
        // sum.FunctionEvaluations = s.num_residual_evaluations;
        // sum.JacobianEvaluations = s.num_jacobian_evaluations;
        // summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        // return x0;
    }
};

}
#endif