#ifndef METRICS_H
#define METRICS_H

#include "../data/data.h"
#include "../util/utils.h"

namespace Brush {
/**
 * @namespace Eval
 * @brief Namespace containing scoring functions for evaluation metrics.
 */
namespace Eval {

/* Scoring functions */

// regression ------------------------------------------------------------------

/**
 * @brief Calculates the mean squared error between the predicted values and the true values.
 * @param y The true values.
 * @param yhat The predicted values.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights (not used for MSE).
 * @return The mean squared error.
 */
float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
            const vector<float>& class_weights=vector<float>() );

// binary classification ------------------------------------------------------- 

/**
 * @brief Calculates the log loss between the predicted probabilities and the true labels.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param class_weights The optional class weights.
 * @return The log loss.
 */
VectorXf log_loss(const VectorXf& y, const VectorXf& predict_proba, 
                    const vector<float>& class_weights=vector<float>());

/**
 * @brief Calculates the mean log loss between the predicted probabilities and the true labels.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The mean log loss.
 */
float mean_log_loss(const VectorXf& y, const VectorXf& predict_proba, VectorXf& loss,
                    const vector<float>& class_weights = vector<float>());

/**
 * @brief Calculates the average precision score between the predicted probabilities and the true labels.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The average precision score.
 */
float average_precision_score(const VectorXf& y, const VectorXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

/**
 * @brief Accuracy for binary classification
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The final accuracy.
 */
float zero_one_loss(const VectorXf& y, const VectorXf& predict_proba,
                        VectorXf& loss, 
                        const vector<float>& class_weights=vector<float>() );

/**
 * @brief Balanced accuracy for binary classification
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The final accuracy.
 */
float bal_zero_one_loss(const VectorXf& y, const VectorXf& predict_proba,
                        VectorXf& loss, 
                        const vector<float>& class_weights=vector<float>() );
                
// multiclass classification ---------------------------------------------------

/**
 * @brief Calculates the multinomial log loss between the predicted probabilities and the true labels.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param class_weights The optional class weights.
 * @return The multinomial log loss.
 */
VectorXf multi_log_loss(const VectorXf& y, const ArrayXXf& predict_proba, 
        const vector<float>& class_weights=vector<float>());

/**
 * @brief Calculates the mean multinomial log loss between the predicted probabilities and the true labels.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The mean multinomial log loss.
 */
float mean_multi_log_loss(const VectorXf& y, const ArrayXXf& predict_proba,
                          VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

/**
 * @brief Accuracy for multi-classification
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The average accuracy in a one-vs-all schema.
 */
float multi_zero_one_loss(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss, 
                        const vector<float>& class_weights=vector<float>() );


} // metrics
} // Brush

#endif