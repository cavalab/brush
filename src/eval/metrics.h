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

/**
 * @brief Precision for binary classification (threshold 0.5, positive label 1).
 * @details Equivalent to sklearn's `precision_score(zero_division=0)`. Class
 * weights are used as sample weights. The loss vector holds the per-sample
 * misclassification indicator (used in lexicase selection).
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The precision.
 */
float precision_score(const VectorXf& y, const VectorXf& predict_proba,
                      VectorXf& loss,
                      const vector<float>& class_weights=vector<float>() );

/**
 * @brief Recall for binary classification (threshold 0.5, positive label 1).
 * @details Equivalent to sklearn's `recall_score(zero_division=0)`. The loss
 * vector holds the per-sample misclassification indicator.
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The recall.
 */
float recall_score(const VectorXf& y, const VectorXf& predict_proba,
                   VectorXf& loss,
                   const vector<float>& class_weights=vector<float>() );

/**
 * @brief Area under the ROC curve for binary classification.
 * @details Equivalent to sklearn's `roc_auc_score`. Returns 0.5 when only one
 * class is present (where the metric is undefined). The loss vector holds the
 * per-sample log loss (used in lexicase selection).
 * @param y The true labels.
 * @param predict_proba The predicted probabilities.
 * @param loss Reference to store the calculated losses for each sample.
 * @param class_weights The optional class weights.
 * @return The AUROC.
 */
float roc_auc_score(const VectorXf& y, const VectorXf& predict_proba,
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

/// Balanced accuracy for multi-classification.
float multi_bal_zero_one_loss(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss,
                        const vector<float>& class_weights=vector<float>() );

/**
 * @brief Macro-averaged precision for multi-classification.
 * @details Equivalent to sklearn's `precision_score(average='macro',
 * zero_division=0)`: averages over classes present in either the true or the
 * predicted labels. The loss vector holds the misclassification indicator.
 */
float multi_precision_score(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss,
                        const vector<float>& class_weights=vector<float>() );

/**
 * @brief Macro-averaged recall for multi-classification.
 * @details Equivalent to sklearn's `recall_score(average='macro',
 * zero_division=0)`. The loss vector holds the misclassification indicator.
 */
float multi_recall_score(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss,
                        const vector<float>& class_weights=vector<float>() );

/**
 * @brief Macro-averaged one-vs-rest AUROC for multi-classification.
 * @details Mean of the binary AUROC of each class against the rest, skipping
 * classes that are absent from `y`. The loss vector holds the per-sample
 * multinomial log loss.
 */
float multi_roc_auc_score(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss,
                        const vector<float>& class_weights=vector<float>() );

/**
 * @brief Macro-averaged one-vs-rest average precision for multi-classification.
 * @details Mean of the binary average precision of each class against the
 * rest, skipping classes that are absent from `y`. The loss vector holds the
 * per-sample multinomial log loss.
 */
float multi_average_precision_score(const VectorXf& y, const ArrayXXf& predict_proba,
                        VectorXf& loss,
                        const vector<float>& class_weights=vector<float>() );


} // metrics
} // Brush

#endif
