import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    classification_report
)
from scipy.stats import t
import xgboost as xgb
import joblib
import shap
import seaborn as sns
import optuna
from typing import Dict, Any


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=0,
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Return negative AUROC for minimization
    return -roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def run_xgboost_classification(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=50):
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns)
    all_shap_values = []
    all_probabilities = []
    all_best_params = []

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i, stratify=y
        )
        
        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state + i, stratify=y_train
        )

        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val),
            n_trials=n_trials
        )
        
        best_params = study.best_params
        all_best_params.append(best_params)
        
        # Train final model with best parameters
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=50,
            verbosity=0,
            random_state=random_state + i,
            **best_params
        )
        
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        all_shap_values.append(shap_values)
        
        # Save model
        joblib.dump({
            'model': final_model,
            'explainer': explainer,
            'best_params': best_params
        }, f'{output_prefix}_model_iter_{i+1}.joblib')

        # Get predictions and probabilities
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        all_probabilities.append(y_prob)

        predictions_list.append(pd.DataFrame({
            'iteration': i,
            'true': y_test,
            'predicted': y_pred,
            'probability': y_prob
        }))

        # Calculate metrics
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)

        metrics.append({
            'iteration': i,
            'auroc': auroc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'brier_score': brier,
            **best_params
        })

        feature_importances[f'iter_{i}'] = final_model.feature_importances_

    # Save predictions
    predictions_df = pd.concat(predictions_list)
    predictions_df.to_csv(f'{output_prefix}_predictions.csv', index=False)

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{output_prefix}_metrics.csv', index=False)

    # Save feature importances
    feature_importances.to_csv(f'{output_prefix}_feature_importance.csv')

    # Save best parameters
    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(f'{output_prefix}_best_parameters.csv', index=False)

    # Calculate and save mean SHAP values
    mean_shap_values = np.mean(all_shap_values, axis=0)
    mean_shap_importance = np.abs(mean_shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(f'{output_prefix}_shap_importance.csv', index=False)

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        mean_shap_values,
        X_test,
        feature_names=X.columns,
        show=False
    )
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_shap_summary.png')
    plt.close()

    # Plot SHAP dependence plots for top features
    top_features = shap_importance_df['feature'].head(5).tolist()
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            mean_shap_values,
            X_test,
            feature_names=X.columns,
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_shap_dependence_{feature}.png')
        plt.close()

    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_shap_features.png')
    plt.close()

    # Calculate summary statistics
    summary = {}
    for metric in ['auroc', 'auprc', 'precision', 'recall', 'f1_score', 'balanced_accuracy', 'brier_score']:
        values = metrics_df[metric]
        mean = values.mean()
        std = values.std()
        ci_lower, ci_upper = t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
        summary[metric] = {
            'mean': mean,
            'std': std,
            '95%_CI_lower': ci_lower,
            '95%_CI_upper': ci_upper
        }

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(f'{output_prefix}_metrics_summary.csv')

    # Plot AUROC across iterations with confidence intervals
    auroc_values = metrics_df['auroc'].values
    mean_auroc = np.mean(auroc_values)
    std_auroc = np.std(auroc_values)
    ci_upper = mean_auroc + 1.96 * std_auroc / np.sqrt(n_iter)
    ci_lower = mean_auroc - 1.96 * std_auroc / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], auroc_values, marker='o', label='AUROC per iteration')
    plt.axhline(mean_auroc, color='green', linestyle='--', label='Mean AUROC')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.title('AUROC Score over Iterations (XGBoost)')
    plt.xlabel('Iteration')
    plt.ylabel('AUROC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_auroc_plot.png')
    plt.close()

    # Plot AUPRC across iterations
    auprc_values = metrics_df['auprc'].values
    mean_auprc = np.mean(auprc_values)
    std_auprc = np.std(auprc_values)
    ci_upper = mean_auprc + 1.96 * std_auprc / np.sqrt(n_iter)
    ci_lower = mean_auprc - 1.96 * std_auprc / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], auprc_values, marker='s', label='AUPRC per iteration')
    plt.axhline(mean_auprc, color='blue', linestyle='--', label='Mean AUPRC')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')
    plt.title('AUPRC Score over Iterations (XGBoost)')
    plt.xlabel('Iteration')
    plt.ylabel('AUPRC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_auprc_plot.png')
    plt.close()

    # Plot feature importance comparison
    plt.figure(figsize=(12, 6))
    feature_importance_mean = feature_importances.mean(axis=1).sort_values(ascending=True)
    feature_importance_mean.tail(10).plot(kind='barh')
    plt.title('Top 10 Features by XGBoost Importance')
    plt.xlabel('Mean Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_feature_importance_plot.png')
    plt.close()

    # Plot confusion matrix for the best iteration
    best_iter = metrics_df['auroc'].idxmax()
    best_predictions = predictions_df[predictions_df['iteration'] == best_iter]
    cm = confusion_matrix(best_predictions['true'], best_predictions['predicted'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Best Iteration {best_iter})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confusion_matrix.png')
    plt.close()

    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    all_probs = np.concatenate(all_probabilities)
    plt.hist(all_probs, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_probability_distribution.png')
    plt.close()

    # Plot hyperparameter importance
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_hyperparameter_importance.png')
    plt.close()

    print("XGBoost classification completed. Results saved to:", output_prefix)

    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': feature_importances,
        'shap_importance': shap_importance_df,
        'summary': summary_df,
        'best_params': best_params_df
    }

results = run_xgboost_classification(X, y, n_iter=10, output_prefix='xgboost_tuned', n_trials=50)