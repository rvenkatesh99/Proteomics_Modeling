import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
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
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 0.5, log=True),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbosity=0,
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Return negative accuracy for minimization
    return -accuracy_score(y_val, model.predict(X_val))


def run_xgboost_multiclass(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=10):
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns)
    all_probabilities = []
    all_best_params = []
    all_models = []
    all_X_tests = []
    all_y_tests = []
    
    # Get unique classes for multiclass metrics
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

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
            objective='multi:softprob',
            eval_metric='mlogloss',
            early_stopping_rounds=10,
            verbosity=0,
            random_state=random_state + i,
            n_jobs=-1,
            **best_params
        )
        
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Save model for possible SHAP analysis later
        all_models.append(final_model)
        all_X_tests.append(X_test)
        all_y_tests.append(y_test)
        
        # Save model
        joblib.dump({
            'model': final_model,
            'best_params': best_params
        }, f'{output_prefix}_model_iter_{i+1}.joblib')

        # Get predictions and probabilities
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)
        all_probabilities.append(y_prob)

        predictions_list.append(pd.DataFrame({
            'iteration': i,
            'true': y_test,
            'predicted': y_pred
        }))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate ROC AUC for multiclass (one-vs-rest)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

        metrics.append({
            'iteration': i,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
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

    # Calculate summary statistics
    summary = {}
    for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                   'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']:
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

    # Plot Accuracy across iterations with confidence intervals
    accuracy_values = metrics_df['accuracy'].values
    mean_accuracy = np.mean(accuracy_values)
    std_accuracy = np.std(accuracy_values)
    ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(n_iter)
    ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], accuracy_values, marker='o', label='Accuracy per iteration')
    plt.axhline(mean_accuracy, color='green', linestyle='--', label='Mean Accuracy')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.title('Accuracy Score over Iterations (XGBoost Multiclass)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_accuracy_plot.png')
    plt.close()

    # Plot F1 Score across iterations
    f1_values = metrics_df['f1_macro'].values
    mean_f1 = np.mean(f1_values)
    std_f1 = np.std(f1_values)
    ci_upper = mean_f1 + 1.96 * std_f1 / np.sqrt(n_iter)
    ci_lower = mean_f1 - 1.96 * std_f1 / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], f1_values, marker='s', label='F1 Score per iteration')
    plt.axhline(mean_f1, color='blue', linestyle='--', label='Mean F1 Score')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')
    plt.title('F1 Score over Iterations (XGBoost Multiclass)')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_f1_plot.png')
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
    best_iter = metrics_df['accuracy'].idxmax()
    best_predictions = predictions_df[predictions_df['iteration'] == best_iter]
    cm = confusion_matrix(best_predictions['true'], best_predictions['predicted'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Confusion Matrix (Best Iteration {best_iter})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confusion_matrix.png')
    plt.close()

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts()
    plt.bar(class_counts.index, class_counts.values)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_class_distribution.png')
    plt.close()

    # Plot hyperparameter importance
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_hyperparameter_importance.png')
    plt.close()

    # --- SHAP analysis only for the best model ---
    best_model_idx = metrics_df['accuracy'].idxmax()
    best_model = all_models[best_model_idx]
    best_X_test = all_X_tests[best_model_idx]
    best_y_test = all_y_tests[best_model_idx]
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(best_X_test)
    
    # For multiclass, calculate mean absolute SHAP values across all classes
    if isinstance(shap_values, list):
        # Multiclass case: shap_values is a list of arrays for each class
        mean_shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Binary case: shap_values is a single array
        mean_shap_importance = np.abs(shap_values).mean(axis=0)
    
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(f'{output_prefix}_shap_importance.csv', index=False)

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        # For multiclass, plot SHAP for the first class as representative
        shap.summary_plot(
            shap_values[0],
            best_X_test,
            feature_names=X.columns,
            show=False,
            max_display=5
        )
    else:
        shap.summary_plot(
            shap_values,
            best_X_test,
            feature_names=X.columns,
            show=False,
            max_display=5
        )
    plt.title('SHAP Feature Importance Summary (Best Model)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_shap_summary.png')
    plt.close()

    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_shap_features.png')
    plt.close()

    print("XGBoost multiclass classification completed. Results saved to:", output_prefix)
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {unique_classes}")

    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': feature_importances,
        'shap_importance': shap_importance_df,
        'summary': summary_df,
        'best_params': best_params_df,
        'n_classes': n_classes,
        'unique_classes': unique_classes
    }

# Example usage:
# results = run_xgboost_multiclass(X, y, n_iter=10, output_prefix='xgboost_multiclass', n_trials=10) 