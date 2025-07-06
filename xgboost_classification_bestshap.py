# %load Proteomics_Modeling/xgboost_classification_bestshap.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve
)
from scipy.interpolate import interp1d
from scipy.stats import t
import xgboost as xgb
import joblib
import shap
import seaborn as sns
import optuna
from typing import Dict, Any


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optimized objective function with tuned hyperparameter ranges for faster convergence"""
    
    # Optimized hyperparameter ranges based on best parameters analysis
    params = {
        # Core parameters - optimized ranges
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.25, log=True),
        
        # Sampling parameters - optimized ranges
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
        
        # Regularization parameters - optimized for small values
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 0.1, log=True),
        
        # Class imbalance handling
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 6.0),
        
        'random_state': 42,
        'verbosity': 0
    }
    
    try:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=20,
            **params
        )
        
        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate AUROC
        auroc = roc_auc_score(y_val, y_pred_proba)
        
        return -auroc  # Negative because we minimize
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def run_xgboost_classification(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=50):
    """Simplified XGBoost classification with optimized hyperparameters and stable results"""
    
    # Create output directory
    os.makedirs(output_prefix, exist_ok=True)
    
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns)
    all_probabilities = []
    all_best_params = []
    all_models = []
    all_X_tests = []
    all_y_tests = []
    
    # Calculate class imbalance ratio
    class_ratio = y.value_counts(normalize=True)
    print(f"Class distribution: {class_ratio}")

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
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state + i)
        )
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val),
            n_trials=n_trials
        )
        best_params = study.best_params
        all_best_params.append(best_params)
        
        print(f"Best trial AUROC: {-study.best_value:.4f}")
        
        # Train final model with best parameters
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=20,
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
            'best_params': best_params,
            'feature_names': X.columns.tolist()
        }, f'{output_prefix}/model_iter_{i+1}.joblib')
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
        
        # Plot hyperparameter importance
        try:
            # Check if study has enough trials and valid parameters
            if len(study.trials) > 1:
                print(f"Attempting to plot hyperparameter importance for {len(study.trials)} trials...")
                
                # Get parameter importance safely
                param_importance = optuna.importance.get_param_importances(study)
                print(f"Raw parameter importance: {param_importance}")
                
                # Filter out any problematic parameters
                valid_params = {}
                for param, importance in param_importance.items():
                    if isinstance(importance, (int, float)) and not np.isnan(importance):
                        valid_params[param] = importance
                    else:
                        print(f"Skipping invalid parameter {param}: {importance} (type: {type(importance)})")
                
                if valid_params:
                    plt.figure(figsize=(12, 8))
                    # Create manual bar plot instead of using optuna's plotting function
                    params = list(valid_params.keys())
                    importances = list(valid_params.values())
                    
                    # Sort by importance
                    sorted_indices = np.argsort(importances)[::-1]
                    params = [params[i] for i in sorted_indices]
                    importances = [importances[i] for i in sorted_indices]
                    
                    plt.barh(range(len(params)), importances)
                    plt.yticks(range(len(params)), params)
                    plt.xlabel('Importance')
                    plt.title(f'Hyperparameter Importance - Iteration {i+1}')
                    plt.tight_layout()
                    plt.savefig(f'{output_prefix}/hyperparameter_importance_iter_{i+1}.png')
                    plt.close()
                    print(f"Successfully saved hyperparameter importance plot for iteration {i+1}")
                else:
                    print(f"No valid parameter importance values found in iteration {i+1}")
            else:
                print(f"Not enough trials ({len(study.trials)}) to plot hyperparameter importance in iteration {i+1}")
                
        except Exception as e:
            print(f"Could not plot hyperparameter importance in iteration {i+1}: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            # Try alternative plotting method
            try:
                if len(study.trials) > 1:
                    # Simple bar plot of best parameters
                    best_params = study.best_params
                    if best_params:
                        plt.figure(figsize=(10, 6))
                        param_names = list(best_params.keys())
                        param_values = list(best_params.values())
                        
                        # Convert values to strings for display
                        param_values_str = [str(v)[:10] + '...' if len(str(v)) > 10 else str(v) for v in param_values]
                        
                        plt.barh(range(len(param_names)), [1] * len(param_names))
                        plt.yticks(range(len(param_names)), param_names)
                        plt.xlabel('Best Value')
                        plt.title(f'Best Parameters - Iteration {i+1}')
                        
                        # Add value labels
                        for i_val, (name, val) in enumerate(zip(param_names, param_values_str)):
                            plt.text(0.5, i_val, val, va='center', ha='center')
                        
                        plt.tight_layout()
                        plt.savefig(f'{output_prefix}/best_parameters_iter_{i+1}.png')
                        plt.close()
                        print(f"Saved best parameters plot for iteration {i+1}")
            except Exception as e2:
                print(f"Alternative plotting also failed in iteration {i+1}: {e2}")

        
    # Save predictions
    predictions_df = pd.concat(predictions_list)
    predictions_df.to_csv(f'{output_prefix}/predictions.csv', index=False)
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{output_prefix}/metrics.csv', index=False)
    # Save feature importances
    feature_importances.to_csv(f'{output_prefix}/feature_importance.csv')
    # Save best parameters
    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(f'{output_prefix}/best_parameters.csv', index=False)
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
    summary_df.to_csv(f'{output_prefix}/metrics_summary.csv')
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
    plt.savefig(f'{output_prefix}/auroc_plot.png')
    plt.close()
    # === Optimized ROC Curve with Confidence Intervals ===
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = [m['auroc'] for m in metrics]  # Use previously computed AUROC values

    for y_test, y_prob in zip(all_y_tests, all_probabilities):
        y_test = np.asarray(y_test).ravel()
        y_prob = np.asarray(y_prob).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Vectorized interpolation
        interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
        interp_tpr = interp_func(mean_fpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2,
                     label='± 1 std. dev.')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve with Confidence Interval (XGBoost)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/roc_curves_summary.png')
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
    plt.savefig(f'{output_prefix}/auprc_plot.png')
    plt.close()
    # Plot feature importance comparison
    plt.figure(figsize=(12, 6))
    feature_importance_mean = feature_importances.mean(axis=1).sort_values(ascending=True)
    feature_importance_mean.tail(10).plot(kind='barh')
    plt.title('Top 10 Features by XGBoost Importance')
    plt.xlabel('Mean Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/feature_importance_plot.png')
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
    plt.savefig(f'{output_prefix}/confusion_matrix.png')
    plt.close()
    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    all_probs = np.concatenate(all_probabilities)
    plt.hist(all_probs, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/probability_distribution.png')
    plt.close()
    # --- SHAP analysis only for the best model ---
    best_model_idx = metrics_df['auroc'].idxmax()
    best_model = all_models[best_model_idx]
    best_X_test = all_X_tests[best_model_idx]
    best_y_test = all_y_tests[best_model_idx]
    
    print(f"Performing SHAP analysis on best model (iteration {best_model_idx})...")
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(best_X_test)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(f'{output_prefix}/shap_importance.csv', index=False)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        best_X_test,
        feature_names=X.columns,
        show=False,
        max_display=20
    )
    plt.title('SHAP Feature Importance Summary (Best Model)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/shap_summary.png')
    plt.close()
    top_features = shap_importance_df['feature'].head(2).tolist()
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            best_X_test,
            feature_names=X.columns,
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {feature} (Best Model)')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}/shap_dependence_{feature}.png')
        plt.close()
    
    print(f"✅ Improved XGBoost classification completed!")
    print(f"Best model AUROC: {auroc_values[best_model_idx]:.4f}")
    print(f"Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print("Results saved to:", output_prefix)
    
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': feature_importances,
        'shap_importance': shap_importance_df,
        'summary': summary_df,
        'best_params': best_params_df
    }

# Example usage:
# results = run_xgboost_classification(X, y, n_iter=3, output_prefix='xgboost_tuned', n_trials=10) 