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
from sklearn.preprocessing import LabelEncoder


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Objective function for multiclass XGBoost optimization with error handling"""
    
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
    
    try:
        # Ensure data is in correct format
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32).ravel()
        y_val = np.asarray(y_val, dtype=np.int32).ravel()
        
        # Check for valid data
        if X_train.size == 0 or X_val.size == 0:
            return 0.0
        
        if len(np.unique(y_train)) < 2:
            return 0.0
        
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
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def run_xgboost_multiclass(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=10):
    """Run XGBoost multiclass classification with proper data handling"""
    
    # Create output directory
    os.makedirs(output_prefix, exist_ok=True)
    
    # Validate and prepare data
    print("Validating input data...")
    
    # Convert to numpy arrays and ensure correct types
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).ravel()
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Found NaN or infinite values in features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels if they're not integers
    if not np.issubdtype(y.dtype, np.integer):
        print("Encoding labels to integers...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Get unique classes for multiclass metrics
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {np.bincount(y)}")
    
    if n_classes < 2:
        raise ValueError("Need at least 2 classes for classification")
    
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns if hasattr(X, 'columns') else range(X.shape[1]))
    all_probabilities = []
    all_best_params = []
    all_models = []
    all_X_tests = []
    all_y_tests = []

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        
        try:
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
                'best_params': best_params,
                'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
            }, f'{output_prefix}/model_iter_{i+1}.joblib')

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
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
                roc_auc = 0.0

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
            
            print(f"Iteration {i+1} - Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            continue

    if not metrics:
        raise ValueError("No successful iterations completed")
    
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
    summary_df.to_csv(f'{output_prefix}/metrics_summary.csv')

    # Plot Accuracy across iterations with confidence intervals
    accuracy_values = metrics_df['accuracy'].values
    mean_accuracy = np.mean(accuracy_values)
    std_accuracy = np.std(accuracy_values)
    ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(len(accuracy_values))
    ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(len(accuracy_values))

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], accuracy_values, marker='o', label='Accuracy per iteration')
    plt.axhline(mean_accuracy, color='green', linestyle='--', label='Mean Accuracy')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.title('Accuracy Score over Iterations (XGBoost Multiclass)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}/accuracy_plot.png')
    plt.close()

    # Plot F1 Score across iterations
    f1_values = metrics_df['f1_macro'].values
    mean_f1 = np.mean(f1_values)
    std_f1 = np.std(f1_values)
    ci_upper = mean_f1 + 1.96 * std_f1 / np.sqrt(len(f1_values))
    ci_lower = mean_f1 - 1.96 * std_f1 / np.sqrt(len(f1_values))

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], f1_values, marker='s', label='F1 Score per iteration')
    plt.axhline(mean_f1, color='blue', linestyle='--', label='Mean F1 Score')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')
    plt.title('F1 Score over Iterations (XGBoost Multiclass)')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}/f1_plot.png')
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
    plt.savefig(f'{output_prefix}/confusion_matrix.png')
    plt.close()

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(y)
    plt.bar(range(len(class_counts)), class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(class_counts)), unique_classes)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/class_distribution.png')
    plt.close()

    print(f"✅ XGBoost multiclass classification completed!")
    print(f"Best model accuracy: {accuracy_values[best_iter]:.4f}")
    print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print("Results saved to:", output_prefix)

    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': feature_importances,
        'summary': summary_df,
        'best_params': best_params_df
    }

# Example usage:
# results = run_xgboost_multiclass(X, y, n_iter=10, output_prefix='xgboost_multiclass', n_trials=10) 