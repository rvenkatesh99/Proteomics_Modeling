import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    average_precision_score, balanced_accuracy_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
import joblib
import shap
import seaborn as sns
import optuna
from typing import Dict, Any
import traceback
from sklearn.model_selection import StratifiedShuffleSplit


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Objective function for multinomial logistic regression optimization: maximize macro-averaged ROC AUC"""
    # Only allow solvers that support multi_class='multinomial'
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    if penalty in ['l1', 'elasticnet']:
        solver = 'saga'
    else:  # l2
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'newton-cg', 'sag'])
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': penalty,
        'solver': solver,
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
        'random_state': 42
    }
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
    try:
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32).ravel()
        y_val = np.asarray(y_val, dtype=np.int32).ravel()
        if X_train.size == 0 or X_val.size == 0:
            return 0.0
        if len(np.unique(y_train)) < 2:
            return 0.0
        model = LogisticRegression(
            multi_class='multinomial',
            class_weight='balanced',
            **params
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro')
        return auc
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def run_multinomial_regression_multiclass(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=50, feature_selection=True):
    """Run multinomial logistic regression multiclass classification with optional SHAP-based feature selection and class balancing."""
    os.makedirs(output_prefix, exist_ok=True)
    print("Validating input data...")
    
    # Ensure X is a DataFrame with unique columns
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with real feature names for SHAP to work.")
    if X.columns.duplicated().any():
        print("Duplicate columns found and removed:", list(X.columns[X.columns.duplicated()]))
        X = X.loc[:, ~X.columns.duplicated()]
    feature_names = X.columns
    y = np.asarray(y).astype(int).ravel()
    if np.any(np.isnan(X.values)) or np.any(np.isinf(X.values)):
        print("Warning: Found NaN or infinite values in features. Replacing with 0.")
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    if n_classes < 2:
        raise ValueError("Need at least 2 classes for classification")

    if feature_selection:
        # --- SHAP-based Feature Selection ---
        print("Performing SHAP-based feature selection...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(sss.split(X, y))
        X_train_fs, y_train_fs = X.iloc[train_idx], y[train_idx]
        
        # Scale features for logistic regression
        scaler_fs = StandardScaler()
        X_train_fs_scaled = scaler_fs.fit_transform(X_train_fs)
        
        model_fs = LogisticRegression(multi_class='multinomial', class_weight='balanced', random_state=random_state)
        model_fs.fit(X_train_fs_scaled, y_train_fs)
        
        # Use LinearExplainer for logistic regression
        explainer_fs = shap.LinearExplainer(model_fs, X_train_fs_scaled)
        shap_values_fs = explainer_fs.shap_values(X_train_fs_scaled)
        
        # For multiclass, sum mean(|shap|) across classes
        if isinstance(shap_values_fs, list):
            mean_abs_shap = np.sum([np.abs(sv).mean(axis=0) for sv in shap_values_fs], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values_fs).mean(axis=0)
        
        top_n = 30
        top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
        top_features = X_train_fs.columns[top_idx]
        print(f"Selected top {top_n} features: {list(top_features)}")
        
        # Use only top features for the rest of the pipeline
        X = X[top_features]
        feature_names = X.columns
    else:
        print("Feature selection disabled: using all features.")

    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=feature_names)
    all_probabilities = []
    all_best_params = []
    all_models = []
    all_X_tests = []
    all_y_tests = []
    all_class_reports = []
    all_roc_aucs = []
    all_shap_data = []
    all_scalers = []

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        try:
            # Always split using DataFrames to preserve column names
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + i, stratify=y
            )
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            # Further split training data for validation
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state + i, stratify=y_train
            )
            X_train_final = X_train_final.copy()
            X_val = X_val.copy()
            
            # Convert all columns to numeric
            X_train_final = X_train_final.apply(pd.to_numeric, errors='raise')
            X_val = X_val.apply(pd.to_numeric, errors='raise')
            X_test = X_test.apply(pd.to_numeric, errors='raise')
            
            # Scale features
            scaler = StandardScaler()
            X_train_final_scaled = scaler.fit_transform(X_train_final)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, X_train_final_scaled, y_train_final, X_val_scaled, y_val),
                n_trials=n_trials
            )
            best_params = study.best_params
            all_best_params.append(best_params)
            
            final_model = LogisticRegression(
                multi_class='multinomial',
                class_weight='balanced',
                random_state=random_state + i,
                **best_params
            )
            final_model.fit(X_train_final_scaled, y_train_final)
            
            all_models.append(final_model)
            all_X_tests.append(X_test)
            all_y_tests.append(y_test)
            all_scalers.append(scaler)
            
            joblib.dump({
                'model': final_model,
                'scaler': scaler,
                'best_params': best_params,
                'feature_names': feature_names.tolist()
            }, f'{output_prefix}/model_iter_{i+1}.joblib')
            
            y_pred = final_model.predict(X_test_scaled)
            y_prob = final_model.predict_proba(X_test_scaled)
            all_probabilities.append(y_prob)
            predictions_list.append(pd.DataFrame({
                'iteration': i,
                'true': y_test,
                'predicted': y_pred
            }))
            
            # --- Per-class metrics ---
            per_class_metrics = []
            for c in unique_classes:
                y_true_bin = (y_test == c).astype(int)
                y_prob_c = y_prob[:, c]
                y_pred_bin = (y_pred == c).astype(int)
                try:
                    auroc = roc_auc_score(y_true_bin, y_prob_c)
                except Exception:
                    auroc = np.nan
                try:
                    auprc = average_precision_score(y_true_bin, y_prob_c)
                except Exception:
                    auprc = np.nan
                precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
                try:
                    brier = brier_score_loss(y_true_bin, y_prob_c)
                except Exception:
                    brier = np.nan
                per_class_metrics.append({
                    'iteration': i,
                    'class': c,
                    'auroc': auroc,
                    'auprc': auprc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'balanced_accuracy': balanced_acc,
                    'brier_score': brier
                })
            if i == 0:
                all_per_class_metrics = []
            all_per_class_metrics.extend(per_class_metrics)
            
            # --- Overall metrics (macro/weighted) ---
            accuracy = accuracy_score(y_test, y_pred)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            balanced_acc_macro = balanced_accuracy_score(y_test, y_pred)
            try:
                roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            except Exception:
                roc_auc_macro = np.nan
            try:
                auprc_macro = average_precision_score(y_test, y_prob, average='macro')
            except Exception:
                auprc_macro = np.nan
            try:
                brier_macro = np.mean([
                    brier_score_loss((y_test == c).astype(int), y_prob[:, c]) for c in unique_classes
                ])
            except Exception:
                brier_macro = np.nan
            
            metrics.append({
                'iteration': i,
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'roc_auc_macro': roc_auc_macro,
                'auprc_macro': auprc_macro,
                'balanced_accuracy_macro': balanced_acc_macro,
                'brier_score_macro': brier_macro,
                **best_params
            })
            
            # Store feature importances (coefficients)
            feature_importances[f'iter_{i}'] = np.abs(final_model.coef_).mean(axis=0)
            print(f"Iteration {i+1} - Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
            
            # --- SHAP calculation for multinomial regression ---
            try:
                X_test_df = X_test.copy()
                print("X_test_df shape before SHAP explainer:", X_test_df.shape)
                print("Model n_features_in_:", getattr(final_model, 'n_features_in_', 'NA'))
                print("X_test_df columns:", X_test_df.columns)
                
                # Use LinearExplainer for logistic regression
                explainer = shap.LinearExplainer(final_model, X_test_scaled)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # Store SHAP values and data for best iteration plotting later
                all_shap_data.append({
                    'iteration': i,
                    'shap_values': shap_values,
                    'X_test_df': X_test_df,
                    'accuracy': accuracy
                })
                
            except Exception as e:
                print(f"SHAP error: {e}")
                traceback.print_exc()
                print('X_test_df shape:', X_test_df.shape)
                print('X_test_df dtypes:', X_test_df.dtypes)
                print('Model n_features_in_:', getattr(final_model, 'n_features_in_', 'NA'))
                print('X_test_df columns:', X_test_df.columns)
                continue
            
            # Per-class confusion matrix
            for c in unique_classes:
                cm = confusion_matrix((y_test == c).astype(int), (y_pred == c).astype(int))
                plt.figure(figsize=(4, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix for class {c} (iter {i+1})')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(f'{output_prefix}/confusion_matrix_class_{c}_iter_{i+1}.png')
                plt.close()
                
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            continue
    
    if not metrics:
        raise ValueError("No successful iterations completed")
    
    predictions_df = pd.concat(predictions_list)
    predictions_df.to_csv(f'{output_prefix}/predictions.csv', index=False)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{output_prefix}/metrics.csv', index=False)
    feature_importances.to_csv(f'{output_prefix}/feature_importance.csv')
    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(f'{output_prefix}/best_parameters.csv', index=False)
    all_class_reports_df = pd.concat(all_class_reports) if all_class_reports else pd.DataFrame()
    if not all_class_reports_df.empty:
        all_class_reports_df.to_csv(f'{output_prefix}/per_class_metrics.csv')
    pd.DataFrame(all_roc_aucs).to_csv(f'{output_prefix}/per_class_roc_auc.csv', index=False)
    
    # Save per-class metrics
    per_class_metrics_df = pd.DataFrame(all_per_class_metrics)
    per_class_metrics_df.to_csv(f'{output_prefix}/per_class_metrics_full.csv', index=False)
    
    # Calculate summary statistics for all overall metrics
    summary_metrics = [
        'accuracy',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted',
        'roc_auc_macro', 'auprc_macro', 'balanced_accuracy_macro', 'brier_score_macro'
    ]
    summary = {}
    for metric in summary_metrics:
        if metric in metrics_df.columns:
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
        else:
            summary[metric] = {
                'mean': None,
                'std': None,
                '95%_CI_lower': None,
                '95%_CI_upper': None
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
    plt.title('Accuracy Score over Iterations (Multinomial Regression)')
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
    plt.title('F1 Score over Iterations (Multinomial Regression)')
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
    plt.title('Top 10 Features by Multinomial Regression Importance')
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

    # --- SHAP plotting for best iteration only ---
    if all_shap_data:
        # Find best iteration by accuracy
        best_shap_data = max(all_shap_data, key=lambda x: x['accuracy'])
        best_iter = best_shap_data['iteration']
        shap_values = best_shap_data['shap_values']
        X_test_df = best_shap_data['X_test_df']
        
        print(f"Creating SHAP plots for best iteration {best_iter + 1} (accuracy: {best_shap_data['accuracy']:.4f})")
        
        if isinstance(shap_values, list):
            # Multiclass: arrange with 2 plots per row for readability
            n_classes = len(shap_values)
            ncols = 2
            nrows = (n_classes + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(18*ncols, 10*nrows), sharey=False)
            axes = axes.flatten() if n_classes > 1 else [axes]
            for class_idx, class_shap in enumerate(shap_values):
                class_name = f'class_{class_idx}'
                mean_shap = np.abs(class_shap).mean(axis=0)
                # Get top 15 features for this class
                top_idx = np.argsort(mean_shap)[-15:][::-1]
                top_features = X_test_df.columns[top_idx]
                class_shap_top = class_shap[:, top_idx]
                X_test_df_top = X_test_df.iloc[:, top_idx]
                shap_df = pd.DataFrame({
                    'feature': top_features,
                    'mean_shap_value': mean_shap[top_idx]
                }).sort_values('mean_shap_value', ascending=False)
                shap_df.to_csv(f'{output_prefix}/shap_importance_{class_name}_best_iter.csv', index=False)
                plt.sca(axes[class_idx])
                shap.summary_plot(
                    class_shap_top, X_test_df_top,
                    show=False, max_display=15
                )
                axes[class_idx].set_title(f'SHAP Summary - {class_name} (Best Iteration {best_iter + 1})', fontsize=16)
                axes[class_idx].tick_params(axis='both', which='major', labelsize=12)
            # Hide any unused axes
            for j in range(class_idx + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_prefix}/shap_panel_best_iter.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Binary classification
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': X_test_df.columns,
                'mean_shap_value': mean_shap
            }).sort_values('mean_shap_value', ascending=False)
            shap_df.to_csv(f'{output_prefix}/shap_importance_best_iter.csv', index=False)
            
            plt.figure(figsize=(15, 10))
            shap.summary_plot(
                shap_values, X_test_df,
                show=False, max_display=15
            )
            plt.title(f'SHAP Summary (Best Iteration {best_iter + 1})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{output_prefix}/shap_summary_best_iter.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Plot summary ROC curves with confidence intervals
    mean_fpr = np.linspace(0, 1, 100)
    class_tprs = {c: [] for c in unique_classes}
    class_aucs = {c: [] for c in unique_classes}

    for i in range(n_iter):
        y_test = all_y_tests[i]
        y_prob = all_probabilities[i]
        for c in unique_classes:
            # One-vs-rest binary labels
            y_true_bin = (y_test == c).astype(int)
            y_score = y_prob[:, c]
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            class_tprs[c].append(interp_tpr)
            try:
                class_aucs[c].append(roc_auc_score(y_true_bin, y_score))
            except:
                class_aucs[c].append(np.nan)

    plt.figure(figsize=(10, 8))
    for c in unique_classes:
        tprs = np.array(class_tprs[c])
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        mean_auc = np.nanmean(class_aucs[c])
        std_auc = np.nanstd(class_aucs[c])
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.plot(mean_fpr, mean_tpr, label=f'Class {c} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Summary ROC Curves with Confidence Intervals (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/roc_curves_summary_multiclass.png')
    plt.close()

    print(f"✅ Multinomial regression multiclass classification completed!")
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
# results = run_multinomial_regression_multiclass(X, y, n_iter=10, output_prefix='multinomial_regression_multiclass', n_trials=50) 