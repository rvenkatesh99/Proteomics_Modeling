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
from scipy.stats import t
import xgboost as xgb
import joblib
import shap
import seaborn as sns
import optuna
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
import traceback
from sklearn.model_selection import StratifiedShuffleSplit


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'random_state': 42
    }
    try:
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32).ravel()
        y_val = np.asarray(y_val, dtype=np.int32).ravel()
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
        # Maximize macro-averaged AUROC
        y_prob = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro')
        return auc
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def run_xgboost_multiclass(X, y, n_iter, output_prefix, test_size=0.2, random_state=42, n_trials=50):
    """Run XGBoost multiclass classification with SHAP-based feature selection and class balancing."""
    os.makedirs(output_prefix, exist_ok=True)
    # Ensure X has unique columns
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
    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state + i, stratify=y
            )
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state + i, stratify=y_train
            )
            X_train_final = X_train_final.copy()
            X_val = X_val.copy()
            # Convert all columns to numeric
            X_train_final = X_train_final.apply(pd.to_numeric, errors='raise')
            X_val = X_val.apply(pd.to_numeric, errors='raise')
            X_test = X_test.apply(pd.to_numeric, errors='raise')
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val),
                n_trials=n_trials
            )
            best_params = study.best_params
            all_best_params.append(best_params)
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
            all_models.append(final_model)
            all_X_tests.append(X_test)
            all_y_tests.append(y_test)
            joblib.dump({
                'model': final_model,
                'best_params': best_params,
                'feature_names': feature_names.tolist()
            }, f'{output_prefix}/model_iter_{i+1}.joblib')
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)
            all_probabilities.append(y_prob)
            predictions_list.append(pd.DataFrame({
                'iteration': i,
                'true': y_test,
                'predicted': y_pred
            }))
            # Per-class metrics
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            class_report_df = pd.DataFrame(class_report).T
            class_report_df['iteration'] = i
            all_class_reports.append(class_report_df)
            # Per-class ROC AUC (one-vs-rest)
            roc_aucs = {}
            for c in unique_classes:
                try:
                    roc_aucs[c] = roc_auc_score((y_test == c).astype(int), y_prob[:, c])
                except Exception as e:
                    roc_aucs[c] = np.nan
            all_roc_aucs.append({'iteration': i, **roc_aucs})
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
            feature_importances[f'iter_{i}'] = final_model.feature_importances_
            print(f"Iteration {i+1} - Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
            # SHAP calculation using DMatrix
            try:
                X_test_df = X_test.copy()
                print("X_test_df shape before SHAP explainer:", X_test_df.shape)
                print("Model n_features_in_:", getattr(final_model, 'n_features_in_', 'NA'))
                print("X_test_df columns:", X_test_df.columns)
                dtest = xgb.DMatrix(X_test_df, feature_names=final_model.get_booster().feature_names)
                explainer = shap.TreeExplainer(final_model.get_booster())
                shap_values = explainer.shap_values(dtest)
                
                # Store SHAP values and data for best iteration
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
                print('Model feature names:', getattr(final_model.get_booster(), 'feature_names', 'NA'))
                continue
            # --- Per-class confusion matrix ---
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
    all_class_reports_df = pd.concat(all_class_reports)
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

    # SHAP plotting - multiclass
    if all_shap_data:
        # Find best iteration by accuracy
        best_shap_data = max(all_shap_data, key=lambda x: x['accuracy'])
        best_iter = best_shap_data['iteration']
        shap_values = best_shap_data['shap_values']
        X_test_df = best_shap_data['X_test_df']
        
        print(f"Creating SHAP plots for best iteration {best_iter + 1} (accuracy: {best_shap_data['accuracy']:.4f})")
        
        if isinstance(shap_values, list):
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
                feature_names=X_test_df.columns,
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
                auc = roc_auc_score(y_true_bin, y_score)
            except Exception:
                auc = np.nan
            class_aucs[c].append(auc)

    plt.figure(figsize=(8, 6))
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