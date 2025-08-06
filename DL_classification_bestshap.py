import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve
)
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import shap
import optuna
from typing import Dict, Any
from scipy.stats import t
import seaborn as sns

class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim: int, params: Dict[str, Any]):
        super(AdvancedClassifier, self).__init__()
        layers = []
        
        # First hidden layer
        layers.extend([
            nn.Linear(input_dim, params['units_1']),
            nn.BatchNorm1d(params['units_1']),
            nn.ReLU(),
            nn.Dropout(params['dropout_1'])
        ])
        
        # Second hidden layer
        layers.extend([
            nn.Linear(params['units_1'], params['units_2']),
            nn.BatchNorm1d(params['units_2']),
            nn.ReLU(),
            nn.Dropout(params['dropout_2'])
        ])
        
        # Optional third layer
        if params['use_third_layer']:
            layers.extend([
                nn.Linear(params['units_2'], params['units_3']),
                nn.BatchNorm1d(params['units_3']),
                nn.ReLU(),
                nn.Dropout(params['dropout_3'])
            ])
            final_input = params['units_3']
        else:
            final_input = params['units_2']
        
        # Output layer with sigmoid for binary classification
        layers.append(nn.Linear(final_input, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> float:
    params = {
        'units_1': trial.suggest_int('units_1', 32, 128),
        'units_2': trial.suggest_int('units_2', 16, 64),  
        'units_3': trial.suggest_int('units_3', 8, 32),   
        'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.4),  
        'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.4), 
        'dropout_3': trial.suggest_float('dropout_3', 0.1, 0.4), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True), 
        'use_third_layer': trial.suggest_categorical('use_third_layer', [True, False])
    }
    
    model = AdvancedClassifier(X_train.shape[1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # larger batch size
    
    best_val_loss = float('inf')
    patience = 5  # reduced patience
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
    
    return best_val_loss.item()

def run_pytorch_classification(X, y, n_iter=10, output_prefix='', test_size=0.2, random_state=42, n_trials=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_prefix, exist_ok=True)
    model_dir = os.path.join(output_prefix, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    all_metrics = []
    all_preds = []
    all_probabilities = []
    all_models = []
    all_X_tests = []
    all_y_tests = []
    all_X_test_scaled = []
    
    for i in range(n_iter):
        print(f'Iteration {i + 1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i, stratify=y
        )

        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state + i, stratify=y_train
        )

        # Scale
        scaler = StandardScaler()
        X_train_final_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val, device),
            n_trials=n_trials
        )
        
        best_params = study.best_params
        model = AdvancedClassifier(X.shape[1], best_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.BCELoss()

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)  # larger batch size

        # Training with early stopping
        best_model = None
        best_val_loss = float('inf')
        patience = 5  # reduced patience
        patience_counter = 0
        
        for epoch in range(100):  # reduced max epochs
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_train_tensor)
                val_loss = criterion(val_pred, y_train_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
        
        model = best_model
        model.eval()
        with torch.no_grad():
            probabilities = model(X_test_tensor).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)
            all_probabilities.append(probabilities)

        # Save model
        all_models.append(model)
        all_X_tests.append(X_test)
        all_y_tests.append(y_test)
        all_X_test_scaled.append(X_test_scaled)
        
        # Save parameters
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': best_params
        }, os.path.join(model_dir, f'dl_iter_{i}.pt'))

        # Calculate metrics
        auroc = roc_auc_score(y_test, probabilities)
        auprc = average_precision_score(y_test, probabilities)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        brier = brier_score_loss(y_test, probabilities)
        
        all_metrics.append({
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
        
        all_preds.append(pd.DataFrame({
            'iteration': i,
            'true': y_test.values,
            'predicted': predictions,
            'probability': probabilities
        }))
    
    # Save results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_prefix, 'metrics.csv'), index=False)
    
    predictions_df = pd.concat(all_preds, ignore_index=True)
    predictions_df.to_csv(os.path.join(output_prefix, 'predictions.csv'), index=False)
    
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
    summary_df.to_csv(os.path.join(output_prefix, 'metrics_summary.csv'), index=False)
    
    # Plot AUROC across iterations
    auroc_values = metrics_df['auroc'].values
    mean_auroc = np.mean(auroc_values)
    std_auroc = np.std(auroc_values)
    ci_upper = mean_auroc + 1.96 * std_auroc / np.sqrt(n_iter)
    ci_lower = mean_auroc - 1.96 * std_auroc / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], auroc_values, marker='o', label='AUROC per iteration')
    plt.axhline(mean_auroc, color='green', linestyle='--', label='Mean AUROC')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.title('AUROC Score over Iterations (Deep Learning)')
    plt.xlabel('Iteration')
    plt.ylabel('AUROC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_prefix, 'auroc_plot.png'))
    plt.close()

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = [m['auroc'] for m in all_metrics]

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
    plt.savefig(f'{output_prefix}_roc_curves_summary.png')
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
    plt.title('AUPRC Score over Iterations (Deep Learning)')
    plt.xlabel('Iteration')
    plt.ylabel('AUPRC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_prefix, 'auprc_plot.png'))
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
    plt.savefig(os.path.join(output_prefix, 'confusion_matrix.png'))
    plt.close()
    
    # SHAP analysis only for the best model - Kernel Explainer (slow)
    best_model_idx = metrics_df['auroc'].idxmax()
    best_model = all_models[best_model_idx]
    best_X_test_scaled = all_X_test_scaled[best_model_idx]
    
    background = shap.kmeans(best_X_test_scaled, 50)
    explainer = shap.KernelExplainer(
        lambda x: best_model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten(),
        background
    )
    shap_values = explainer.shap_values(best_X_test_scaled)
    
    # Save mean SHAP values
    mean_shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(os.path.join(output_prefix, 'shap_importance.csv'), index=False)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        all_X_tests[best_model_idx],
        feature_names=X.columns,
        show=False,
        max_display=20 
    )
    plt.title('SHAP Feature Importance Summary (Best Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, 'shap_summary.png'))
    plt.close()
    
    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance (Best Model)')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, 'top_shap_features.png'))
    plt.close()
    
    print("Deep Learning classification completed. Results saved to:", output_prefix)
    print(f"Best model (iteration {best_iter}) AUROC: {auroc_values[best_iter]:.4f}")
    
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'summary': summary_df,
        'shap_importance': shap_importance_df,
        'best_iteration': best_iter
    }