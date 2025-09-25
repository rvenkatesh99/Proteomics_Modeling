import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import shap
import optuna
from typing import Dict, Any

class AdvancedRegressor(nn.Module):
    def __init__(self, input_dim: int, params: Dict[str, Any]):
        super(AdvancedRegressor, self).__init__()
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
        
        # Output layer
        layers.append(nn.Linear(final_input, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> float:
    params = {
        'units_1': trial.suggest_int('units_1', 32, 256),
        'units_2': trial.suggest_int('units_2', 16, 128),
        'units_3': trial.suggest_int('units_3', 8, 64),
        'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5),
        'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5),
        'dropout_3': trial.suggest_float('dropout_3', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'use_third_layer': trial.suggest_categorical('use_third_layer', [True, False])
    }
    
    model = AdvancedRegressor(X_train.shape[1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience = 10
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


def run_pytorch_regression(X, y, n_iter=10, output_prefix='torch_output', test_size=0.2, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_metrics = []
    all_preds = []
    all_feature_importances = pd.DataFrame(index=X.columns)
    all_shap_values = []
    
    for i in range(n_iter):
        print(f'Iteration {i + 1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=random_state + i
        )
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val, device),
            n_trials=20
        )
        
        best_params = study.best_params
        model = AdvancedRegressor(X.shape[1], best_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training with early stopping
        best_model = None
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(300):
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
            preds = model(X_test_tensor).cpu().numpy().flatten()

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        metrics = {
            'iteration': i,
            'r2': r2,
            'rmse': rmse,
            **best_params
        }
        all_metrics.append(metrics)

        df_preds = pd.DataFrame({
            'iteration': i,
            'sample_index': y_test.index,
            'true_value': y_test.values,
            'predicted_value': preds
        })
        all_preds.append(df_preds)

        # Save model
        model_path = f'{output_prefix}_model_iter_{i+1}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': best_params
        }, model_path)

        # Calculate SHAP values
        background = shap.kmeans(X_train_scaled, 100)
        explainer = shap.KernelExplainer(
            lambda x: model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy(),
            background
        )
        shap_values = explainer.shap_values(X_test_scaled)
#         shap_values = np.array(shap_values)  # Ensure it's an ndarray
        # Handle multi-output SHAP format
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Assuming single output regression

        all_shap_values.append(shap_values)  # Should now be (n_samples, n_features)

        # Permutation importance
        importances = []
        base_r2 = r2_score(y_test, preds)
        for col in range(X.shape[1]):
            X_test_permuted = X_test.copy()
            X_test_permuted.iloc[:, col] = np.random.permutation(X_test_permuted.iloc[:, col])
            X_test_scaled_perm = scaler.transform(X_test_permuted)
            X_test_tensor_perm = torch.tensor(X_test_scaled_perm, dtype=torch.float32).to(device)
            with torch.no_grad():
                perm_preds = model(X_test_tensor_perm).cpu().numpy().flatten()
            perm_r2 = r2_score(y_test, perm_preds)
            importances.append(base_r2 - perm_r2)

        all_feature_importances[f'Iter_{i+1}'] = importances

    # Save results
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_preds)
    all_feature_importances.index = X.columns
    all_feature_importances.to_csv(f'{output_prefix}_feature_importances.csv')
    metrics_df.to_csv(f'{output_prefix}_metrics.csv', index=False)
    predictions_df.to_csv(f'{output_prefix}_predictions.csv', index=False)

    feature_stats = pd.DataFrame({
        'mean': all_feature_importances.mean(axis=1),
        'std': all_feature_importances.std(axis=1)
    })
    feature_stats.to_csv(f'{output_prefix}_feature_importances_mean_sd.csv')

    # Calculate and save mean SHAP values
    mean_shap_values = np.mean(all_shap_values, axis=0)
    # all_shap_values: list of (n_samples, n_features)
    all_shap_values_arr = np.stack(all_shap_values, axis=0)  # shape: (n_iter, n_samples, n_features)
    # Mean absolute SHAP value per feature
    mean_shap_importance = np.abs(all_shap_values_arr).mean(axis=(0, 1))  # (n_features,)
    mean_shap_importance = np.ravel(mean_shap_importance)
    
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance.flatten()
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

    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_shap_features.png')
    plt.close()

    # Plot R² over iterations
    r2_values = metrics_df['r2'].values
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    ci_upper = mean_r2 + 1.96 * std_r2 / np.sqrt(n_iter)
    ci_lower = mean_r2 - 1.96 * std_r2 / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], r2_values, marker='o', label='R²')
    plt.axhline(mean_r2, color='green', linestyle='--', label='Mean R²')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, alpha=0.2, color='green', label='95% CI')
    plt.xlabel('Iteration')
    plt.ylabel('R²')
    plt.title('PyTorch R² with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_r2_plot.png')
    plt.close()

    print(f"PyTorch regression complete. Output saved with prefix: {output_prefix}")
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_stats': feature_stats,
        'shap_importance': shap_importance_df
    }
