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

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def run_pytorch_regression(X, y, n_iter=10, output_prefix='torch_output', test_size=0.2, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_metrics = []
    all_preds = []
    all_feature_importances = pd.DataFrame(index=X.columns)
    
    for i in range(n_iter):
        print(f'Iteration {i + 1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = SimpleRegressor(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(300):
            for xb, yb in train_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy().flatten()

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        metrics = {
            'iteration': i,
            'r2': r2,
            'rmse': rmse
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
        torch.save(model.state_dict(), model_path)

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
        'feature_stats': feature_stats
    }
