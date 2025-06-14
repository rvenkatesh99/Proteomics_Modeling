import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import t
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def run_autoencoder(X, y, n_iter=10, test_size=0.2, output_dir="autoencoder_results", random_state=42, batch_size=32, epochs=300):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    metrics = []
    reconstructions_list = []
    anomaly_scores_all = []
    embeddings_all = []

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )
            self.regressor = nn.Linear(32, 1)

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            pred = self.regressor(encoded)
            return decoded, encoded, pred.squeeze()

    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

        model = Autoencoder(X.shape[1])
        reconstruction_criterion = nn.MSELoss()
        prediction_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                outputs, _, preds = model(inputs)

                loss_recon = reconstruction_criterion(outputs, inputs)
                loss_pred = prediction_criterion(preds, targets)
                loss = loss_recon + loss_pred

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            reconstructions, embeddings, predictions = model(X_test_tensor)
            mse_recon = mean_squared_error(X_test_tensor.numpy(), reconstructions.numpy())
            mse_pred = mean_squared_error(y_test_tensor.numpy(), predictions.numpy())
            r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
            anomaly_scores = np.mean((X_test_tensor.numpy() - reconstructions.numpy())**2, axis=1)

        torch.save(model.state_dict(), os.path.join(model_dir, f'autoencoder_iter_{i}.pt'))

        metrics.append({
            'iteration': i,
            'mse_recon': mse_recon,
            'mse_pred': mse_pred,
            'r2': r2
        })

        anomaly_scores_df = pd.DataFrame({
            'iteration': i,
            'sample_index': np.arange(len(anomaly_scores)),
            'anomaly_score': anomaly_scores
        })
        anomaly_scores_all.append(anomaly_scores_df)

        embeddings_df = pd.DataFrame(embeddings.numpy())
        embeddings_df['iteration'] = i
        embeddings_all.append(embeddings_df)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    anomaly_scores_all_df = pd.concat(anomaly_scores_all, ignore_index=True)
    anomaly_scores_all_df.to_csv(os.path.join(output_dir, 'anomaly_scores.csv'), index=False)

    embeddings_all_df = pd.concat(embeddings_all, ignore_index=True)
    embeddings_all_df.to_csv(os.path.join(output_dir, 'embeddings.csv'), index=False)

    # Plot PCA visualization of embeddings
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_all_df.drop(columns='iteration'))
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=embeddings_all_df['iteration'], cmap='viridis', s=10)
    plt.colorbar(label='Iteration')
    plt.title('PCA of Autoencoder Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'embeddings_pca.png'))
    plt.close()

    # Calculate summary statistics
    for col in ['mse_recon', 'mse_pred', 'r2']:
        values = metrics_df[col]
        mean = values.mean()
        std = values.std()
        ci_lower, ci_upper = t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
        metrics_df[f'{col}_mean'] = mean
        metrics_df[f'{col}_std'] = std
        metrics_df[f'{col}_ci_lower'] = ci_lower
        metrics_df[f'{col}_ci_upper'] = ci_upper

    summary_df = metrics_df[[
        'mse_recon_mean', 'mse_recon_std', 'mse_recon_ci_lower', 'mse_recon_ci_upper',
        'mse_pred_mean', 'mse_pred_std', 'mse_pred_ci_lower', 'mse_pred_ci_upper',
        'r2_mean', 'r2_std', 'r2_ci_lower', 'r2_ci_upper']].drop_duplicates()
    summary_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)

    # Plot reconstruction MSE over iterations
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], metrics_df['mse_recon'], marker='o', label='Reconstruction MSE')
    plt.plot(metrics_df['iteration'], metrics_df['mse_pred'], marker='s', label='Prediction MSE')
    plt.title('Loss over Iterations (Autoencoder)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_over_iterations.png'))
    plt.close()

    print("✅ Semi-supervised Autoencoder training completed. Results saved to:", output_dir)

    return metrics_df, summary_df, anomaly_scores_all_df, embeddings_all_df
