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
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import shap
import seaborn as sns
import optuna
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
import copy

class AdvancedAutoencoderClassifier(nn.Module):
    def __init__(self, input_dim: int, params: Dict[str, Any]):
        super(AdvancedAutoencoderClassifier, self).__init__()
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        
        for i in range(params['n_encoder_layers']):
            next_dim = params[f'encoder_units_{i}']
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(params['dropout_rate'])
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        current_dim = params[f'encoder_units_{params["n_encoder_layers"]-1}']
        
        for i in range(params['n_decoder_layers']):
            next_dim = params[f'decoder_units_{i}']
            decoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(params['dropout_rate'])
            ])
            current_dim = next_dim
        
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(params[f'encoder_units_{params["n_encoder_layers"]-1}'], params['classifier_units']),
            nn.ReLU(),
            nn.Dropout(params['dropout_rate']),
            nn.Linear(params['classifier_units'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        pred = self.classifier(encoded)
        return decoded, encoded, pred.squeeze()

def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> float:
    params = {
        'n_encoder_layers': trial.suggest_int('n_encoder_layers', 2, 4),
        'n_decoder_layers': trial.suggest_int('n_decoder_layers', 2, 4),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'classifier_units': trial.suggest_int('classifier_units', 16, 64)
    }
    
    # Add encoder units
    for i in range(params['n_encoder_layers']):
        params[f'encoder_units_{i}'] = trial.suggest_int(f'encoder_units_{i}', 32, 256)
    
    # Add decoder units
    for i in range(params['n_decoder_layers']):
        params[f'decoder_units_{i}'] = trial.suggest_int(f'decoder_units_{i}', 32, 256)
    
    model = AdvancedAutoencoderClassifier(X_train.shape[1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.BCELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            outputs, _, preds = model(inputs)
            
            loss_recon = reconstruction_criterion(outputs, inputs)
            loss_class = classification_criterion(preds, targets)
            loss = loss_recon + loss_class
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs, _, val_preds = model(X_val_tensor)
            val_loss_recon = reconstruction_criterion(val_outputs, X_val_tensor)
            val_loss_class = classification_criterion(val_preds, y_val_tensor)
            val_loss = val_loss_recon + val_loss_class
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
    
    return best_val_loss.item()

def run_autoencoder_classification(X, y, n_iter=10, test_size=0.2, output_dir="autoencoder_classification_results", random_state=42, n_trials=20):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    
    metrics = []
    predictions_list = []
    reconstructions_list = []
    anomaly_scores_all = []
    embeddings_all = []
    all_shap_values = []
    all_probabilities = []
    
    for i in range(n_iter):
        print(f'Iteration {i + 1}/{n_iter}')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i, stratify=y
        )
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=random_state + i, stratify=y_train
        )
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val, device),
            n_trials=n_trials
        )
        
        best_params = study.best_params
        model = AdvancedAutoencoderClassifier(X.shape[1], best_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        reconstruction_criterion = nn.MSELoss()
        classification_criterion = nn.BCELoss()
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        
        # Training with early stopping
        best_model = None
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(300):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                outputs, _, preds = model(inputs)
                
                loss_recon = reconstruction_criterion(outputs, inputs)
                loss_class = classification_criterion(preds, targets)
                loss = loss_recon + loss_class
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs, _, val_preds = model(X_train_tensor)
                val_loss_recon = reconstruction_criterion(val_outputs, X_train_tensor)
                val_loss_class = classification_criterion(val_preds, y_train_tensor)
                val_loss = val_loss_recon + val_loss_class
                
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
            reconstructions, embeddings, probabilities = model(X_test_tensor)
            predictions = (probabilities > 0.5).float()
            
            # Calculate metrics
            auroc = roc_auc_score(y_test_tensor.cpu().numpy(), probabilities.cpu().numpy())
            auprc = average_precision_score(y_test_tensor.cpu().numpy(), probabilities.cpu().numpy())
            precision = precision_score(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
            recall = recall_score(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
            f1 = f1_score(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
            balanced_acc = balanced_accuracy_score(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
            brier = brier_score_loss(y_test_tensor.cpu().numpy(), probabilities.cpu().numpy())
            
            mse_recon = mean_squared_error(X_test_scaled, reconstructions.cpu().numpy())
            anomaly_scores = np.mean((X_test_scaled - reconstructions.cpu().numpy())**2, axis=1)
        
        # Calculate SHAP values
        background = shap.kmeans(X_train_scaled, 100)
        explainer = shap.KernelExplainer(
            lambda x: model(torch.tensor(x, dtype=torch.float32).to(device))[2].cpu().detach().numpy(),
            background
        )
        shap_values = explainer.shap_values(X_test_scaled)
        all_shap_values.append(shap_values)
        all_probabilities.append(probabilities.cpu().numpy())
        
        # Save model and parameters
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': best_params
        }, os.path.join(model_dir, f'autoencoder_iter_{i}.pt'))
        
        metrics.append({
            'iteration': i,
            'auroc': auroc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'brier_score': brier,
            'mse_recon': mse_recon,
            **best_params
        })
        
        predictions_list.append(pd.DataFrame({
            'iteration': i,
            'true': y_test_tensor.cpu().numpy(),
            'predicted': predictions.cpu().numpy(),
            'probability': probabilities.cpu().numpy()
        }))
        
        anomaly_scores_df = pd.DataFrame({
            'iteration': i,
            'sample_index': np.arange(len(anomaly_scores)),
            'anomaly_score': anomaly_scores
        })
        anomaly_scores_all.append(anomaly_scores_df)
        
        embeddings_df = pd.DataFrame(embeddings.cpu().numpy())
        embeddings_df['iteration'] = i
        embeddings_all.append(embeddings_df)
    
    # Save results
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    predictions_df = pd.concat(predictions_list)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    anomaly_scores_all_df = pd.concat(anomaly_scores_all, ignore_index=True)
    anomaly_scores_all_df.to_csv(os.path.join(output_dir, 'anomaly_scores.csv'), index=False)
    
    embeddings_all_df = pd.concat(embeddings_all, ignore_index=True)
    embeddings_all_df.to_csv(os.path.join(output_dir, 'embeddings.csv'), index=False)
    
    # Calculate and save mean SHAP values
    mean_shap_values = np.mean(all_shap_values, axis=0)
    mean_shap_importance = np.abs(mean_shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=False)
    
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
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()
    
    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_shap_features.png'))
    plt.close()
    
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
    plt.title('AUROC Score over Iterations (Autoencoder)')
    plt.xlabel('Iteration')
    plt.ylabel('AUROC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'auroc_plot.png'))
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
    plt.title('AUPRC Score over Iterations (Autoencoder)')
    plt.xlabel('Iteration')
    plt.ylabel('AUPRC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'auprc_plot.png'))
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    all_probs = np.concatenate(all_probabilities)
    plt.hist(all_probs, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()
    
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
    summary_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    
    print("✅ Autoencoder classification completed. Results saved to:", output_dir)
    
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': shap_importance_df,
        'summary': summary_df,
        'anomaly_scores': anomaly_scores_all_df,
        'embeddings': embeddings_all_df
    } 