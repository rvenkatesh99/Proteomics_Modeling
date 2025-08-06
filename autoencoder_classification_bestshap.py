import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    classification_report, mean_squared_error, roc_curve
)
from scipy.interpolate import interp1d
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
        'n_encoder_layers': trial.suggest_int('n_encoder_layers', 2, 3),
        'n_decoder_layers': trial.suggest_int('n_decoder_layers', 2, 3),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'classifier_units': trial.suggest_int('classifier_units', 16, 32)
    }
    
    # Add encoder + decoder units with smaller ranges
    for i in range(params['n_encoder_layers']):
        params[f'encoder_units_{i}'] = trial.suggest_int(f'encoder_units_{i}', 32, 128)  # reduced max
    for i in range(params['n_decoder_layers']):
        params[f'decoder_units_{i}'] = trial.suggest_int(f'decoder_units_{i}', 32, 128)  # reduced max
    
    model = AdvancedAutoencoderClassifier(X_train.shape[1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.BCELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # larger batch size
    
    best_val_loss = float('inf')
    patience = 5
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
            
            try:
                trial.report(val_loss.item(), epoch)
                
                # Prune if the trial is performing poorly
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except AttributeError as e:
                if "module 'numpy' has no attribute 'float'" in str(e):
                    pass
                else:
                    raise e
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
    
    return best_val_loss.item()

def run_autoencoder_classification(X, y, n_iter=5, test_size=0.2, output_dir="", random_state=42, n_trials=10):
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
    all_probabilities = []
    all_models = []
    all_X_tests = []
    all_y_tests = []
    all_X_test_scaled = []
    
    for i in range(n_iter):
        print(f'Iteration {i + 1}/{n_iter}')
        
        # split data
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
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=2)
            study = optuna.create_study(direction='minimize', pruner=pruner)
        except Exception as e:
            if "module 'numpy' has no attribute 'float'" in str(e):
                # Fallback to no pruner if NumPy compatibility issue
                print("Warning: Using fallback mode without pruner due to NumPy compatibility issue")
                study = optuna.create_study(direction='minimize')
            else:
                raise e
                
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
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        
        # Training with early stopping
        best_model = None
        best_val_loss = float('inf')
        patience = 5  # reduced patience
        patience_counter = 0
        
        for epoch in range(100):  # reduced max epochs
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
            
            # Check probability distribution per iter
            prob_np = probabilities.cpu().numpy()
            print(f"Iteration {i} - Probabilities: min={prob_np.min():.4f}, max={prob_np.max():.4f}, "
                  f"mean={prob_np.mean():.4f}, std={prob_np.std():.4f}")
            print(f"Iteration {i} - Predictions: {predictions.sum().item()}/{len(predictions)} positive")
            
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
        
        # Save model
        all_models.append(model)
        all_X_tests.append(X_test)
        all_y_tests.append(y_test)
        all_X_test_scaled.append(X_test_scaled)
        all_probabilities.append(probabilities.cpu().numpy())

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
    
    # summary statistics
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
    summary_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    
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
    plt.title('Mean ROC Curve with Confidence Interval (Autoencoder)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_summary.png'))
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
    
    # confusion matrix for the best iter
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
    
    # SHAP analysis only for the best model using DeepExplainer
    best_model_idx = metrics_df['auroc'].idxmax()
    best_model = all_models[best_model_idx]
    best_X_test_scaled = all_X_test_scaled[best_model_idx]
    
    class ClassifierWrapper(nn.Module):
        def __init__(self, autoencoder_model):
            super(ClassifierWrapper, self).__init__()
            self.autoencoder = autoencoder_model
            
        def forward(self, x):
            # Only return the classification output
            _, _, pred = self.autoencoder(x)
            return pred.unsqueeze(1)  # Add dimension for DeepExplainer
    
    # Convert to tensor for DeepExplainer
    best_X_test_tensor = torch.tensor(best_X_test_scaled, dtype=torch.float32).to(device)
    
    # Create wrapper
    classifier_model = ClassifierWrapper(best_model).to(device)
    classifier_model.eval()
    
    # DeepExplainer for faster SHAP 
    background_size = min(50, len(best_X_test_scaled))
    explainer = shap.DeepExplainer(
        classifier_model,
        best_X_test_tensor[:background_size]  # Use smaller background
    )
    
    test_subset_size = min(200, len(best_X_test_tensor))
    shap_values = explainer.shap_values(best_X_test_tensor[:test_subset_size])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Calculate and save mean SHAP values
    mean_shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=False)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        all_X_tests[best_model_idx].iloc[:test_subset_size],  # Use same subset
        feature_names=X.columns,
        show=False,
        max_display=20 
    )
    plt.title('SHAP Feature Importance Summary (Best Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()
    
    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance (Best Model)')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_shap_features.png'))
    plt.close()
    
    print("Autoencoder classification completed. Results saved to:", output_dir)
    print(f"Best model (iteration {best_iter}) AUROC: {auroc_values[best_iter]:.4f}")
    
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'summary': summary_df,
        'shap_importance': shap_importance_df,
        'anomaly_scores': anomaly_scores_all_df,
        'embeddings': embeddings_all_df,
        'best_iteration': best_iter
    }