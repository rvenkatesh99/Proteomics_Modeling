import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from scipy import interp  # or use np.interp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix, 
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    average_precision_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

def run_elasticnet_classification(X, y, n_iter, output_prefix, test_size=0.2, random_state=42):
    feature_names = X.columns
    all_metrics = []
    all_preds = []
    all_feature_importances = pd.DataFrame(index=feature_names)
    all_shap_values = []

    os.makedirs(output_prefix, exist_ok=True)

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state + i
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression with ElasticNet regularization
        model = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[.1, .5, .7, .9, .95, .99, 1],
            Cs=10,
            cv=5,
            max_iter=10000,
            random_state=random_state + i,
            scoring='roc_auc'
        )
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probas = model.predict_proba(X_test_scaled)[:, 1]

        # Use a small background for SHAP
        background = shap.sample(X_train, 100, random_state=42) if X_train.shape[0] > 100 else X_train

        # Use LinearExplainer for ElasticNet
        explainer = shap.LinearExplainer(model, background, feature_dependence="independent")

        # Limit test samples for SHAP calculation
        X_shap = X_test if X_test.shape[0] <= 100 else shap.sample(X_test, 100, random_state=42)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_shap)

        # Save SHAP values (no duplicates)
        shap_df = pd.DataFrame(shap_values, columns=X_test.columns if hasattr(X_test, 'columns') else None)
        shap_df.to_csv(f'{output_prefix}/shap_values_iter_{i+1}.csv', index=False)

        # Metrics
        metrics = {
            'iteration': i,
            'accuracy': accuracy_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probas),
            'auprc': average_precision_score(y_test, probas),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'balanced_accuracy': balanced_accuracy_score(y_test, preds),
            'brier_score': brier_score_loss(y_test, probas)
        }
        all_metrics.append(metrics)

        # Predictions
        df_preds = pd.DataFrame({
            'iteration': i,
            'sample_index': y_test.index,
            'true_label': y_test.values,
            'predicted_label': preds,
            'predicted_proba': probas
        })
        all_preds.append(df_preds)

        # Feature importances
        all_feature_importances[f'Iter_{i+1}'] = model.coef_.flatten()

        # Save model
        joblib.dump({'model': model, 'scaler': scaler}, f'{output_prefix}/model_iter_{i+1}.joblib')

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix Iteration {i+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}/confusion_matrix_iter_{i+1}.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probas)
        plt.plot(fpr, tpr, label=f'Iter {i+1} (AUC = {metrics["roc_auc"]:.2f})')

    # Save summary outputs
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_preds)
    all_feature_importances.to_csv(f'{output_prefix}/feature_importances.csv')
    metrics_df.to_csv(f'{output_prefix}/metrics.csv', index=False)
    predictions_df.to_csv(f'{output_prefix}/predictions.csv', index=False)

    # Mean and std of feature importances
    feature_stats = pd.DataFrame({
        'mean': all_feature_importances.mean(axis=1),
        'std': all_feature_importances.std(axis=1)
    })
    feature_stats.to_csv(f'{output_prefix}/feature_importances_mean_sd.csv')

    # SHAP summary plot
    mean_shap_values = np.mean(all_shap_values, axis=0)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(mean_shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/shap_summary.png')
    plt.close()

    # ROC curves
    # Interpolation setup
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    plt.figure(figsize=(6, 5))

    # Loop through predictions
    for i, metrics in enumerate(all_metrics):
        y_test = all_preds[i]['true_label'].values
        probas = all_preds[i]['predicted_proba'].values
        fpr, tpr, _ = roc_curve(y_test, probas)
        
        # Interpolate TPRs at common FPR points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        auc = roc_auc_score(y_test, probas)
        aucs.append(auc)

    # Compute mean and std dev
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Plot mean ROC and CI
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label='± 1 std. dev.')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Summary ROC Curve with Confidence Interval')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}/roc_curves_summary.png')
    plt.close()

    print(f"✅ All results saved to: {output_prefix}")
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_stats': feature_stats
    }
