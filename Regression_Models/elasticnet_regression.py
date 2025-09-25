import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os
import shap
import seaborn as sns

def run_elasticnet_quantitative(X, y, n_iter, output_prefix, test_size=0.2, random_state=42):
    feature_names = X.columns
    all_metrics = []
    all_preds = []
    all_feature_importances = pd.DataFrame(index=feature_names)
    all_shap_values = []

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ElasticNetCV
        model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=np.logspace(-4, 4, 100),
            cv=5,
            max_iter=10000,
            random_state=random_state + i
        )
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        # Calculate SHAP values
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
        all_shap_values.append(shap_values)

        # Metrics
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        metrics = {
            'iteration': i,
            'r2': r2,
            'rmse': rmse,
            'best_alpha': model.alpha_,
            'best_l1_ratio': model.l1_ratio_
        }
        all_metrics.append(metrics)

        # Predictions
        df_preds = pd.DataFrame({
            'iteration': i,
            'sample_index': y_test.index,
            'true_value': y_test.values,
            'predicted_value': preds
        })
        all_preds.append(df_preds)

        # Coefficients
        all_feature_importances[f'Iter_{i+1}'] = model.coef_

        # Save model
        joblib.dump({'model': model, 'scaler': scaler}, f'{output_prefix}_model_iter_{i+1}.joblib')

    # Convert outputs
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_preds)
    all_feature_importances.to_csv(f'{output_prefix}_feature_importances.csv')
    metrics_df.to_csv(f'{output_prefix}_metrics.csv', index=False)
    predictions_df.to_csv(f'{output_prefix}_predictions.csv', index=False)

    # Feature Importances: Mean and Std
    feature_stats = pd.DataFrame({
        'mean': all_feature_importances.mean(axis=1),
        'std': all_feature_importances.std(axis=1)
    })
    feature_stats.to_csv(f'{output_prefix}_feature_importances_mean_sd.csv')

    # Calculate mean SHAP values across iterations
    mean_shap_values = np.mean(all_shap_values, axis=0)
    mean_shap_importance = np.abs(mean_shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_shap_value': mean_shap_importance
    }).sort_values('mean_shap_value', ascending=False)
    shap_importance_df.to_csv(f'{output_prefix}_shap_importance.csv', index=False)

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        mean_shap_values,
        X_test_scaled,
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_shap_summary.png')
    plt.close()

    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=shap_importance_df.head(10),
        x='mean_shap_value',
        y='feature'
    )
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_shap_features.png')
    plt.close()

    # Plot R² with 95% CI
    r2_values = metrics_df['r2'].values
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    ci_upper = mean_r2 + 1.96 * std_r2 / np.sqrt(n_iter)
    ci_lower = mean_r2 - 1.96 * std_r2 / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], r2_values, label='R² per iteration', marker='o')
    plt.hlines(mean_r2, xmin=0, xmax=n_iter-1, colors='green', linestyles='dashed', label='Mean R²')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.xlabel('Iteration')
    plt.ylabel('R²')
    plt.title('ElasticNet R² with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_r2_plot.png')
    plt.close()

    print(f"All results saved with prefix: {output_prefix}")
    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_stats': feature_stats,
        'shap_importance': shap_importance_df
    }
