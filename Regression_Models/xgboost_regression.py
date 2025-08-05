import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import t
import xgboost as xgb
import joblib
import shap
import seaborn as sns


def run_xgboost_regression(X, y, n_iter, output_prefix, test_size=0.2, random_state=42):
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns)
    all_shap_values = []

    for i in range(n_iter):
        print(f'Iteration {i+1}/{n_iter}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state + i,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        all_shap_values.append(shap_values)
        
        # Save model
        joblib.dump({
            'model': model,
            'explainer': explainer
        }, f'{output_prefix}_model_iter_{i+1}.joblib')

        y_pred = model.predict(X_test)

        predictions_list.append(pd.DataFrame({
            'iteration': i,
            'true': y_test,
            'predicted': y_pred
        }))

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        metrics.append({
            'iteration': i,
            'r2': r2,
            'rmse': rmse,
            'mse': mse,
            'mae': mae
        })

        feature_importances[f'iter_{i}'] = model.feature_importances_

    # Save predictions
    predictions_df = pd.concat(predictions_list)
    predictions_df.to_csv(f'{output_prefix}_predictions.csv', index=False)

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{output_prefix}_metrics.csv', index=False)

    # Save feature importances
    feature_importances.to_csv(f'{output_prefix}_feature_importance.csv')

    # Calculate and save mean SHAP values
    mean_shap_values = np.mean(all_shap_values, axis=0)
    mean_shap_importance = np.abs(mean_shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap_value': mean_shap_importance
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

    # Plot SHAP dependence plots for top features
    top_features = shap_importance_df['feature'].head(5).tolist()
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            mean_shap_values,
            X_test,
            feature_names=X.columns,
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_shap_dependence_{feature}.png')
        plt.close()

    # Plot top features by SHAP importance
    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['feature'].head(10), shap_importance_df['mean_shap_value'].head(10))
    plt.title('Top 10 Features by SHAP Importance')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_shap_features.png')
    plt.close()

    # Calculate summary statistics
    summary = {}
    for metric in ['r2', 'mse', 'mae']:
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
    summary_df.to_csv(f'{output_prefix}_metrics_summary.csv')

    # Plot R^2 across iterations with confidence intervals
    r2_values = metrics_df['r2'].values
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    ci_upper = mean_r2 + 1.96 * std_r2 / np.sqrt(n_iter)
    ci_lower = mean_r2 - 1.96 * std_r2 / np.sqrt(n_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], r2_values, marker='o', label='R² per iteration')
    plt.axhline(mean_r2, color='green', linestyle='--', label='Mean R²')
    plt.fill_between(metrics_df['iteration'], ci_lower, ci_upper, color='green', alpha=0.2, label='95% CI')
    plt.title('R² Score over Iterations (XGBoost)')
    plt.xlabel('Iteration')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_r2_plot.png')
    plt.close()

    # Plot feature importance comparison
    plt.figure(figsize=(12, 6))
    feature_importance_mean = feature_importances.mean(axis=1).sort_values(ascending=True)
    feature_importance_mean.tail(10).plot(kind='barh')
    plt.title('Top 10 Features by XGBoost Importance')
    plt.xlabel('Mean Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_feature_importance_plot.png')
    plt.close()

    print("XGBoost regression completed. Results saved to:", output_prefix)

    return {
        'metrics': metrics_df,
        'predictions': predictions_df,
        'feature_importances': feature_importances,
        'shap_importance': shap_importance_df,
        'summary': summary_df
    }