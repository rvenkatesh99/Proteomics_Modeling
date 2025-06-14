import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import t
import xgboost as xgb


def run_xgboost_regression(X, y, n_iter, output_prefix, test_size=0.2, random_state=42):
    metrics = []
    predictions_list = []
    feature_importances = pd.DataFrame(index=X.columns)

    for i in range(n_iter):
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
        
        # Save model
        joblib.dump({'model': model}, f'{output_prefix}_model_iter_{i+1}.joblib')

        y_pred = model.predict(X_test)

        predictions_list.append(pd.DataFrame({
            'iteration': i,
            'true': y_test,
            'predicted': y_pred
        }))

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        metrics.append({
            'iteration': i,
            'r2': r2,
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

    # Plot R^2 across iterations
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['iteration'], metrics_df['r2'], marker='o')
    plt.title('R² Score over Iterations (XGBoost)')
    plt.xlabel('Iteration')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_r2_plot.png')
    plt.close()

    print("XGBoost regression completed. Results saved to:", output_prefix)

    return model, metrics_df, feature_importances, summary_df