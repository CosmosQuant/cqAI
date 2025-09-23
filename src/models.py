# src/models.py - ML model creation and management
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

def create_model(model_config):
    """
    Create a machine learning model based on configuration.
    
    Args:
        model_config (dict): Model configuration with 'algorithm' and 'params'
        
    Returns:
        sklearn-compatible model: Configured model instance
    """
    algorithm = model_config['algorithm']
    params = model_config.get('params', {})
    
    if algorithm == 'ridge':
        return Ridge(**params)
    
    elif algorithm == 'linear':
        return LinearRegression(**params)
    
    elif algorithm == 'random_forest':
        return RandomForestRegressor(random_state=42, **params)
    
    elif algorithm == 'lightgbm':
        # Set default parameters for LightGBM
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42
        }
        default_params.update(params)
        return LightGBMWrapper(**default_params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

class LightGBMWrapper:
    """
    Wrapper for LightGBM to make it sklearn-compatible.
    """
    
    def __init__(self, **params):
        self.params = params
        self.model = None
        
    def fit(self, X, y):
        """Fit the LightGBM model."""
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data)
        return self
        
    def predict(self, X):
        """Make predictions with the fitted model."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params.copy()
        
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.params.update(params)
        return self

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate model predictions with multiple metrics.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'correlation': np.nan,
            'n_samples': 0
        }
    
    metrics = {
        'mse': mean_squared_error(y_true_clean, y_pred_clean),
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'r2': r2_score(y_true_clean, y_pred_clean),
        'n_samples': len(y_true_clean)
    }
    
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation coefficient
    if len(y_true_clean) > 1:
        correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics['correlation'] = 0.0
    
    return metrics

def cross_validate_model(model, X, y, cv_splitter):
    """
    Perform cross-validation on a model.
    
    Args:
        model: sklearn-compatible model
        X (array-like): Feature matrix
        y (array-like): Target vector
        cv_splitter: Cross-validation splitter with split method
        
    Returns:
        list: List of CV results for each fold
    """
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_metrics = evaluate_predictions(y_train, y_pred_train)
        test_metrics = evaluate_predictions(y_test, y_pred_test)
        
        cv_result = {
            'fold': fold_idx,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
        
        cv_results.append(cv_result)
    
    return cv_results

def get_model_feature_importance(model, feature_names=None):
    """
    Extract feature importance from fitted model.
    
    Args:
        model: Fitted sklearn-compatible model
        feature_names (list, optional): Names of features
        
    Returns:
        dict or None: Feature importance scores
    """
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_)
    elif hasattr(model, 'model') and hasattr(model.model, 'feature_importance'):
        # LightGBM
        importance = model.model.feature_importance(importance_type='gain')
    
    if importance is not None:
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return importance
    
    return None

class EnsembleModel:
    """
    Simple ensemble model that averages predictions from multiple models.
    """
    
    def __init__(self, models, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            models (list): List of sklearn-compatible models
            weights (list, optional): Weights for each model
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred



