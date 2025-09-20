"""
Model training utilities for the career recommender system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import json
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Evaluation metrics
from sklearn.metrics import ndcg_score, precision_score, recall_score


class RerankerModel:
    """XGBoost-based reranker for career recommendations"""
    
    def __init__(self, model_params: Dict[str, Any] = None):
        """Initialize reranker model"""
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.params = {**default_params, **(model_params or {})}
        self.model = xgb.XGBClassifier(**self.params)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def prepare_features(self, features_df: pd.DataFrame, feature_columns: List[str] = None) -> np.ndarray:
        """Prepare features for training/prediction"""
        if feature_columns is None:
            feature_columns = [col for col in features_df.columns 
                             if col not in ['user_id', 'job_id', 'label']]
        
        X = features_df[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return X, feature_columns
    
    def train(self, features_df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the reranker model"""
        # Prepare features and labels
        X, self.feature_columns = self.prepare_features(features_df)
        y = features_df['label'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric='logloss',
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.is_fitted = True
        
        # Evaluate on validation set
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        y_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            'auc': roc_auc_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        return metrics
    
    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for user-job pairs"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(features_df, self.feature_columns)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict binary labels for user-job pairs"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(features_df, self.feature_columns)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def save(self, model_path: Path):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'params': self.params
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, model_path: Path) -> 'RerankerModel':
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.is_fitted = True
        
        return instance


def train_reranker_model(features_df: pd.DataFrame, 
                        model_params: Dict[str, Any] = None,
                        validation_split: float = 0.2) -> Tuple[RerankerModel, Dict[str, Any]]:
    """Train reranker model with given features"""
    
    print(f"Training reranker with {len(features_df)} samples")
    print(f"Positive samples: {features_df['label'].sum()}")
    print(f"Negative samples: {len(features_df) - features_df['label'].sum()}")
    
    # Initialize and train model
    reranker = RerankerModel(model_params)
    metrics = reranker.train(features_df, validation_split)
    
    print("Training completed!")
    print(f"Validation AUC: {metrics['auc']:.4f}")
    print(f"Validation Precision: {metrics['precision']:.4f}")
    print(f"Validation Recall: {metrics['recall']:.4f}")
    
    return reranker, metrics


def evaluate_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                           user_groups: np.ndarray = None, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """Evaluate ranking metrics like NDCG@k, Precision@k"""
    metrics = {}
    
    if user_groups is None:
        # Treat as single group
        for k in k_values:
            if len(y_true) >= k:
                # NDCG@k
                ndcg_k = ndcg_score([y_true], [y_scores], k=k)
                metrics[f'ndcg@{k}'] = ndcg_k
                
                # Precision@k
                top_k_indices = np.argsort(y_scores)[-k:]
                precision_k = y_true[top_k_indices].sum() / k
                metrics[f'precision@{k}'] = precision_k
    else:
        # Group by users and average metrics
        unique_users = np.unique(user_groups)
        
        for k in k_values:
            ndcg_scores = []
            precision_scores = []
            
            for user in unique_users:
                user_mask = user_groups == user
                user_true = y_true[user_mask]
                user_scores = y_scores[user_mask]
                
                if len(user_true) >= k and user_true.sum() > 0:
                    # NDCG@k for this user
                    ndcg_k = ndcg_score([user_true], [user_scores], k=k)
                    ndcg_scores.append(ndcg_k)
                    
                    # Precision@k for this user
                    top_k_indices = np.argsort(user_scores)[-k:]
                    precision_k = user_true[top_k_indices].sum() / k
                    precision_scores.append(precision_k)
            
            if ndcg_scores:
                metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
                metrics[f'precision@{k}'] = np.mean(precision_scores)
    
    return metrics


def hyperparameter_tuning(features_df: pd.DataFrame, 
                         param_grid: Dict[str, List] = None,
                         cv_folds: int = 3) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Perform hyperparameter tuning using GridSearchCV"""
    
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
    
    # Prepare data
    X = features_df[[col for col in features_df.columns 
                    if col not in ['user_id', 'job_id', 'label']]].values
    y = features_df['label'].values
    
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid search
    xgb_model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        xgb_model, param_grid, 
        cv=cv_folds, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return best_params, {'cv_auc': best_score}