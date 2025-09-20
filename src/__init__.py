"""
Career Recommender System - Core modules for job recommendation.

This package provides utilities for:
- Feature engineering from user profiles and job descriptions
- Model training with XGBoost rerankers
- Data fetching from external job APIs
- Complete recommendation pipeline with semantic search and reranking
"""

from .feature_engineering import (
    clean_text_fields,
    parse_skills_interests,
    calculate_skill_overlap,
    create_user_job_features,
    get_feature_columns
)

from .model_training import (
    RerankerModel,
    train_reranker_model,
    evaluate_ranking_metrics,
    hyperparameter_tuning
)

from .data_fetch import (
    JobDataFetcher,
    collect_job_data,
    get_sample_queries
)

from .recommender import CareerRecommender

__version__ = "1.0.0"
__all__ = [
    "clean_text_fields",
    "parse_skills_interests", 
    "calculate_skill_overlap",
    "create_user_job_features",
    "get_feature_columns",
    "RerankerModel",
    "train_reranker_model",
    "evaluate_ranking_metrics",
    "hyperparameter_tuning",
    "JobDataFetcher",
    "collect_job_data",
    "get_sample_queries",
    "CareerRecommender"
]