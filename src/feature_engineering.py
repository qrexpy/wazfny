"""
Feature engineering utilities for the career recommender system.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


def clean_text_fields(text: str) -> str:
    """Clean and normalize text fields"""
    if pd.isna(text):
        return ""
    # Convert to lowercase, remove extra spaces
    text = str(text).lower().strip()
    # Remove special characters but keep commas for skill/interest separation
    text = re.sub(r'[^\w\s,]', '', text)
    return text


def parse_skills_interests(text: str) -> List[str]:
    """Parse comma-separated skills/interests into list"""
    if pd.isna(text) or text == "":
        return []
    items = [item.strip() for item in str(text).split(',')]
    return [item for item in items if item]  # Remove empty strings


def calculate_skill_overlap(user_skills: List[str], job_skills: List[str]) -> float:
    """Calculate Jaccard similarity between user skills and job requirements"""
    if not user_skills or not job_skills:
        return 0.0
    
    user_set = set(user_skills)
    job_set = set(job_skills)
    intersection = len(user_set.intersection(job_set))
    union = len(user_set.union(job_set))
    
    return intersection / union if union > 0 else 0.0


def extract_experience_years(exp_text: str) -> int:
    """Extract minimum experience years from requirement text"""
    if pd.isna(exp_text):
        return 0
    
    # Look for patterns like "2-4 years", "3+ years", "1-3 years"
    numbers = re.findall(r'(\d+)', str(exp_text))
    if numbers:
        return int(numbers[0])  # Take first number as minimum
    return 0


def parse_salary_range(salary_text: str) -> Tuple[int, int, int]:
    """Parse salary range and return min, max, and average"""
    if pd.isna(salary_text):
        return 0, 0, 0
    
    numbers = re.findall(r'(\d+)', str(salary_text))
    if len(numbers) >= 2:
        min_sal, max_sal = int(numbers[0]), int(numbers[1])
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    elif len(numbers) == 1:
        sal = int(numbers[0])
        return sal, sal, sal
    return 0, 0, 0


def create_user_job_features(user_row: pd.Series, job_row: pd.Series) -> Dict[str, Any]:
    """Create features for a user-job pair"""
    features = {}
    
    # Basic compatibility features
    features['gpa_normalized'] = user_row.get('gpa_normalized', user_row.get('gpa', 0) / 4.0)
    features['experience_years'] = user_row.get('experience_years', 0)
    features['education_level_match'] = 1 if user_row.get('education_level_encoded', 0) >= job_row.get('education_requirement_encoded', 0) else 0
    features['experience_match'] = 1 if user_row.get('experience_years', 0) >= job_row.get('min_experience_years', 0) else 0
    
    # Skill overlap
    user_skills = user_row.get('skills_list', [])
    job_skills = job_row.get('required_skills_list', [])
    features['skill_overlap'] = calculate_skill_overlap(user_skills, job_skills)
    
    # Education over-qualification (might be negative for some positions)
    features['education_overqualified'] = max(0, user_row.get('education_level_encoded', 0) - job_row.get('education_requirement_encoded', 0))
    
    # Experience over-qualification
    features['experience_overqualified'] = max(0, user_row.get('experience_years', 0) - job_row.get('min_experience_years', 0))
    
    # Salary features (normalized)
    features['salary_avg_normalized'] = job_row.get('salary_avg', 0) / 150000.0  # Normalize by reasonable max
    
    # Location match (simple string matching)
    user_location = str(user_row.get('preferred_location', '')).lower()
    job_location = str(job_row.get('location', '')).lower()
    features['location_match'] = 1 if (user_location in job_location or 
                                     job_location in user_location or 
                                     'remote' in user_location or 
                                     'remote' in job_location) else 0
    
    return features


def create_training_features(users_df: pd.DataFrame, jobs_df: pd.DataFrame, 
                           user_job_pairs: List[Tuple[int, int, int]] = None) -> pd.DataFrame:
    """
    Create training features for user-job pairs
    
    Args:
        users_df: DataFrame with user profiles
        jobs_df: DataFrame with job catalog
        user_job_pairs: List of (user_id, job_id, label) tuples. If None, generates all pairs.
    
    Returns:
        DataFrame with features for each user-job pair
    """
    features_list = []
    
    if user_job_pairs is None:
        # Generate all possible pairs (expensive for large datasets)
        user_job_pairs = []
        for _, user in users_df.iterrows():
            for _, job in jobs_df.iterrows():
                # Simple heuristic for creating labels
                label = 1 if (user.get('education_level_encoded', 0) >= job.get('education_requirement_encoded', 0) and
                             calculate_skill_overlap(user.get('skills_list', []), job.get('required_skills_list', [])) > 0.1) else 0
                user_job_pairs.append((user['user_id'], job['job_id'], label))
    
    users_dict = users_df.set_index('user_id').to_dict('index')
    jobs_dict = jobs_df.set_index('job_id').to_dict('index')
    
    for user_id, job_id, label in user_job_pairs:
        if user_id in users_dict and job_id in jobs_dict:
            user_row = pd.Series(users_dict[user_id])
            job_row = pd.Series(jobs_dict[job_id])
            
            features = create_user_job_features(user_row, job_row)
            features['user_id'] = user_id
            features['job_id'] = job_id
            features['label'] = label
            
            features_list.append(features)
    
    return pd.DataFrame(features_list)


def get_feature_columns() -> List[str]:
    """Get list of feature columns for ML models"""
    return [
        'gpa_normalized',
        'experience_years', 
        'education_level_match',
        'experience_match',
        'skill_overlap',
        'education_overqualified',
        'experience_overqualified',
        'salary_avg_normalized',
        'location_match'
    ]