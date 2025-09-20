"""
Feature engineering utilities for the career recommender system.
Enhanced to handle structured skills from real job datasets.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Set, Union
from collections import Counter
import json


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


def calculate_skill_overlap(user_skills: Union[List[str], str], job_skills: Union[List[str], str]) -> float:
    """Calculate Jaccard similarity between user skills and job requirements"""
    # Handle different skill formats
    user_skills_set = normalize_skills_to_set(user_skills)
    job_skills_set = normalize_skills_to_set(job_skills)
    
    if not user_skills_set or not job_skills_set:
        return 0.0
    
    intersection = len(user_skills_set.intersection(job_skills_set))
    union = len(user_skills_set.union(job_skills_set))
    
    return intersection / union if union > 0 else 0.0


def normalize_skills_to_set(skills: Union[List[str], str, None]) -> Set[str]:
    """Normalize various skill formats to a clean set of skills"""
    if not skills or pd.isna(skills):
        return set()
    
    if isinstance(skills, str):
        # Handle JSON-like strings from real datasets
        if skills.startswith('[') and skills.endswith(']'):
            try:
                skills_list = json.loads(skills.replace("'", '"'))
                return set(clean_skill(skill) for skill in skills_list if skill)
            except (json.JSONDecodeError, TypeError):
                # Fall back to comma-separated parsing
                skills_list = [s.strip() for s in skills.split(',')]
                return set(clean_skill(skill) for skill in skills_list if skill)
        else:
            # Comma-separated skills
            skills_list = [s.strip() for s in skills.split(',')]
            return set(clean_skill(skill) for skill in skills_list if skill)
    
    elif isinstance(skills, list):
        return set(clean_skill(skill) for skill in skills if skill)
    
    return set()


def clean_skill(skill: str) -> str:
    """Clean and normalize individual skill names"""
    if not skill or pd.isna(skill):
        return ""
    
    skill = str(skill).lower().strip()
    
    # Remove common variations and normalize
    skill_mappings = {
        'js': 'javascript',
        'ts': 'typescript', 
        'py': 'python',
        'html5': 'html',
        'css3': 'css',
        'reactjs': 'react',
        'nodejs': 'node.js',
        'node': 'node.js',
        'postgresql': 'postgres',
        'amazon web services': 'aws',
        'google cloud platform': 'gcp',
        'microsoft azure': 'azure',
        'machine learning': 'ml',
        'artificial intelligence': 'ai',
        'data science': 'data analysis',
        'microsoft office': 'office',
        'microsoft excel': 'excel',
        'powerpoint': 'presentation',
        'power bi': 'powerbi'
    }
    
    # Apply mappings
    for old_skill, new_skill in skill_mappings.items():
        if old_skill == skill:
            skill = new_skill
            break
    
    return skill


def calculate_structured_skill_match(user_skills: Union[List[str], str], 
                                   job_skills: Union[List[str], str],
                                   job_type_skills: Union[List[str], str, None] = None) -> Dict[str, float]:
    """
    Calculate skill matching with enhanced analysis for structured skills
    
    Args:
        user_skills: User's skills (list or comma-separated string)
        job_skills: Job's required skills
        job_type_skills: Job type specific skills from structured data
    
    Returns:
        Dictionary with various skill match metrics
    """
    user_skills_set = normalize_skills_to_set(user_skills)
    job_skills_set = normalize_skills_to_set(job_skills) 
    job_type_skills_set = normalize_skills_to_set(job_type_skills) if job_type_skills else set()
    
    # Combine job skills with job type skills
    combined_job_skills = job_skills_set.union(job_type_skills_set)
    
    if not user_skills_set or not combined_job_skills:
        return {
            'skill_overlap': 0.0,
            'skill_coverage': 0.0,
            'skill_precision': 0.0,
            'exact_matches': 0,
            'total_job_skills': len(combined_job_skills),
            'total_user_skills': len(user_skills_set)
        }
    
    # Calculate various metrics
    intersection = user_skills_set.intersection(combined_job_skills)
    
    # Jaccard similarity (overlap)
    union = user_skills_set.union(combined_job_skills)
    skill_overlap = len(intersection) / len(union) if union else 0.0
    
    # Coverage (what % of job requirements user meets)
    skill_coverage = len(intersection) / len(combined_job_skills) if combined_job_skills else 0.0
    
    # Precision (what % of user skills are relevant)
    skill_precision = len(intersection) / len(user_skills_set) if user_skills_set else 0.0
    
    return {
        'skill_overlap': skill_overlap,
        'skill_coverage': skill_coverage, 
        'skill_precision': skill_precision,
        'exact_matches': len(intersection),
        'total_job_skills': len(combined_job_skills),
        'total_user_skills': len(user_skills_set)
    }


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
    """Create enhanced features for a user-job pair using structured skills"""
    features = {}
    
    # Basic compatibility features
    features['gpa_normalized'] = user_row.get('gpa_normalized', user_row.get('gpa', 0) / 4.0)
    features['experience_years'] = user_row.get('experience_years', 0)
    features['education_level_match'] = 1 if user_row.get('education_level_encoded', 0) >= job_row.get('education_requirement_encoded', 0) else 0
    features['experience_match'] = 1 if user_row.get('experience_years', 0) >= job_row.get('min_experience_years', 0) else 0
    
    # Enhanced skill matching with structured skills
    user_skills = user_row.get('skills_list', user_row.get('skills', []))
    job_skills = job_row.get('required_skills_list', job_row.get('required_skills', []))
    job_type_skills = job_row.get('job_type_skills')  # From real dataset
    
    skill_metrics = calculate_structured_skill_match(user_skills, job_skills, job_type_skills)
    
    # Add skill-based features
    features['skill_overlap'] = skill_metrics['skill_overlap']
    features['skill_coverage'] = skill_metrics['skill_coverage']
    features['skill_precision'] = skill_metrics['skill_precision']
    features['exact_skill_matches'] = skill_metrics['exact_matches']
    features['skill_match_ratio'] = skill_metrics['exact_matches'] / max(1, skill_metrics['total_job_skills'])
    
    # Education over-qualification (might be negative for some positions)
    features['education_overqualified'] = max(0, user_row.get('education_level_encoded', 0) - job_row.get('education_requirement_encoded', 0))
    
    # Experience over-qualification
    features['experience_overqualified'] = max(0, user_row.get('experience_years', 0) - job_row.get('min_experience_years', 0))
    
    # Salary features (normalized)
    salary_avg = job_row.get('salary_avg', 0)
    if pd.isna(salary_avg):
        salary_avg = 0
    features['salary_avg_normalized'] = float(salary_avg) / 150000.0  # Normalize by reasonable max
    
    # Location match (enhanced matching)
    features['location_match'] = calculate_location_match(
        user_row.get('preferred_location', ''),
        job_row.get('location', '')
    )
    
    # Industry alignment
    user_interests = str(user_row.get('interests', '')).lower()
    job_industry = str(job_row.get('industry', '')).lower()
    features['industry_interest_match'] = 1 if job_industry in user_interests else 0
    
    # Remote work preference
    user_location = str(user_row.get('preferred_location', '')).lower()
    job_location = str(job_row.get('location', '')).lower()
    is_remote_job = 'remote' in job_location or job_row.get('is_remote', False)
    wants_remote = 'remote' in user_location
    features['remote_preference_match'] = 1 if (is_remote_job and wants_remote) or (not is_remote_job and not wants_remote) else 0
    
    return features


def calculate_location_match(user_location: str, job_location: str) -> float:
    """Enhanced location matching with partial matches"""
    if not user_location or not job_location or pd.isna(user_location) or pd.isna(job_location):
        return 0.0
    
    user_loc = str(user_location).lower().strip()
    job_loc = str(job_location).lower().strip()
    
    # Exact match
    if user_loc == job_loc:
        return 1.0
    
    # Remote work
    if 'remote' in user_loc or 'remote' in job_loc:
        return 1.0 if ('remote' in user_loc and 'remote' in job_loc) else 0.8
    
    # City/state matching
    user_parts = [part.strip() for part in user_loc.replace(',', ' ').split()]
    job_parts = [part.strip() for part in job_loc.replace(',', ' ').split()]
    
    common_parts = set(user_parts).intersection(set(job_parts))
    if common_parts:
        # Partial match based on shared location components
        return min(0.8, len(common_parts) / max(len(user_parts), len(job_parts)))
    
    return 0.0


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
    """Get list of feature columns for ML models (enhanced with structured skills)"""
    return [
        'gpa_normalized',
        'experience_years', 
        'education_level_match',
        'experience_match',
        'skill_overlap',
        'skill_coverage',
        'skill_precision', 
        'exact_skill_matches',
        'skill_match_ratio',
        'education_overqualified',
        'experience_overqualified',
        'salary_avg_normalized',
        'location_match',
        'industry_interest_match',
        'remote_preference_match'
    ]


def analyze_skill_distribution(jobs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze skill distribution in job dataset for insights
    
    Args:
        jobs_df: DataFrame with job postings
    
    Returns:
        Dictionary with skill analysis
    """
    all_skills = []
    
    # Extract all skills from jobs
    for _, job in jobs_df.iterrows():
        job_skills = normalize_skills_to_set(job.get('required_skills', ''))
        job_type_skills = normalize_skills_to_set(job.get('job_type_skills', ''))
        combined_skills = job_skills.union(job_type_skills)
        all_skills.extend(list(combined_skills))
    
    # Count skill frequency
    skill_counts = Counter(all_skills)
    
    # Get industry-specific skills
    industry_skills = {}
    if 'industry' in jobs_df.columns:
        for industry in jobs_df['industry'].unique():
            if pd.isna(industry):
                continue
                
            industry_jobs = jobs_df[jobs_df['industry'] == industry]
            industry_skill_list = []
            
            for _, job in industry_jobs.iterrows():
                job_skills = normalize_skills_to_set(job.get('required_skills', ''))
                job_type_skills = normalize_skills_to_set(job.get('job_type_skills', ''))
                combined_skills = job_skills.union(job_type_skills)
                industry_skill_list.extend(list(combined_skills))
            
            industry_skills[industry] = Counter(industry_skill_list).most_common(10)
    
    return {
        'total_unique_skills': len(skill_counts),
        'most_common_skills': skill_counts.most_common(20),
        'industry_skills': industry_skills,
        'avg_skills_per_job': np.mean([len(normalize_skills_to_set(job.get('required_skills', ''))) for _, job in jobs_df.iterrows()]),
        'jobs_with_structured_skills': sum(1 for _, job in jobs_df.iterrows() if job.get('job_type_skills'))
    }


def create_skill_compatibility_matrix(users_df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compatibility matrix between users and jobs based on skills
    
    Args:
        users_df: DataFrame with user profiles
        jobs_df: DataFrame with job postings
    
    Returns:
        DataFrame with compatibility scores
    """
    compatibility_matrix = []
    
    for _, user in users_df.iterrows():
        user_skills = normalize_skills_to_set(user.get('skills', ''))
        
        for _, job in jobs_df.iterrows():
            job_skills = normalize_skills_to_set(job.get('required_skills', ''))
            job_type_skills = normalize_skills_to_set(job.get('job_type_skills', ''))
            
            skill_metrics = calculate_structured_skill_match(
                user_skills, job_skills, job_type_skills
            )
            
            compatibility_matrix.append({
                'user_id': user.get('user_id', user.name),
                'job_id': job.get('job_id', job.name), 
                'skill_overlap': skill_metrics['skill_overlap'],
                'skill_coverage': skill_metrics['skill_coverage'],
                'skill_precision': skill_metrics['skill_precision'],
                'exact_matches': skill_metrics['exact_matches']
            })
    
    return pd.DataFrame(compatibility_matrix)