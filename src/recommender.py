"""
Main recommender system class that combines embedding-based retrieval with reranking.
"""

import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle
import json
from sentence_transformers import SentenceTransformer

from .feature_engineering import create_user_job_features, get_feature_columns
from .model_training import RerankerModel


class CareerRecommender:
    """Career recommendation system combining semantic search with learned reranking"""
    
    def __init__(self, models_dir: Path):
        """Initialize recommender with models directory"""
        self.models_dir = Path(models_dir)
        self.embedding_model = None
        self.reranker_model = None
        self.job_index = None
        self.jobs_df = None
        self.users_df = None
        self.metadata = None
        self.is_loaded = False
    
    def load_models(self):
        """Load all trained models and data"""
        print("Loading career recommender models...")
        
        # Load metadata
        with open(self.models_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load embedding model - prefer JobBERT-v3 for job-specific embeddings
        model_name = self.metadata.get('embedding_model', 'TechWolf/JobBERT-v3')
        
        # Use JobBERT-v3 if available, fallback to original model
        try:
            if 'JobBERT' not in model_name:
                print(f"Upgrading to JobBERT-v3 for better job title embeddings...")
                model_name = 'TechWolf/JobBERT-v3'
            
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load processed data
        self.users_df = pd.read_pickle(self.models_dir / "users_processed.pkl")
        self.jobs_df = pd.read_pickle(self.models_dir / "jobs_processed.pkl")
        
        # Load FAISS index
        self.job_index = faiss.read_index(str(self.models_dir / "job_index.faiss"))
        
        # Load reranker model
        reranker_path = self.models_dir / "reranker_model.pkl"
        if reranker_path.exists():
            self.reranker_model = RerankerModel.load(reranker_path)
            print("Reranker model loaded successfully")
        else:
            print("Warning: Reranker model not found. Using embedding similarity only.")
        
        self.is_loaded = True
        print("All models loaded successfully!")
    
    def get_user_embedding(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for user profile"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded before generating embeddings")
        
        # Create profile text similar to training data
        profile_text = f"{user_profile.get('field_of_study', '')} {user_profile.get('interests', '')} {user_profile.get('skills', '')}"
        profile_text = profile_text.strip().lower()
        
        embedding = self.embedding_model.encode([profile_text], convert_to_numpy=True)
        return embedding[0]
    
    def semantic_search(self, user_embedding: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic search to find candidate jobs"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded before search")
        
        # Search FAISS index
        user_embedding = user_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.job_index.search(user_embedding, k)
        
        return distances[0], indices[0]
    
    def rerank_candidates(self, user_profile: Dict[str, Any], candidate_job_indices: np.ndarray) -> np.ndarray:
        """Rerank candidate jobs using trained reranker model"""
        if not self.reranker_model:
            # If no reranker, return original order
            return np.arange(len(candidate_job_indices))
        
        # Create features for user-job pairs
        features_list = []
        
        # Convert user profile to required format
        user_row = self._profile_to_series(user_profile)
        
        for job_idx in candidate_job_indices:
            job_row = self.jobs_df.iloc[job_idx]
            features = create_user_job_features(user_row, job_row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Get reranking scores
        scores = self.reranker_model.predict_proba(features_df)
        
        # Return indices sorted by score (descending)
        return np.argsort(scores)[::-1]
    
    def recommend_jobs(self, user_profile: Dict[str, Any], 
                      num_candidates: int = 50, 
                      num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Generate job recommendations for a user
        
        Args:
            user_profile: Dictionary with user information
            num_candidates: Number of candidates from semantic search
            num_recommendations: Final number of recommendations to return
        
        Returns:
            List of recommended jobs with scores and explanations
        """
        if not self.is_loaded:
            self.load_models()
        
        # Step 1: Generate user embedding
        user_embedding = self.get_user_embedding(user_profile)
        
        # Step 2: Semantic search for candidates
        distances, candidate_indices = self.semantic_search(user_embedding, num_candidates)
        
        # Step 3: Rerank candidates
        rerank_order = self.rerank_candidates(user_profile, candidate_indices)
        
        # Step 4: Generate final recommendations
        recommendations = []
        
        for i in range(min(num_recommendations, len(rerank_order))):
            rerank_idx = rerank_order[i]
            job_idx = candidate_indices[rerank_idx]
            job = self.jobs_df.iloc[job_idx]
            
            # Calculate compatibility scores
            user_row = self._profile_to_series(user_profile)
            features = create_user_job_features(user_row, job)
            
            recommendation = {
                'job_id': job['job_id'],
                'job_title': job['job_title'],
                'company': job['company'],
                'industry': job['industry'],
                'location': job['location'],
                'description': job['description'][:200] + "..." if len(job['description']) > 200 else job['description'],
                'required_skills': job['required_skills'],
                'education_requirement': job['education_requirement'],
                'experience_requirement': job['experience_requirement'],
                'salary_range': job['salary_range'],
                'semantic_similarity': float(1 / (1 + distances[rerank_idx])),  # Convert distance to similarity
                'compatibility_scores': {
                    'skill_overlap': features['skill_overlap'],
                    'education_match': features['education_level_match'],
                    'experience_match': features['experience_match'],
                    'location_match': features['location_match']
                },
                'overall_score': float(self.reranker_model.predict_proba(pd.DataFrame([features]))[0] if self.reranker_model else 0.5)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def explain_recommendation(self, user_profile: Dict[str, Any], job_recommendation: Dict[str, Any]) -> str:
        """Generate explanation for why a job was recommended"""
        explanation_parts = []
        
        scores = job_recommendation['compatibility_scores']
        
        if scores['skill_overlap'] > 0.2:
            explanation_parts.append(f"Strong skill match ({scores['skill_overlap']:.1%} overlap)")
        
        if scores['education_match']:
            explanation_parts.append("Education requirements met")
        
        if scores['experience_match']:
            explanation_parts.append("Experience requirements satisfied")
        
        if scores['location_match']:
            explanation_parts.append("Location preference matched")
        
        if job_recommendation['semantic_similarity'] > 0.7:
            explanation_parts.append("High semantic similarity to your profile")
        
        if not explanation_parts:
            explanation_parts.append("Good overall compatibility based on profile analysis")
        
        return "Recommended because: " + ", ".join(explanation_parts)
    
    def batch_recommend(self, user_profiles: List[Dict[str, Any]], 
                       num_recommendations: int = 10) -> List[List[Dict[str, Any]]]:
        """Generate recommendations for multiple users"""
        results = []
        
        for i, profile in enumerate(user_profiles):
            print(f"Processing user {i+1}/{len(user_profiles)}")
            recommendations = self.recommend_jobs(profile, num_recommendations=num_recommendations)
            results.append(recommendations)
        
        return results
    
    def save_recommendations_csv(self, recommendations: List[Dict[str, Any]], 
                               filename: str, user_id: str = None):
        """Save recommendations to CSV file"""
        # Flatten recommendations for CSV
        flattened = []
        
        for rec in recommendations:
            row = {
                'user_id': user_id,
                'job_id': rec['job_id'],
                'job_title': rec['job_title'],
                'company': rec['company'],
                'industry': rec['industry'],
                'location': rec['location'],
                'salary_range': rec['salary_range'],
                'overall_score': rec['overall_score'],
                'semantic_similarity': rec['semantic_similarity'],
                'skill_overlap': rec['compatibility_scores']['skill_overlap'],
                'education_match': rec['compatibility_scores']['education_match'],
                'experience_match': rec['compatibility_scores']['experience_match'],
                'location_match': rec['compatibility_scores']['location_match']
            }
            flattened.append(row)
        
        df = pd.DataFrame(flattened)
        df.to_csv(filename, index=False)
        print(f"Recommendations saved to {filename}")
    
    def _profile_to_series(self, user_profile: Dict[str, Any]) -> pd.Series:
        """Convert user profile dictionary to pandas Series with required fields"""
        # Map education levels
        education_mapping = self.metadata.get('education_mapping', {})
        
        profile_data = {
            'gpa_normalized': user_profile.get('gpa', 3.0) / 4.0,
            'experience_years': user_profile.get('experience_years', 0),
            'education_level_encoded': education_mapping.get(user_profile.get('education_level', 'Bachelor'), 2),
            'skills_list': user_profile.get('skills', '').split(',') if isinstance(user_profile.get('skills', ''), str) else user_profile.get('skills', []),
            'preferred_location': user_profile.get('preferred_location', ''),
            'field_of_study': user_profile.get('field_of_study', ''),
            'interests': user_profile.get('interests', '')
        }
        
        return pd.Series(profile_data)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models and data"""
        if not self.is_loaded:
            return {"error": "Models not loaded"}
        
        stats = {
            'embedding_model': self.metadata['embedding_model'],
            'embedding_dimension': self.metadata['embedding_dim'],
            'num_users': len(self.users_df),
            'num_jobs': len(self.jobs_df),
            'has_reranker': self.reranker_model is not None,
            'industries': self.jobs_df['industry'].value_counts().to_dict(),
            'education_levels': self.jobs_df['education_requirement'].value_counts().to_dict()
        }
        
        return stats