"""
Data loading utilities for real job datasets.
Supports loading from Hugging Face datasets including lukebarousse/data_jobs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


class RealJobDataLoader:
    """Load and preprocess real job datasets"""
    
    def __init__(self):
        self.datasets_available = DATASETS_AVAILABLE
    
    def load_lukebarousse_dataset(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load the lukebarousse/data_jobs dataset (785K job postings)
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            DataFrame with job postings
        """
        if not self.datasets_available:
            print("❌ datasets library not available. Install with: pip install datasets")
            return pd.DataFrame()
        
        try:
            print("📥 Loading lukebarousse/data_jobs dataset...")
            
            # Load the dataset
            dataset = load_dataset("lukebarousse/data_jobs")
            
            # Convert to pandas DataFrame
            df = dataset['train'].to_pandas()
            
            if max_samples and len(df) > max_samples:
                print(f"📊 Sampling {max_samples} from {len(df)} total jobs")
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} job postings")
            print(f"📊 Dataset columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return pd.DataFrame()
    
    def preprocess_lukebarousse_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the lukebarousse dataset to match our schema
        
        Args:
            df: Raw dataset DataFrame
        
        Returns:
            Preprocessed DataFrame matching our job schema
        """
        if df.empty:
            return df
        
        print("🔧 Preprocessing job data...")
        
        # Create standardized job DataFrame
        processed_df = pd.DataFrame()
        
        # Map columns to our standard schema
        column_mapping = {
            'job_id': 'job_id',
            'job_title': 'job_title', 
            'company_name': 'company',
            'job_location': 'location',
            'job_posted_date': 'posted_date',
            'job_schedule_type': 'schedule_type',
            'job_work_from_home': 'remote',
            'search_location': 'search_location',
            'job_via': 'source',
            'job_description': 'description',
            'job_is_remote': 'is_remote',
            'salary_rate': 'salary_rate',
            'salary_year_avg': 'salary_avg',
            'salary_hour_avg': 'salary_hour_avg',
            'company_num_employees': 'company_size',
            'job_skills': 'required_skills',
            'job_type_skills': 'job_type_skills'
        }
        
        # Map existing columns
        for original_col, new_col in column_mapping.items():
            if original_col in df.columns:
                processed_df[new_col] = df[original_col]
        
        # Handle missing required columns
        if 'job_id' not in processed_df.columns:
            processed_df['job_id'] = range(len(processed_df))
        
        # Standardize salary information
        if 'salary_avg' in processed_df.columns:
            processed_df['salary_range'] = processed_df['salary_avg'].apply(self._format_salary_range)
        else:
            processed_df['salary_range'] = ""
        
        # Extract industry from job skills or title
        processed_df['industry'] = processed_df.apply(self._classify_industry_from_skills, axis=1)
        
        # Standardize education requirements (extract from description)
        if 'description' in processed_df.columns:
            processed_df['education_requirement'] = processed_df['description'].apply(self._extract_education_requirement)
            processed_df['experience_requirement'] = processed_df['description'].apply(self._extract_experience_requirement)
        else:
            processed_df['education_requirement'] = "Bachelor"
            processed_df['experience_requirement'] = "2-5 years"
        
        # Clean and standardize text fields
        text_columns = ['job_title', 'description', 'company', 'location']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str).fillna("")
        
        # Handle skills data - convert from list format if needed
        if 'required_skills' in processed_df.columns:
            processed_df['required_skills'] = processed_df['required_skills'].apply(self._format_skills_list)
        
        print(f"✅ Preprocessed {len(processed_df)} job postings")
        print(f"📊 Industries found: {processed_df['industry'].value_counts().head()}")
        
        return processed_df
    
    def _format_salary_range(self, salary_avg) -> str:
        """Format salary average into a range string"""
        if pd.isna(salary_avg) or salary_avg == 0:
            return ""
        
        # Estimate range as +/- 15% of average
        salary_avg = float(salary_avg)
        min_salary = int(salary_avg * 0.85)
        max_salary = int(salary_avg * 1.15)
        
        return f"{min_salary}-{max_salary}"
    
    def _format_skills_list(self, skills) -> str:
        """Format skills from various formats into comma-separated string"""
        if pd.isna(skills):
            return ""
        
        if isinstance(skills, list):
            return ", ".join([str(s) for s in skills if s])
        elif isinstance(skills, str):
            # Already a string, clean it up
            return skills.strip()
        else:
            return str(skills)
    
    def _classify_industry_from_skills(self, row) -> str:
        """Classify industry based on job skills and title"""
        # Get text to analyze
        title = str(row.get('job_title', '')).lower()
        skills = str(row.get('required_skills', '')).lower()
        job_type_skills = str(row.get('job_type_skills', '')).lower()
        
        text = f"{title} {skills} {job_type_skills}"
        
        industry_keywords = {
            'Technology': [
                'software', 'engineer', 'developer', 'programming', 'tech', 'IT', 
                'data scientist', 'python', 'java', 'javascript', 'react', 'sql',
                'machine learning', 'ai', 'artificial intelligence', 'devops',
                'cloud', 'aws', 'azure', 'kubernetes', 'docker'
            ],
            'Healthcare': [
                'medical', 'healthcare', 'nurse', 'doctor', 'clinical', 
                'pharmaceutical', 'hospital', 'patient', 'therapy'
            ],
            'Finance': [
                'finance', 'banking', 'investment', 'accounting', 'financial',
                'analyst', 'trading', 'risk', 'compliance', 'audit'
            ],
            'Marketing': [
                'marketing', 'advertising', 'social media', 'brand', 'campaign',
                'content', 'seo', 'digital marketing', 'growth'
            ],
            'Education': [
                'teacher', 'education', 'academic', 'university', 'school',
                'training', 'curriculum', 'learning'
            ],
            'Sales': [
                'sales', 'business development', 'account manager', 'customer',
                'client', 'revenue', 'quota', 'crm'
            ],
            'Design': [
                'design', 'creative', 'graphic', 'UI', 'UX', 'visual',
                'photoshop', 'figma', 'branding'
            ],
            'Manufacturing': [
                'manufacturing', 'production', 'factory', 'assembly',
                'quality', 'operations', 'supply chain'
            ],
            'Consulting': [
                'consulting', 'consultant', 'advisory', 'strategy',
                'transformation', 'process improvement'
            ]
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                return industry
        
        return 'Other'
    
    def _extract_education_requirement(self, description: str) -> str:
        """Extract education requirements from job description"""
        if pd.isna(description):
            return "Bachelor"
        
        text_lower = str(description).lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate']):
            return "PhD"
        elif any(term in text_lower for term in ['master', 'mba', 'ms', 'm.s']):
            return "Master"
        elif any(term in text_lower for term in ['bachelor', 'ba', 'bs', 'b.s', 'b.a', 'degree']):
            return "Bachelor"
        elif any(term in text_lower for term in ['associate', 'aa', 'as']):
            return "Associate"
        else:
            return "Bachelor"  # Default assumption
    
    def _extract_experience_requirement(self, description: str) -> str:
        """Extract experience requirements from job description"""
        if pd.isna(description):
            return "2-5 years"
        
        import re
        
        # Look for experience patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*of?\s*experience',
            r'(\d+)\s*to\s*(\d+)\s*years?\s*experience',
            r'(\d+)-(\d+)\s*years?\s*experience'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(description).lower())
            if match:
                return match.group(0)
        
        # Default based on common terms
        text_lower = str(description).lower()
        if 'entry level' in text_lower or 'junior' in text_lower:
            return "0-2 years"
        elif 'senior' in text_lower:
            return "5+ years"
        elif 'lead' in text_lower or 'principal' in text_lower:
            return "7+ years"
        else:
            return "2-5 years"
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for testing when real data is not available
        
        Args:
            n_samples: Number of sample jobs to create
        
        Returns:
            DataFrame with sample job data
        """
        print(f"🔧 Creating sample dataset with {n_samples} jobs...")
        
        # Sample job titles by industry
        job_titles_by_industry = {
            'Technology': [
                'Software Engineer', 'Data Scientist', 'Frontend Developer', 
                'Backend Developer', 'DevOps Engineer', 'Machine Learning Engineer',
                'Product Manager', 'Technical Lead', 'Software Architect',
                'Full Stack Developer', 'Data Engineer', 'Cloud Engineer'
            ],
            'Healthcare': [
                'Registered Nurse', 'Medical Doctor', 'Physician Assistant',
                'Medical Technologist', 'Healthcare Administrator', 'Pharmacist',
                'Physical Therapist', 'Clinical Research Coordinator'
            ],
            'Finance': [
                'Financial Analyst', 'Investment Banker', 'Risk Analyst',
                'Accounting Manager', 'Financial Advisor', 'Credit Analyst',
                'Portfolio Manager', 'Compliance Officer'
            ],
            'Marketing': [
                'Marketing Manager', 'Digital Marketing Specialist', 'Content Writer',
                'SEO Specialist', 'Brand Manager', 'Social Media Manager',
                'Growth Marketing Manager', 'Marketing Analyst'
            ],
            'Education': [
                'Teacher', 'Professor', 'Academic Advisor', 'Curriculum Developer',
                'Education Administrator', 'Research Associate', 'Tutor'
            ]
        }
        
        companies = [
            'TechCorp', 'InnovateInc', 'DataSolutions', 'CloudSystems', 'AICompany',
            'HealthcarePlus', 'MedTech', 'CareProviders', 'FinanceGroup', 'BankCorp',
            'InvestmentFirm', 'MarketingAgency', 'BrandStudio', 'EduTech', 'LearnCorp'
        ]
        
        locations = [
            'San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX',
            'Boston, MA', 'Chicago, IL', 'Los Angeles, CA', 'Denver, CO',
            'Atlanta, GA', 'Portland, OR', 'Remote', 'Miami, FL'
        ]
        
        # Generate sample data
        np.random.seed(42)
        sample_data = []
        
        for i in range(n_samples):
            # Choose industry
            industry = np.random.choice(list(job_titles_by_industry.keys()))
            job_title = np.random.choice(job_titles_by_industry[industry])
            
            # Generate other fields
            company = np.random.choice(companies)
            location = np.random.choice(locations)
            
            # Generate realistic salary based on job level
            if 'senior' in job_title.lower() or 'lead' in job_title.lower():
                salary_avg = np.random.randint(90000, 150000)
                experience = np.random.choice(['5-7 years', '7+ years', '5+ years'])
            elif 'manager' in job_title.lower() or 'director' in job_title.lower():
                salary_avg = np.random.randint(100000, 180000)
                experience = np.random.choice(['7+ years', '5-10 years', '8+ years'])
            else:
                salary_avg = np.random.randint(50000, 120000)
                experience = np.random.choice(['1-3 years', '2-5 years', '0-2 years'])
            
            education = np.random.choice(['Bachelor', 'Master', 'Associate', 'PhD'], 
                                       p=[0.6, 0.25, 0.1, 0.05])
            
            sample_data.append({
                'job_id': i + 1,
                'job_title': job_title,
                'company': company,
                'industry': industry,
                'location': location,
                'description': f"Join our team as a {job_title} at {company}. We are looking for someone with experience in {industry.lower()}.",
                'required_skills': self._generate_skills_for_industry(industry),
                'education_requirement': education,
                'experience_requirement': experience,
                'salary_range': f"{int(salary_avg * 0.85)}-{int(salary_avg * 1.15)}",
                'salary_avg': salary_avg,
                'source': 'sample_data',
                'posted_date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            })
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample dataset with {len(df)} jobs")
        return df
    
    def _generate_skills_for_industry(self, industry: str) -> str:
        """Generate realistic skills for each industry"""
        skills_by_industry = {
            'Technology': ['Python', 'JavaScript', 'SQL', 'React', 'Git', 'AWS', 'Docker'],
            'Healthcare': ['Patient Care', 'Medical Terminology', 'HIPAA', 'Clinical Skills'],
            'Finance': ['Excel', 'Financial Modeling', 'Bloomberg', 'Risk Analysis', 'SQL'],
            'Marketing': ['Google Analytics', 'SEO', 'Content Creation', 'Social Media', 'HubSpot'],
            'Education': ['Curriculum Development', 'Teaching', 'Assessment', 'Communication']
        }
        
        skills = skills_by_industry.get(industry, ['Communication', 'Problem Solving'])
        selected_skills = np.random.choice(skills, size=min(len(skills), 4), replace=False)
        return ', '.join(selected_skills)
    
    def save_processed_data(self, df: pd.DataFrame, output_dir: Path):
        """Save processed job data to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for fast loading
        df.to_pickle(output_dir / "jobs_real.pkl")
        
        # Save as CSV for manual inspection
        df.to_csv(output_dir / "jobs_real.csv", index=False)
        
        # Save metadata
        metadata = {
            'num_jobs': len(df),
            'industries': df['industry'].value_counts().to_dict(),
            'locations': df['location'].value_counts().head(10).to_dict(),
            'avg_salary': df['salary_avg'].mean() if 'salary_avg' in df.columns else None,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(output_dir / "real_data_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved processed data to {output_dir}")
        print(f"📊 Files: jobs_real.pkl, jobs_real.csv, real_data_metadata.json")


def load_real_or_sample_data(max_samples: Optional[int] = 10000, 
                           prefer_real: bool = True) -> pd.DataFrame:
    """
    Load real job data if available, otherwise create sample data
    
    Args:
        max_samples: Maximum number of samples to load
        prefer_real: Try to load real data first
    
    Returns:
        DataFrame with job data
    """
    loader = RealJobDataLoader()
    
    if prefer_real and loader.datasets_available:
        try:
            # Try to load real dataset
            df = loader.load_lukebarousse_dataset(max_samples=max_samples)
            if not df.empty:
                df = loader.preprocess_lukebarousse_data(df)
                print(f"✅ Using real job data: {len(df)} jobs")
                return df
        except Exception as e:
            print(f"⚠️ Could not load real data: {e}")
    
    # Fallback to sample data
    print("📝 Falling back to sample data generation...")
    df = loader.create_sample_dataset(max_samples or 1000)
    return df


def demo_real_data_loading():
    """Demo function showing how to load and use real job data"""
    print("=== Real Job Data Loading Demo ===")
    print()
    
    # Load a small sample for demo
    df = load_real_or_sample_data(max_samples=100, prefer_real=True)
    
    if not df.empty:
        print(f"📊 Loaded {len(df)} jobs")
        print(f"🏭 Industries: {', '.join(df['industry'].value_counts().head().index)}")
        print(f"💰 Avg salary range: {df['salary_avg'].mean() if 'salary_avg' in df.columns else 'N/A'}")
        print(f"📍 Top locations: {', '.join(df['location'].value_counts().head(3).index)}")
        
        print("\n📋 Sample job:")
        sample_job = df.iloc[0]
        print(f"  Title: {sample_job['job_title']}")
        print(f"  Company: {sample_job['company']}")
        print(f"  Industry: {sample_job['industry']}")
        print(f"  Skills: {sample_job['required_skills']}")
        print(f"  Location: {sample_job['location']}")
    
    return df