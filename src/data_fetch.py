"""
Data fetching utilities for external job APIs and data sources.
"""

import os
import requests
import pandas as pd
import time
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta


class JobDataFetcher:
    """Fetch job data from various external APIs"""
    
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.indeed_api_key = os.getenv("INDEED_API_KEY")
    
    def fetch_google_jobs(self, query: str, location: str = "", num_results: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch job postings from Google Jobs via SerpAPI
        
        Args:
            query: Job search query (e.g., "software engineer")
            location: Location filter (e.g., "San Francisco, CA")
            num_results: Number of results to fetch
        
        Returns:
            List of job dictionaries
        """
        if not self.serpapi_key:
            print("Warning: SERPAPI_KEY not found. Returning empty list.")
            return []
        
        jobs = []
        start = 0
        
        while len(jobs) < num_results:
            params = {
                "engine": "google_jobs",
                "q": query,
                "api_key": self.serpapi_key,
                "start": start
            }
            
            if location:
                params["location"] = location
            
            try:
                response = requests.get("https://serpapi.com/search", params=params)
                response.raise_for_status()
                data = response.json()
                
                if "jobs_results" in data:
                    for job in data["jobs_results"]:
                        job_data = {
                            "job_id": job.get("job_id", ""),
                            "job_title": job.get("title", ""),
                            "company": job.get("company_name", ""),
                            "location": job.get("location", ""),
                            "description": job.get("description", ""),
                            "salary_range": self._extract_salary(job),
                            "posted_date": job.get("detected_extensions", {}).get("posted_at", ""),
                            "source": "google_jobs",
                            "scraped_at": datetime.now().isoformat()
                        }
                        jobs.append(job_data)
                        
                        if len(jobs) >= num_results:
                            break
                
                start += 10
                time.sleep(1)  # Rate limiting
                
                # Break if no more results
                if "jobs_results" not in data or len(data["jobs_results"]) == 0:
                    break
                    
            except requests.RequestException as e:
                print(f"Error fetching jobs: {e}")
                break
        
        return jobs[:num_results]
    
    def fetch_indeed_jobs(self, query: str, location: str = "", num_results: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch job postings from Indeed API (if available)
        Note: Indeed API is not publicly available anymore, this is a placeholder
        """
        print("Warning: Indeed API is not publicly available. Consider web scraping alternatives.")
        return []
    
    def _extract_salary(self, job_data: Dict[str, Any]) -> str:
        """Extract salary information from job data"""
        # Check various fields where salary might be stored
        extensions = job_data.get("detected_extensions", {})
        
        if "salary" in extensions:
            return extensions["salary"]
        
        # Look in description for salary patterns
        description = job_data.get("description", "")
        salary_patterns = [
            r'\$[\d,]+\s*-\s*\$[\d,]+',
            r'\$[\d,]+k?\s*-\s*\$?[\d,]+k?',
            r'[\d,]+k?\s*-\s*[\d,]+k?\s*per\s*year'
        ]
        
        import re
        for pattern in salary_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    def save_jobs_to_csv(self, jobs: List[Dict[str, Any]], filename: str):
        """Save job data to CSV file"""
        if not jobs:
            print("No jobs to save")
            return
        
        df = pd.DataFrame(jobs)
        df.to_csv(filename, index=False)
        print(f"Saved {len(jobs)} jobs to {filename}")
    
    def enrich_job_data(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich job data with additional features"""
        enriched_df = jobs_df.copy()
        
        # Add industry classification based on job title and description
        enriched_df['industry'] = enriched_df.apply(self._classify_industry, axis=1)
        
        # Extract required skills from description
        enriched_df['required_skills'] = enriched_df['description'].apply(self._extract_skills)
        
        # Classify education requirements
        enriched_df['education_requirement'] = enriched_df['description'].apply(self._extract_education_requirement)
        
        # Extract experience requirements
        enriched_df['experience_requirement'] = enriched_df['description'].apply(self._extract_experience_requirement)
        
        return enriched_df
    
    def _classify_industry(self, job_row: pd.Series) -> str:
        """Classify job into industry based on title and description"""
        text = (str(job_row.get('job_title', '')) + ' ' + 
                str(job_row.get('description', ''))).lower()
        
        industry_keywords = {
            'Technology': ['software', 'engineer', 'developer', 'programming', 'tech', 'IT', 'data scientist'],
            'Healthcare': ['medical', 'healthcare', 'nurse', 'doctor', 'clinical', 'pharmaceutical'],
            'Finance': ['finance', 'banking', 'investment', 'accounting', 'financial'],
            'Marketing': ['marketing', 'advertising', 'social media', 'brand', 'campaign'],
            'Education': ['teacher', 'education', 'academic', 'university', 'school'],
            'Sales': ['sales', 'business development', 'account manager', 'customer'],
            'Design': ['design', 'creative', 'graphic', 'UI', 'UX'],
            'Manufacturing': ['manufacturing', 'production', 'factory', 'assembly'],
            'Consulting': ['consulting', 'consultant', 'advisory', 'strategy']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                return industry
        
        return 'Other'
    
    def _extract_skills(self, description: str) -> str:
        """Extract technical skills from job description"""
        if pd.isna(description):
            return ""
        
        # Common technical skills to look for
        skills_list = [
            'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'R', 'MATLAB',
            'Excel', 'PowerPoint', 'Tableau', 'Power BI', 'Salesforce',
            'AWS', 'Azure', 'Docker', 'Kubernetes', 'Git', 'Linux',
            'Machine Learning', 'Data Analysis', 'Statistics', 'AI',
            'React', 'Angular', 'Node.js', 'HTML', 'CSS', 'PHP',
            'Photoshop', 'Illustrator', 'Figma', 'Sketch'
        ]
        
        found_skills = []
        text_lower = description.lower()
        
        for skill in skills_list:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return ', '.join(found_skills)
    
    def _extract_education_requirement(self, description: str) -> str:
        """Extract education requirements from job description"""
        if pd.isna(description):
            return "High School"
        
        text_lower = description.lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate']):
            return "PhD"
        elif any(term in text_lower for term in ['master', 'mba', 'ms', 'm.s']):
            return "Master"
        elif any(term in text_lower for term in ['bachelor', 'ba', 'bs', 'b.s', 'b.a', 'degree']):
            return "Bachelor"
        elif any(term in text_lower for term in ['associate', 'aa', 'as']):
            return "Associate"
        else:
            return "High School"
    
    def _extract_experience_requirement(self, description: str) -> str:
        """Extract experience requirements from job description"""
        if pd.isna(description):
            return "0-1 years"
        
        import re
        
        # Look for experience patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*of?\s*experience',
            r'(\d+)\s*to\s*(\d+)\s*years?\s*experience',
            r'(\d+)-(\d+)\s*years?\s*experience'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description.lower())
            if match:
                return match.group(0)
        
        # Default based on common terms
        text_lower = description.lower()
        if 'entry level' in text_lower or 'junior' in text_lower:
            return "0-2 years"
        elif 'senior' in text_lower:
            return "5+ years"
        elif 'lead' in text_lower or 'principal' in text_lower:
            return "7+ years"
        else:
            return "2-5 years"


# Utility functions for data collection
def collect_job_data(queries: List[str], locations: List[str] = [""], 
                    num_results_per_query: int = 50) -> pd.DataFrame:
    """
    Collect job data for multiple queries and locations
    
    Args:
        queries: List of job search queries
        locations: List of locations to search in
        num_results_per_query: Number of results per query
    
    Returns:
        DataFrame with collected job data
    """
    fetcher = JobDataFetcher()
    all_jobs = []
    
    for query in queries:
        for location in locations:
            print(f"Fetching jobs for '{query}' in '{location}'...")
            jobs = fetcher.fetch_google_jobs(query, location, num_results_per_query)
            all_jobs.extend(jobs)
            time.sleep(2)  # Rate limiting
    
    if all_jobs:
        df = pd.DataFrame(all_jobs)
        # Remove duplicates based on job_title and company
        df = df.drop_duplicates(subset=['job_title', 'company'], keep='first')
        
        # Enrich the data
        df = fetcher.enrich_job_data(df)
        return df
    else:
        return pd.DataFrame()


def get_sample_queries() -> List[str]:
    """Get sample job search queries for data collection"""
    return [
        "software engineer",
        "data scientist", 
        "product manager",
        "marketing specialist",
        "financial analyst",
        "graphic designer",
        "business analyst",
        "sales representative",
        "project manager",
        "web developer"
    ]