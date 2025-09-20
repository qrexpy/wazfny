# Data Directory

This directory contains the datasets used by the career recommender system.

## Files

### Sample Data (included in repository)
- `sample_users.csv` - Sample user profiles with education, interests, and skills
- `sample_jobs.csv` - Sample job catalog with descriptions and requirements

### Data Format

#### Users CSV Format
- `user_id`: Unique identifier
- `education_level`: Bachelor, Master, PhD, Associate, High School
- `field_of_study`: Academic field or major
- `gpa`: Grade Point Average (0.0-4.0 scale)
- `interests`: Comma-separated list of interests
- `skills`: Comma-separated list of technical skills
- `experience_years`: Years of work experience
- `preferred_location`: Preferred work location

#### Jobs CSV Format
- `job_id`: Unique identifier
- `job_title`: Job position title
- `company`: Company name
- `industry`: Industry sector
- `description`: Detailed job description
- `required_skills`: Comma-separated list of required skills
- `education_requirement`: Minimum education level
- `experience_requirement`: Required years of experience
- `salary_range`: Salary range (format: min-max)
- `location`: Job location

## Usage

Place your own datasets in this directory following the same format. The system will automatically detect and use any CSV files matching the expected schema.

For larger datasets, consider using the data fetching modules in `src/data_fetch.py` to pull real-time job data from APIs.