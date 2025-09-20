# Career Recommender System

A comprehensive AI-powered career recommendation system that matches users with relevant job opportunities based on their education, GPA, interests, and skills. The system combines semantic similarity search with learned reranking for accurate and personalized job recommendations.

## ✨ Features

- **Semantic Job Matching**: Uses sentence-transformers to find semantically similar jobs
- **Intelligent Reranking**: XGBoost model reranks candidates based on structured features
- **Scalable Vector Search**: FAISS-powered similarity search for efficient retrieval
- **Interactive Notebooks**: Complete pipeline from data preprocessing to inference
- **Flexible Data Sources**: Support for CSV data and external API integration
- **Comprehensive Evaluation**: NDCG, precision@k, and other ranking metrics

## 🏗️ Architecture

```
User Profile → Text Embedding → Semantic Search → Candidate Jobs
                                      ↓
Feature Engineering ← Job Requirements ← FAISS Index
        ↓
XGBoost Reranker → Final Recommendations
```

## 📁 Project Structure

```
career-recommender/
├── notebooks/                 # Jupyter notebooks for pipeline
│   ├── 01_preprocess.ipynb   # Data cleaning and embedding generation
│   ├── 02_train_xgboost.ipynb # Reranker model training
│   ├── 03_evaluate.ipynb    # Model evaluation and metrics
│   └── 04_inference_demo.ipynb # Interactive recommendation demo
├── src/                      # Reusable Python modules
│   ├── __init__.py
│   ├── feature_engineering.py # Feature creation utilities
│   ├── model_training.py     # XGBoost training and evaluation
│   ├── data_fetch.py        # External API data collection
│   └── recommender.py       # Main recommendation system
├── data/                     # Dataset storage
│   ├── sample_users.csv     # Sample user profiles
│   ├── sample_jobs.csv      # Sample job catalog
│   └── README.md
├── models/                   # Trained model artifacts
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd career-recommender

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.template .env
# Edit .env with your API keys
```

### 2. Run the Pipeline

#### Step 1: Data Preprocessing
```bash
jupyter notebook notebooks/01_preprocess.ipynb
```
This notebook will:
- Load and clean user profiles and job data
- Generate embeddings using sentence-transformers
- Create FAISS index for similarity search
- Save processed data for training

#### Step 2: Train Reranker Model
```bash  
jupyter notebook notebooks/02_train_xgboost.ipynb
```
Trains the XGBoost reranker using features like:
- Skill overlap between user and job
- Education level matching
- Experience requirements compatibility
- GPA and other profile features

#### Step 3: Evaluate System
```bash
jupyter notebook notebooks/03_evaluate.ipynb  
```
Measures performance using:
- NDCG@k (Normalized Discounted Cumulative Gain)
- Precision@k and Recall@k
- AUC and other classification metrics

#### Step 4: Interactive Demo
```bash
jupyter notebook notebooks/04_inference_demo.ipynb
```
Try the system with new user profiles and get recommendations!

### 3. Python API Usage

```python
from src.recommender import CareerRecommender

# Initialize and load models
recommender = CareerRecommender(models_dir="./models")
recommender.load_models()

# Create user profile
user_profile = {
    "education_level": "Bachelor",
    "field_of_study": "Computer Science", 
    "gpa": 3.8,
    "interests": "machine learning, data analysis, programming",
    "skills": "Python, SQL, JavaScript, Git",
    "experience_years": 2,
    "preferred_location": "Remote"
}

# Get recommendations
recommendations = recommender.recommend_jobs(
    user_profile, 
    num_recommendations=10
)

# Display results
for i, job in enumerate(recommendations, 1):
    print(f"{i}. {job['job_title']} at {job['company']}")
    print(f"   Score: {job['overall_score']:.3f}")
    print(f"   Skills: {job['required_skills']}")
    print()
```

## 🔑 API Keys and Configuration

### Required API Keys

The system works with sample data out of the box, but you can enhance it with external APIs:

#### 1. Hugging Face (Optional)
For downloading models and using the Hub:
```bash
export HF_TOKEN="your_huggingface_token"
```

#### 2. SerpAPI (Optional)
For real-time job data from Google Jobs:
```bash
export SERPAPI_KEY="your_serpapi_key"
```

#### 3. Vector Database (Optional)
For cloud vector databases:
```bash
export PINECONE_API_KEY="your_pinecone_key"
export WEAVIATE_URL="your_weaviate_url"
```

### Environment Setup
```bash
# Copy template and edit
cp .env.template .env

# Or set directly
echo "HF_TOKEN=your_token_here" >> .env
```

## 📊 Data Format

### User Profiles (CSV)
```csv
user_id,education_level,field_of_study,gpa,interests,skills,experience_years,preferred_location
1,Bachelor,Computer Science,3.8,"programming,ML","Python,SQL",2,Remote
```

### Job Catalog (CSV) 
```csv
job_id,job_title,company,industry,description,required_skills,education_requirement,experience_requirement,salary_range,location
1,Software Engineer,TechCorp,Technology,"Develop web apps","Python,JavaScript",Bachelor,2-4 years,70000-95000,Remote
```

## ⚙️ Model Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension**: 384
- **Purpose**: Semantic similarity between user profiles and job descriptions

### Reranker Model  
- **Algorithm**: XGBoost Classifier
- **Features**: 9 structured compatibility features
- **Purpose**: Rerank semantic search results using learned preferences

### Vector Database
- **Engine**: FAISS (CPU version)
- **Index Type**: Flat L2 distance
- **Purpose**: Fast similarity search across job embeddings

## 📈 Evaluation Metrics

The system is evaluated using standard ranking metrics:

- **NDCG@k**: Normalized Discounted Cumulative Gain at rank k
- **Precision@k**: Fraction of relevant items in top k results  
- **Recall@k**: Fraction of all relevant items found in top k
- **AUC**: Area under ROC curve for binary relevance classification

## 🔧 Customization

### Adding New Features
Edit `src/feature_engineering.py` to add custom compatibility features:

```python
def create_user_job_features(user_row, job_row):
    features = {}
    # Add your custom features here
    features['custom_score'] = your_scoring_function(user_row, job_row)
    return features
```

### Using Different Models
Replace the embedding model in preprocessing:

```python
# In 01_preprocess.ipynb
model_name = "your-preferred-model"  # e.g., "all-mpnet-base-v2"
embedding_model = SentenceTransformer(model_name)
```

### External Data Sources
Use `src/data_fetch.py` to integrate with job APIs:

```python
from src.data_fetch import JobDataFetcher, collect_job_data

# Collect real job data
queries = ["software engineer", "data scientist"]
jobs_df = collect_job_data(queries, num_results_per_query=100)
```

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **FAISS Installation Issues**: Try CPU version
   ```bash
   pip install faiss-cpu
   ```

3. **Memory Issues**: Reduce batch sizes in notebooks or use smaller models

4. **API Rate Limits**: Add delays between API calls in `data_fetch.py`

### Performance Optimization

1. **Faster Embeddings**: Use smaller models like `all-MiniLM-L6-v2`
2. **GPU Acceleration**: Install `faiss-gpu` for large datasets  
3. **Caching**: Save embeddings to avoid recomputation
4. **Batch Processing**: Process multiple users simultaneously

## 📝 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v
```

### Code Style
```bash
# Install formatting tools
pip install black flake8

# Format code
black src/ notebooks/
flake8 src/
```

### Adding Features
1. Implement feature in appropriate `src/` module
2. Add tests in `tests/` directory  
3. Update documentation and examples
4. Test with sample data

## 📚 References

- [Sentence Transformers](https://www.sbert.net/) - Text embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: Use GitHub Issues for bug reports and features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainers for private inquiries

---

**Built for career counselors, educators, and job seekers.** 🎯