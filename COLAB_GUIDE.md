# 🚀 Career Recommender System - Colab/Kaggle Guide

Complete tutorial for running the career recommender system in Google Colab, Kaggle, or other Jupyter-like environments.

## 📋 Quick Start Checklist

- [ ] Upload notebooks to your environment
- [ ] Run pip install commands
- [ ] Either upload data files OR use sample data generation
- [ ] Follow the notebook sequence: 01 → 02 → 03 → 04

## 🗂️ File Overview

### Required Notebooks (run in order):
1. **`01_preprocess.ipynb`** - Data preprocessing and embedding generation
2. **`02_train_xgboost.ipynb`** - Train reranker model  
3. **`03_evaluate.ipynb`** - Evaluate system performance
4. **`04_inference_demo.ipynb`** - Interactive recommendations

### Data Files (optional - will generate if missing):
- `sample_users.csv` - User profiles
- `sample_jobs.csv` - Job catalog

## 🎯 Method 1: Google Colab (Recommended)

### Step 1: Setup Environment
```python
# First cell in any notebook
!pip install pandas numpy scikit-learn matplotlib seaborn plotly
!pip install sentence-transformers transformers torch
!pip install faiss-cpu tqdm python-dotenv huggingface-hub
!pip install xgboost lightgbm ipywidgets
```

### Step 2: Upload Files (Optional)
If you have the data files:
1. Click the folder icon in Colab sidebar
2. Upload `sample_users.csv` and `sample_jobs.csv`
3. Or create a `data/` folder and upload there

**OR** just run the notebooks - they'll generate sample data automatically!

### Step 3: Run Notebooks in Order

#### 🔹 Notebook 1: Preprocessing (`01_preprocess.ipynb`)
```python
# The notebook will:
# ✅ Install dependencies
# ✅ Load or generate sample data  
# ✅ Clean and process user profiles and jobs
# ✅ Generate embeddings using sentence-transformers
# ✅ Create FAISS vector index
# ✅ Save processed data

# Expected runtime: 3-5 minutes
# Output: Processed data files in models/ directory
```

#### 🔹 Notebook 2: Training (`02_train_xgboost.ipynb`)  
```python
# The notebook will:
# ✅ Load preprocessed features or generate sample training data
# ✅ Train XGBoost reranker model
# ✅ Evaluate model performance (AUC, precision, recall)
# ✅ Show feature importance analysis
# ✅ Save trained model

# Expected runtime: 2-3 minutes  
# Output: Trained reranker model
```

#### 🔹 Notebook 3: Evaluation (`03_evaluate.ipynb`)
```python
# The notebook will:
# ✅ Load trained models
# ✅ Evaluate recommendation quality with NDCG@k
# ✅ Compare embedding-only vs full pipeline
# ✅ Generate performance reports

# Expected runtime: 1-2 minutes
```

#### 🔹 Notebook 4: Demo (`04_inference_demo.ipynb`)
```python
# The notebook will: 
# ✅ Create interactive recommendation interface
# ✅ Test with sample user profiles
# ✅ Show job recommendations with explanations
# ✅ Allow custom user input

# Expected runtime: 1 minute
# Output: Interactive widgets for testing
```

## 🎯 Method 2: Kaggle Notebooks

### Setup in Kaggle:
1. Go to Kaggle.com → Notebooks → New Notebook
2. Choose "Notebook" (not Script)
3. Enable Internet access in settings (for downloading models)
4. Copy-paste notebook code or upload .ipynb files

### Kaggle-Specific Notes:
```python
# Kaggle has most packages pre-installed, but run this to be sure:
!pip install sentence-transformers faiss-cpu huggingface-hub

# Check GPU availability (optional, for faster training):
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🎯 Method 3: Local Jupyter + Upload to Cloud

### Local Setup:
```bash
# Install Jupyter if needed
pip install jupyter

# Install all dependencies  
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install sentence-transformers transformers torch
pip install faiss-cpu tqdm python-dotenv huggingface-hub
pip install xgboost lightgbm ipywidgets

# Start Jupyter
jupyter notebook
```

### Then Upload to Cloud:
- Download notebooks as .ipynb files
- Upload to Google Drive → Open with Colab
- Or upload to Kaggle Datasets → Create Notebook

## 🔧 Troubleshooting Common Issues

### Issue 1: Package Installation Errors
```python
# If you get installation errors, try:
!pip install --upgrade pip
!pip install torch --index-url https://download.pytorch.org/whl/cpu
!pip install sentence-transformers --no-deps
!pip install transformers tokenizers huggingface-hub
```

### Issue 2: Memory Issues in Colab
```python
# If you run out of memory:
# 1. Use smaller embedding models
model_name = "all-MiniLM-L6-v2"  # Instead of larger models

# 2. Process data in smaller batches
batch_size = 32  # Reduce if needed

# 3. Clear variables between cells
import gc
del large_variable
gc.collect()
```

### Issue 3: File Not Found Errors
```python
# The notebooks are designed to work without uploaded files
# They will generate sample data automatically
# But if you want to use your own data, ensure files are in the right location:

import os
print("Current directory:", os.getcwd())
print("Files available:", os.listdir('.'))

# Create data directory if needed
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
```

### Issue 4: Slow Model Download
```python
# If sentence-transformers is slow to download:
# 1. Use Colab Pro for faster downloads
# 2. Or download once and save to Google Drive

from sentence_transformers import SentenceTransformer
import torch

# Download and save model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/content/drive/MyDrive/sentence_transformer_model')

# Later, load from saved location
model = SentenceTransformer('/content/drive/MyDrive/sentence_transformer_model')
```

## 📊 Expected Outputs

### After Notebook 1 (Preprocessing):
```
✅ Environment setup complete!
✅ Dataset loaded successfully!
✅ User profiles preprocessed!
✅ Job catalog preprocessed!
✅ Generated 20 user embeddings with dimension 384
✅ Generated 30 job embeddings with dimension 384
✅ FAISS index created with 30 job embeddings
✅ All processed data saved!
```

### After Notebook 2 (Training):
```
✅ Model training completed!
📊 Model Performance:
AUC: 0.8234
Precision: 0.7456
Recall: 0.6789
F1-Score: 0.7103
✅ Model saved successfully!
```

### After Notebook 3 (Evaluation):
```
📈 Recommendation System Evaluation:
NDCG@5: 0.7832
NDCG@10: 0.8156
Precision@5: 0.6400
Precision@10: 0.5800
```

### After Notebook 4 (Demo):
```
🎯 Career Recommendations for User:
1. Software Engineer at TechCorp (Score: 0.923)
2. Data Scientist at DataInc (Score: 0.887)
3. Product Manager at InnovateCo (Score: 0.845)
...
```

## 🚀 Advanced Usage

### Custom Data Format
If you want to use your own data, format it as CSV with these columns:

**Users CSV:**
```csv
user_id,education_level,field_of_study,gpa,interests,skills,experience_years,preferred_location
1,Bachelor,Computer Science,3.8,"programming,ML","Python,SQL",2,Remote
```

**Jobs CSV:**
```csv
job_id,job_title,company,industry,description,required_skills,education_requirement,experience_requirement,salary_range,location
1,Software Engineer,TechCorp,Technology,"Develop apps","Python,JS",Bachelor,2-4 years,70k-95k,Remote
```

### API Integration
To fetch real job data, set up API keys:
```python
import os
os.environ['SERPAPI_KEY'] = 'your_serpapi_key_here'
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
```

### Scaling Up
For larger datasets:
```python
# Use GPU acceleration (if available)
!pip install faiss-gpu  # Instead of faiss-cpu

# Use larger, more accurate models
model_name = "all-mpnet-base-v2"  # Better but slower

# Batch processing for large datasets
batch_size = 1000  # Process in chunks
```

## 🤝 Getting Help

### Common Questions:

**Q: Can I run this without uploading data files?**  
A: Yes! The notebooks generate sample data automatically.

**Q: How long does it take to run everything?**  
A: About 10-15 minutes total in Colab (most time is downloading models).

**Q: Can I use this for real job recommendations?**  
A: Yes! Replace sample data with real user profiles and job postings.

**Q: Do I need paid Colab/Kaggle?**  
A: No, free tier works fine. Paid tier is faster for large datasets.

### Still Need Help?

1. **Check the notebook outputs** - they show progress and errors
2. **Read error messages carefully** - usually indicate missing packages or files  
3. **Try restarting the runtime** - fixes many environment issues
4. **Use smaller data/models** if you hit memory limits

---

## 🎉 You're Ready!

Start with **`01_preprocess.ipynb`** and follow the sequence. Each notebook is self-contained and includes detailed explanations.

**Happy recommending! 🚀**