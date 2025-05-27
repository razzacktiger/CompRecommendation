# Property Appraisal Recommendation System

## Project Overview

This project develops machine learning-based systems to recommend comparable properties (comps) for real estate appraisal. The project includes two distinct approaches:

1. **Solution 1 (Original)**: Traditional ML pipeline with feature engineering and classification models
2. **Solution 2 (Advanced)**: Comprehensive analytical scoring system with ML enhancement that mimics professional appraiser decision-making

Both solutions aim to predict which properties from a list of potential comparables would actually be chosen in historical appraisals, but use different methodologies and levels of sophistication.

## 🏗️ Project Architecture

The project follows a dual-solution approach:

- **Solution 1**: Focuses on traditional ML classification with basic feature engineering
- **Solution 2**: Emphasizes analytical scoring, comprehensive feature engineering, and real appraiser ground truth validation

## Project Structure

The project is organized as follows:

```
CompRecommendation/
├── .git/                   # Git version control files
├── .venv/                  # Python virtual environment
├── src/                    # Source code directory
│   ├── Solution1/          # Solution 1: Original ML pipeline
│   │   ├── __init__.py     # Makes src a Python package
│   │   ├── config.py       # Configuration variables (e.g., file paths)
│   │   ├── data_loader.py  # Functions for loading and initial EDA of data
│   │   ├── feature_engineering.py # Functions for creating features from raw data
│   │   ├── geocoding_utils.py  # Utilities for address geocoding and caching
│   │   ├── model_pipeline.py   # Functions for training, evaluating models
│   │   ├── main.py         # Main script to run the full pipeline
│   │   ├── utils.py        # General helper functions
│   │   ├── appraisals_dataset.json # The raw dataset
│   │   ├── geocoding_cache.json  # Cache for geocoded addresses
│   │   ├── comp_model.pkl  # Saved model (e.g., XGBoost) from a previous run (example)
│   │   └── comp_scaler.pkl # Saved scaler from a previous run (example)
│   └── Solution2/          # Solution 2: Advanced analytical + ML system
│       ├── README.md       # Solution 2 specific documentation
│       ├── config.py       # Configuration settings and file paths
│       ├── data/
│       │   └── processed/  # Processed datasets
│       │       ├── subjects_cleaned.csv           # Subject properties (88 properties)
│       │       ├── comps_cleaned_with_subjects.csv # Real appraiser selections (264 comparables)
│       │       ├── properties_deduplicated.csv    # Deduplicated property data
│       │       ├── properties_cleaned_with_subjects.csv # Properties with subject context
│       │       ├── properties_comparison_engineered.csv # Full feature engineering dataset
│       │       └── properties_model_ready.csv     # ML-ready dataset
│       ├── cleaning.py     # Data cleaning utilities and functions
│       ├── improved_duplicate_detection.py # Advanced duplicate detection algorithms
│       ├── cleaning.ipynb  # Data cleaning and preprocessing notebook
│       ├── feature_engineering.ipynb # Feature engineering and scoring notebook
│       └── ml_training_pipeline.ipynb # Machine learning training and evaluation
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
├── requirements.txt        # Python package dependencies
├── README.md               # This file (main project documentation)
├── PLANNING.md             # Project planning document
└── TASK.MD                 # Task tracking document
```

## 🚀 Solution Comparison

### Solution 1: Traditional ML Pipeline
**Focus**: Classic machine learning approach with basic feature engineering

**Key Features**:
- Traditional classification models (XGBoost, LightGBM, Logistic Regression, KNN)
- Basic feature engineering from raw data
- Standard ML evaluation metrics
- Single-script execution pipeline

**Performance**: 
- AUPRC: 0.31-0.33 (XGBoost/LightGBM)
- F1-Score: ~0.36 (after threshold tuning)
- Challenge: Severe class imbalance

**Best For**: Quick prototyping and baseline ML performance

### Solution 2: Advanced Analytical + ML System ⭐ **Recommended**
**Focus**: Professional appraiser simulation with comprehensive analytical scoring

**Key Features**:
- **Real Ground Truth**: 264 actual appraiser selections for validation
- **Analytical Scoring**: 90+ engineered features with composite scoring (0-100 scale)
- **Professional Standards**: Follows industry appraisal criteria
- **Interpretable Results**: Clear explanations for property recommendations
- **Advanced ML**: XGBoost with feature importance analysis

**Performance**:
- **Analytical System**: Interpretable 0-100 composite scores
- **ML Enhancement**: AUC 0.651 (moderate due to extreme class imbalance)
- **Feature Insights**: Structure type matching most critical (13.9% importance)

**Best For**: Production deployment and professional appraiser tools

## 🎯 Which Solution to Use?

### Use Solution 1 if:
- You need a quick baseline ML implementation
- Working with limited computational resources
- Prototyping or research purposes
- Learning traditional ML pipelines

### Use Solution 2 if: ⭐ **Recommended**
- Building a production system for real estate professionals
- Need interpretable and explainable recommendations
- Want to validate against real appraiser decisions
- Require comprehensive feature analysis
- Building appraiser support tools

## 🚀 Quick Start Guide

### Solution 1 (Traditional ML)
```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
cd src/Solution1
python main.py
```

### Solution 2 (Advanced System) ⭐ **Recommended**
```bash
# Setup (same as above)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline
cd src/Solution2
jupyter notebook cleaning.ipynb          # Step 1: Clean data (required)
jupyter notebook feature_engineering.ipynb  # Step 2: Engineer features (required)
jupyter notebook ml_training_pipeline.ipynb # Step 3: Train ML models (optional)
```

## 📊 Key Results Summary

### Solution 1 Results
- **Best Models**: XGBoost & LightGBM (AUPRC ~0.31-0.33)
- **Main Challenge**: Severe class imbalance
- **Status**: Functional baseline implementation

### Solution 2 Results
- **Analytical Scoring**: 0-100 composite scores for all properties
- **Top Features**: Structure type match (13.9%), subject size (9.1%), property type (8.5%)
- **ML Performance**: AUC 0.651 (research phase)
- **Ground Truth**: Validated against 264 real appraiser selections
- **Status**: Production-ready analytical system, ML enhancement in research

## 🎯 Business Applications

### Solution 1
- Research and development
- ML model benchmarking
- Academic studies

### Solution 2
- **Appraisal Support Tools**: Pre-filter and rank comparables for appraisers
- **Quality Control**: Validate appraiser comparable selections
- **Training Systems**: Educational tools for new appraisers
- **Market Analysis**: Understand property similarity patterns
- **API Integration**: Real-time comparable recommendations

## File Descriptions

### Solution 1 Files
- **`src/Solution1/config.py`**: Stores global configuration constants, primarily file paths for data and caches.
- **`src/Solution1/data_loader.py`**: Functions for loading `appraisals_dataset.json` and performing initial EDA.
- **`src/Solution1/utils.py`**: Helper functions for data parsing, text standardization, and field-specific calculations.
- **`src/Solution1/geocoding_utils.py`**: Address geocoding using Nominatim with rate limiting and caching.
- **`src/Solution1/feature_engineering.py`**: Core function to transform raw appraisal data into feature-rich DataFrame.
- **`src/Solution1/model_pipeline.py`**: Generic function for training and evaluating classification models with imbalance handling.
- **`src/Solution1/main.py`**: Main executable script that orchestrates the entire pipeline.

### Solution 2 Files
- **`src/Solution2/cleaning.ipynb`**: Comprehensive data cleaning and preprocessing pipeline
- **`src/Solution2/feature_engineering.ipynb`**: Advanced feature engineering with analytical scoring system
- **`src/Solution2/ml_training_pipeline.ipynb`**: ML training with real appraiser ground truth validation
- **`src/Solution2/cleaning.py`**: Reusable data cleaning utilities and functions
- **`src/Solution2/improved_duplicate_detection.py`**: Advanced duplicate detection algorithms

## Setup and Installation

1. **Clone the repository (if applicable).**

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Current Performance Summary

### Solution 1 Performance
- **XGBoost & LightGBM**: AUPRC around 0.31-0.33, F1-score ~0.36 after threshold tuning
- **Logistic Regression**: AUPRC ~0.07, F1 ~0.07 (linear models struggle with complexity)
- **KNN with SMOTE**: AUPRC ~0.02, F1 ~0.06 (sensitive to high dimensionality)

### Solution 2 Performance
- **Analytical System**: Provides interpretable 0-100 composite scores
- **ML Enhancement**: XGBoost achieves AUC 0.651 with real appraiser validation
- **Feature Insights**: Structure type matching, subject characteristics, and location are key factors
- **Business Value**: Production-ready analytical scoring with ML research potential

## ⚠️ Known Limitations

### Solution 1
- Severe class imbalance remains challenging
- Limited feature engineering depth
- No real appraiser validation

### Solution 2
- ML component has moderate performance (AUC ~0.65) due to extreme class imbalance
- Limited ground truth data (264 appraiser selections)
- Appraiser decision subjectivity adds complexity

## Future Development

### Short-term Priorities
1. **Solution 2 Enhancement**: Improve ML performance with more ground truth data
2. **API Development**: Create REST API for Solution 2 analytical scoring
3. **User Interface**: Build web application for appraiser tools
4. **Integration**: Combine best aspects of both solutions

### Long-term Goals
- **Ensemble Approach**: Combine analytical scoring with improved ML models
- **Real-time Recommendations**: Deploy as production recommendation engine
- **Feedback Loop**: Collect user interactions to continuously improve models
- **Market Expansion**: Extend to different property types and geographic markets

---

## 📞 Support and Contribution

For questions about specific solutions:
- **Solution 1**: Refer to inline code documentation and `src/` files
- **Solution 2**: See `src/Solution2/README.md` for detailed documentation

**Technologies**: Python, Pandas, Scikit-learn, XGBoost, Jupyter Notebooks, Real Estate Appraisal
**Status**: Solution 1 (Functional), Solution 2 (Production-Ready Analytical + Research ML)
