
## 📓 Notebooks and Files Description

### 1. `cleaning.ipynb` - Data Preprocessing Pipeline
**Purpose**: Clean and standardize raw real estate data for analysis

**What it does**:
- Loads raw appraisal dataset from JSON format
- Standardizes property addresses and geocoding
- Handles missing values and data type conversions
- Removes invalid or incomplete property records
- Creates consistent subject and comparable property datasets

**Key Outputs**:
- `subjects_cleaned.csv`: Clean subject property data
- `comps_cleaned_with_subjects.csv`: Real appraiser selections
- `properties_deduplicated.csv`: Deduplicated property dataset

**Run Time**: ~1-2 minutes (on mac arm)

### 2. `feature_engineering.ipynb` - Analytical Scoring System
**Purpose**: Create comprehensive feature set for property comparison and analytical scoring

**What it does**:
- **Physical Similarity**: Size, bedrooms, bathrooms, structure type matching
- **Location Analysis**: Distance calculation, same city/neighborhood detection
- **Temporal Features**: Sale recency, market timing factors
- **Market Analysis**: Price per sqft analysis, local market percentiles
- **Composite Scoring**: Combines all factors into unified property scores

**Key Features Created**:
- Physical similarity scores (size, bedroom, bathroom matches)
- Location proximity metrics (distance, same city indicators)
- Temporal recency scores (days since sale, market timing)
- Market-based features (price percentiles, price per sqft ratios)
- **Composite Score**: Final analytical ranking (0-100 scale)

**Key Output**: `properties_comparison_engineered.csv` (90+ features)

**Run Time**: ~15-20 minutes

### 3. `ml_training_pipeline.ipynb` - Machine Learning Training
**Purpose**: Train ML models to predict appraiser selection patterns

**What it does**:
- Maps real appraiser selections to engineered dataset
- Creates binary labels (selected vs. not selected by appraisers)
- Trains multiple ML algorithms (Random Forest, XGBoost, Logistic Regression, SVM)
- Handles extreme class imbalance (3% positive class)
- Evaluates model performance and feature importance

**Current Results**:
- **Best Model**: XGBoost (AUC: 0.651)
- **Key Challenge**: Extreme class imbalance and limited positive samples
- **Feature Insights**: Structure type matching is most important factor

**Key Output**: Trained ML models and performance analysis

**Run Time**: ~1 minutes

### 4. `cleaning.py` - Data Processing Utilities
**Purpose**: Placeholder for refactoring cleaning.ipynb into a py file, placeholder utility use for now. 

### 5. `improved_duplicate_detection.py` - Advanced Deduplication
**Purpose**: Sophisticated algorithms to detect and handle property duplicates (ran inside of cleaning.ipynb)

**Features**:
- Multi-field similarity matching
- Fuzzy address matching
- Price and characteristics comparison
- Confidence scoring for duplicate detection

### 6. `config.py` - Configuration Management
**Purpose**: Centralized configuration for file paths and settings

## 🚀 How to Run the Project

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy scikit-learn xgboost
pip install geopy matplotlib seaborn
pip install jupyter notebook
pip install requirements.txt
```

### Step-by-Step Execution

1. **Data Cleaning** (Required - First Step)
   ```bash
   jupyter notebook cleaning.ipynb
   ```
   - Run all cells sequentially
   - Outputs: Clean subject, property and comps datasets
   - **Must complete before other notebooks**

2. **Feature Engineering** (Required - Second Step)  
   ```bash
   jupyter notebook feature_engineering.ipynb
   ```
   - Creates analytical scoring system
   - Generates 90+ engineered features
   - Produces composite property scores

3. **ML Training** (Optional - Third Step)
   ```bash
   jupyter notebook ml_training_pipeline.ipynb
   ```
   - Trains ML models on appraiser selections
   - Evaluates model performance
   - Provides feature importance analysis

### Quick Start (Full Pipeline)
```bash
# Run complete pipeline
jupyter notebook cleaning.ipynb          # Step 1: Clean data
jupyter notebook feature_engineering.ipynb  # Step 2: Engineer features  
jupyter notebook ml_training_pipeline.ipynb # Step 3: Train ML models
```

## 📈 Key Results and Insights

### Analytical Scoring System Performance
- **Composite Score Range**: 0-100 (higher = better comparable)
- **Key Factors**: Structure type match, size similarity, location proximity, sale recency
- **Business Value**: Provides interpretable property rankings

### Machine Learning Results
- **Current Performance**: Moderate (AUC ~0.65)
- **Key Challenge**: Extreme class imbalance (only 3% positive samples)
- **Top Features**: Structure type matching, subject characteristics, location
- **Status**: Research phase - not ready for production

### Feature Importance Rankings
1. **Structure Type Match** (13.9%) - Must match for good comparable
2. **Subject Property Size** (9.1%) - Reference point for comparison  
3. **Property Type** (8.5%) - Classification affects selection patterns
4. **Location District** (7.7%) - Local market area importance
5. **Property Size** (7.0%) - Fundamental comparison factor

## 🎯 Business Applications

### Current Capabilities
- **Property Ranking**: Analytical scores for comparable property ranking
- **Feature Analysis**: Understanding of key property comparison factors
- **Data Processing**: Clean, standardized real estate datasets
- **Insights**: Professional appraiser decision pattern analysis

### Potential Use Cases
- **Appraisal Support**: Pre-filter and rank potential comparables
- **Market Analysis**: Understand local property similarity patterns  
- **Quality Control**: Validate appraiser comparable selections
- **Training Tool**: Educational resource for new appraisers

## ⚠️ Current Limitations

### Data Challenges
- **Limited Ground Truth**: Only 264 real appraiser selections
- **Class Imbalance**: 97% negative samples makes ML training difficult
- **Subjective Decisions**: Appraiser preferences vary by individual
- **Data Quality**: Some mapping errors between datasets

### Model Performance
- **Moderate Accuracy**: ML models achieve ~65% AUC
- **High False Positives**: Models recommend too many properties
- **Low Precision**: Only 13.5% of ML recommendations are correct
- **Research Stage**: Not ready for production deployment

## 🛠️ Future Improvements

### Data Enhancement
- **More Ground Truth**: Collect additional appraiser selections
- **Feature Engineering**: Add property condition, renovation history
- **Market Data**: Include local market trends and comparable sales volume
- **Appraiser Feedback**: Collect reasoning behind selections

### Model Improvements  
- **Ensemble Methods**: Combine multiple algorithms
- **Learning-to-Rank**: Focus on ranking rather than classification
- **Hybrid Approach**: Combine analytical scoring with ML refinement
- **Explainable AI**: Provide reasoning for recommendations

### Business Integration
- **API Development**: Create REST API for real-time recommendations
- **User Interface**: Build web application for appraisers
- **Feedback Loop**: Collect user interactions to improve models
- **A/B Testing**: Compare analytical vs. ML recommendations

## 📊 Data Schema

### Key Datasets

**subjects_cleaned.csv** (88 rows)
**comps_cleaned_with_subjects.csv** 
**properties_comparison_engineered**