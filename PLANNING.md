# PLANNING.md

## 1. Project Goals

- Develop a recommendation system to select the best comparable properties (comps) from a dataset of appraisals.
- Ensure the system is scalable to handle increasing amounts of data.
- Benchmark the system on a validation set.
- Focus on back-end development and machine learning.
- a featrues to consider for a good ocmparable is location/neighborhood, quality/age/condition structure type, how recent (sold within 90 days?), features (bedroom, bath), living area (gla perhaps) 

## 2. Core Features & Milestones

The project will be developed through the following milestones:

### Milestone 1: Statistical Modeling & Core Recommendation Engine

- **Data Standardization and Cleaning:**
  - Identify and handle duplicate entries between "comps" and "properties".
  - Standardize naming conventions for data points.
  - Implement robust data cleaning pipelines.
- **Statistical Modeling for Comp Scoring:**
  - Develop a system for scoring properties based on their quality as a comp.
  - Explore and implement statistical modeling techniques, such as:
    - Clustering algorithms (e.g., K-Means, DBSCAN).
    - Nearest Neighbors (NN) algorithms (e.g., k-NN).
    - Other relevant statistical methods and distribution analysis.
- **Core Recommendation Logic:**
  - Implement the logic to recommend comps based on the scoring system.

### Milestone 2: Explainability

- **Integrate Explainable AI (XAI):**
  - Incorporate techniques to explain why a property is a good or bad comparable.
  - Explore using Large Language Models (LLMs) for generating natural language explanations.
  - Consider other XAI methods relevant to the statistical models used.
  - Potentially explore fine-tuning LLMs with Reinforcement Learning (RL) for improved explanation quality, if feasible.

### Milestone 3: Self-Improving System

- **Feedback Loop Implementation:**
  - Design a mechanism to incorporate new data points as they become available.
  - Develop a system to capture and utilize feedback from appraisers on selected comps.
- **Model Refinement and Retraining:**
  - Implement processes for continuous model refinement based on new data and feedback.
  - Establish a retraining schedule or trigger-based retraining.

## 3. Architecture & Technology Stack (Initial Thoughts)

- **Language:** Python (as per custom instructions, and suitable for ML).
- **Data Handling & ML:**
  - Libraries like `pandas` for data manipulation.
  - `scikit-learn` for statistical modeling (clustering, NN, etc.).
  - Potentially `NumPy` for numerical operations.
- **Explainability (LLMs):**
  - Libraries like `transformers` (Hugging Face) if using pre-trained LLMs.
  - Consider frameworks for fine-tuning if pursuing that route.
- **API/Backend (if exposing the system):**
  - `FastAPI` or `Django` (as per custom instructions). `FastAPI` is often preferred for ML-focused APIs due to its speed and modern features.
- **Data Validation:** `pydantic` (as per custom instructions).
- **Database (for storing appraisal data, feedback, model versions - TBD):**
  - Could range from simple file storage (CSVs, Parquet) for initial stages to a more robust database (e.g., PostgreSQL with SQLAlchemy/SQLModel) as the system scales. This needs further evaluation based on data size and query needs.
- **Testing:** `pytest` (as per custom instructions).

## 4. Development Style & Conventions

- **PEP8:** Adhere to PEP8 style guidelines.
- **Type Hints:** Use type hints for all function signatures and variables.
- **Formatting:** Use `black` for code formatting.
- **Docstrings:** Google style docstrings for all functions, classes, and modules.
- **Modularity:** Organize code into clearly separated modules by feature or responsibility.
- **File Length:** Aim to keep files under 500 lines; refactor if they grow larger.
- **Imports:** Use clear, consistent imports (prefer relative imports within packages).
- **Comments:** Comment non-obvious code with `# Reason:` comments for complex logic.
- **Version Control:** Git.

## 5. Constraints & Considerations

- **Data Quality:** The initial dataset requires significant standardization and cleaning. This is a critical first step.
- **Scalability:** Design choices should consider the need to handle increasing data volumes.
- **Benchmarking:** A validation set will be used for benchmarking; this implies the need for a clear evaluation strategy and metrics.
- **Iterative Development:** The milestones suggest an iterative approach, building complexity gradually.
- **Resource Availability:** For LLM fine-tuning or extensive training, computational resources (GPU, etc.) might be a constraint to consider.

## 6. Project Structure (Initial Proposal)

```
/CompRecommendation
|-- app/                      # Main application code
|   |-- __init__.py
|   |-- data_processing/      # Modules for data loading, cleaning, standardization
|   |   |-- __init__.py
|   |   |-- loader.py
|   |   |-- cleaner.py
|   |   |-- standardizer.py
|   |-- modeling/             # Statistical models, scoring logic
|   |   |-- __init__.py
|   |   |-- statistical_models.py
|   |   |-- scoring.py
|   |-- explainability/       # XAI and LLM integration
|   |   |-- __init__.py
|   |   |-- explainer.py
|   |-- feedback/             # Modules for handling feedback and retraining
|   |   |-- __init__.py
|   |   |-- collector.py
|   |   |-- retraining.py
|   |-- api/                  # FastAPI/Django API endpoints (if applicable)
|   |   |-- __init__.py
|   |   |-- routes.py
|   |   |-- schemas.py        # Pydantic schemas
|   |-- core/                 # Core configuration, utilities
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- utils.py
|-- data/                     # Raw, processed, and validation datasets
|   |-- raw/
|   |-- processed/
|   |-- validation/
|-- notebooks/                # Jupyter notebooks for exploration, experimentation
|-- tests/                    # Pytest unit tests
|   |-- __init__.py
|   |-- data_processing/
|   |-- modeling/
|   |-- explainability/
|   |-- api/
|-- scripts/                  # Helper scripts (e.g., for training, data conversion)
|-- .gitignore
|-- README.md
|-- PLANNING.md               # This file
|-- TASK.MD                   # Task tracking
|-- requirements.txt          # Python dependencies
```

## 7. Further Questions/Clarifications Needed

- What is the expected size and format of the initial dataset?
  Answer: The size is 21.4 MB and is in json format
- Are there specific performance metrics for the recommendation system?
- What are the expectations for the "validation set" and how will benchmarking be performed?
- Is there a preference for specific clustering or NN algorithms?
- What is the timeline or priority for each milestone?
- Will the system be a standalone library, a service with an API, or integrated into another system?
