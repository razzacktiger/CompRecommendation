# Comp Recommendation System

## Project Overview

This project aims to build a recommendation system for selecting the best comparable properties (comps) from a given dataset of appraisals. Each appraisal in the dataset contains information about the subject property, all available properties, and the comps that were ultimately selected. The system should be scalable to handle increasing amounts of data and will be benchmarked on a validation set. The primary focus is on back-end development and machine learning.

Before starting, you must take time to standardize and clean the data you're working with as there may be duplicates between the "comps" and "properties" as well as different names for certain datapoints.

## Milestones

The project can be implemented in a variety of different approaches, here are some options with increasing levels of difficulty:

### 1. Statistical Modeling

Develop a system for scoring properties based on their quality as a comp. This will involve using statistical modeling techniques, which may include:

- Clustering algorithms
- Nearest Neighbors (NN)
- Other statistical methods and distribution analysis

### 2. Explainability

Integrate Large Language Models (LLMs) or explainable AI techniques into the recommendation process to provide explanations for why a particular property is a good or bad comparable in relation to others. Can try techniques like finetuning LLMs with RL

### 3. Self-Improving System

Design the system to learn and improve from human feedback. It should be capable of:

- Incorporating new data points as they become available.
- Utilizing feedback from appraisers on the actual comps selected.
- Using this new information for model refinement and retraining.

## Focus

- **Back-end Development:** Building the core logic and infrastructure for the recommendation system.
- **Machine Learning:** Implementing and training the models for scoring and explainability.

## Dataset

The core dataset consists of appraisals, where each record includes:

- Subject property details.
- A list of all potentially comparable properties.
- The set of comparable properties that were selected by an appraiser.
