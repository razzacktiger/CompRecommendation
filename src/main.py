'''Main script to run the property appraisal recommendation system.'''

import pandas as pd

# Import a new module that we will create next to hold the main logic.
# This is to keep this main.py file clean and focused on orchestration.

import config
import utils
import geocoding_utils
import data_loader
import feature_engineering
import model_pipeline


def main():
    """Orchestrates the full data processing and modeling pipeline."""
    print("Starting the Appraisal Recommendation Pipeline...")

    # 1. Load Data
    appraisals_data = data_loader.load_appraisals_data()
    if appraisals_data is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Initial EDA (Optional - can be toggled)
    # data_loader.perform_initial_eda(appraisals_data)

    # 3. Load Geocoding Cache
    # Now uses default file_name from config, path constructed within the function
    geocoding_cache = geocoding_utils.load_geocoding_cache()

    # 4. Feature Engineering
    # Pass the cache, and get the (potentially updated) cache back
    df_features, geocoding_cache = feature_engineering.create_feature_dataframe(
        appraisals_data, geocoding_cache)

    # 5. Save updated geocoding cache
    # Now uses default file_name from config, path constructed within the function
    geocoding_utils.save_geocoding_cache(geocoding_cache)

    if df_features.empty:
        print("Feature DataFrame is empty. Exiting.")
        return

    # 6. Describe Engineered Features (Optional)
    # feature_engineering.describe_engineered_features(df_features)

    # 7. Model Training and Evaluation
    # XGBoost
    model_pipeline.train_evaluate_model(df_features, model_name='XGBoost')

    # LightGBM
    model_pipeline.train_evaluate_model(df_features, model_name='LightGBM')

    # Logistic Regression
    model_pipeline.train_evaluate_model(
        df_features, model_name='LogisticRegression')

    # KNN
    model_pipeline.train_evaluate_model(df_features, model_name='KNN')

    print("\nAppraisal Recommendation Pipeline finished.")


if __name__ == "__main__":
    main()
