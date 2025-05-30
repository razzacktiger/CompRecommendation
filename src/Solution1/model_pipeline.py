'''Functions for training, evaluating, and tuning machine learning models.'''
import pandas as pd
import numpy as np
# , StratifiedKFold, RandomizedSearchCV (Removed for now)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# Define a standard set of features to be used by models
# This could also be moved to config.py if it gets very complex or is shared more widely
DEFAULT_FEATURE_COLUMNS = [
    'distance_to_subject',
    'distance_to_subject_missing',
    'days_since_sale',
    'prop_sold_after_eff',
    'age_diff',
    'age_diff_missing',
    'bath_diff',
    'bath_diff_missing',
    'bed_diff',
    'bed_diff_missing',
    # 'room_diff', # Formally removed as models assign it no importance
    # 'room_diff_missing', # Formally removed as models assign it no importance
    'struct_type_match',
    'fsa_match'
    # Other features confirmed as not used or having low/no impact:
    # style_match, city_match, distance_squared, age_diff_squared,
    # dist_X_age_diff, fsa_X_days_since_sale
]


def train_evaluate_model(df, model_name='XGBoost', feature_columns=None, random_state=42, test_size=0.2):
    """Trains, evaluates a specified model, and performs threshold tuning."""
    if df.empty:
        print(f"DataFrame is empty. Cannot train/evaluate {model_name}.")
        return

    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    # Ensure all specified feature columns exist in the DataFrame
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(
            f"Error: The following feature columns are missing from the DataFrame: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print(f"\n\n--- Starting {model_name} Model Training and Evaluation ---")

    X = df[feature_columns]
    y = df['is_chosen_comp']

    print(f"\nShape of X (features): {X.shape}")
    print(f"Shape of y (target): {y.shape}")
    if X.empty or y.empty:
        print("Feature set X or target y is empty. Aborting training.")
        return
    if y.nunique() < 2:
        print(
            f"Target variable 'is_chosen_comp' has only {y.nunique()} unique value(s). Needs at least 2 for classification. Aborting.")
        return

    print("\nPerforming stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("Distribution of target in y_train:")
    print(y_train.value_counts(normalize=True) * 100)
    print("Distribution of target in y_test:")
    print(y_test.value_counts(normalize=True) * 100)

    # Prepare data for scaling and fitting - potentially apply SMOTE for KNN
    X_train_to_scale = X_train
    y_train_for_fit = y_train

    counts = y_train.value_counts()  # Original y_train counts

    if model_name.lower() == 'knn':
        print("\nAttempting SMOTE for KNN training data...")
        # Count of positive class (1) in original y_train
        minority_class_count = counts.get(1, 0)

        if minority_class_count < 2:
            print(
                f"Warning: Minority class count in training set ({minority_class_count}) is too small for SMOTE. Using original training data for KNN.")
            # X_train_to_scale and y_train_for_fit remain as original X_train, y_train
        else:
            smote_k_neighbors = min(5, minority_class_count - 1)
            if smote_k_neighbors < 1:  # Should ideally not be reached if minority_class_count >= 2
                print(
                    f"Warning: Calculated k_neighbors for SMOTE is {smote_k_neighbors}. SMOTE requires k_neighbors >= 1. Using original data.")
            else:
                print(
                    f"Applying SMOTE with k_neighbors={smote_k_neighbors} for KNN.")
                smote = SMOTE(random_state=random_state,
                              k_neighbors=smote_k_neighbors)
                try:
                    X_train_resampled, y_train_resampled = smote.fit_resample(
                        X_train, y_train)
                    print(
                        f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
                    print("Distribution of target in y_train after SMOTE:")
                    print(pd.Series(y_train_resampled).value_counts(
                        normalize=True) * 100)
                    X_train_to_scale = X_train_resampled
                    y_train_for_fit = y_train_resampled
                except ValueError as e:
                    print(
                        f"Error during SMOTE: {e}. Using original training data for KNN.")

    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    # Fit on X_train_to_scale (original or resampled)
    X_train_scaled = scaler.fit_transform(X_train_to_scale)
    X_test_scaled = scaler.transform(X_test)  # Transform original X_test

    # Calculate scale_pos_weight for handling class imbalance (based on original y_train counts)
    # This is primarily for XGBoost/LightGBM if SMOTE is not used for them.
    if len(counts) < 2 or counts.get(1, 0) == 0:
        print("Warning: Positive class (1) has zero samples in the training set after split, or only one class present.")
        print("Cannot calculate scale_pos_weight. Using default of 1.")
        scale_pos_weight = 1
    else:
        scale_pos_weight = counts[0] / counts[1]
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    model_instance = None
    if model_name.lower() == 'xgboost':
        print("\nTraining XGBoost model...")
        model_instance = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_name.lower() == 'lightgbm':
        print("\nTraining LightGBM model...")
        model_instance = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_estimators=100
        )
    elif model_name.lower() == 'logisticregression':
        print("\nTraining Logistic Regression model...")
        model_instance = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
    elif model_name.lower() == 'knn':
        print("\nTraining KNN model...")
        model_instance = KNeighborsClassifier(n_neighbors=5)
    else:
        print(
            f"Unsupported model type: {model_name}. Supported: XGBoost, LightGBM, LogisticRegression, KNN. Skipping.")
        return

    # Fit with potentially SMOTEd y_train_for_fit
    model_instance.fit(X_train_scaled, y_train_for_fit)
    print(f"{model_name} model training complete.")

    print(
        f"\nEvaluating {model_name} model on the test set (default 0.5 threshold)...")
    y_pred = model_instance.predict(X_test_scaled)
    y_pred_proba = model_instance.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report (0.5 threshold):")
    print(classification_report(y_test, y_pred, zero_division=0))

    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"\nAverage Precision Score (AUPRC) for {model_name}: {auprc:.4f}")

    print(f"\nFeature Importances ({model_name} - Gain/Weight):")
    if hasattr(model_instance, 'feature_importances_'):
        importances = pd.Series(
            model_instance.feature_importances_, index=X.columns)
        print(importances.sort_values(ascending=False))
    elif hasattr(model_instance, 'coef_'):
        importances = pd.Series(model_instance.coef_[0], index=X.columns)
        print(importances.abs().sort_values(ascending=False))
        print("\n(Note: For Logistic Regression, these are coefficient magnitudes)")
    elif model_name.lower() == 'knn':
        print("KNN does not provide direct feature importances in the same way as other models.")
    else:
        print("Model does not support feature_importances_ or coef_ attribute.")

    # --- Threshold Tuning ---
    print(
        f"\n--- Starting Threshold Tuning for Positive Class (1) - {model_name} ---")
    thresholds = np.arange(0.01, 0.6, 0.01)
    best_f1_positive = -1.0  # Initialize to a value that will be overridden
    best_threshold_f1 = 0.5
    best_precision_positive = 0
    best_recall_positive = 0

    print("Threshold | Precision (1) | Recall (1) | F1-score (1)")
    print("-----------------------------------------------------")

    # Check if there are positive samples in y_test to evaluate against
    if 1 not in y_test.unique():
        print("No positive samples (class 1) in the test set. Skipping threshold tuning for positive class.")
    else:
        for threshold in thresholds:
            y_pred_tuned = (y_pred_proba >= threshold).astype(int)
            # Ensure labels=[1] to focus on the positive class
            precision, recall, f1, support = precision_recall_fscore_support(
                # Use average='binary' for single class metrics
                y_test, y_pred_tuned, labels=[1], zero_division=0, average='binary'
            )
            # If average='binary', precision, recall, f1 are floats, not arrays
            print(
                f"{threshold:9.2f} | {precision:13.4f} | {recall:10.4f} | {f1:12.4f}")
            if f1 > best_f1_positive:
                best_f1_positive = f1
                best_threshold_f1 = threshold
                best_precision_positive = precision
                best_recall_positive = recall
            elif f1 == best_f1_positive and precision > best_precision_positive:
                best_threshold_f1 = threshold
                best_precision_positive = precision
                best_recall_positive = recall

    print("-----------------------------------------------------")
    if best_f1_positive >= 0:  # Check if any positive class was actually found and scored
        print(
            f"Best F1-score (Positive Class) for {model_name}: {best_f1_positive:.4f} at threshold {best_threshold_f1:.2f}")
        print(
            f"  Precision (Positive Class) at best F1 threshold: {best_precision_positive:.4f}")
        print(
            f"  Recall (Positive Class) at best F1 threshold: {best_recall_positive:.4f}")

        print(
            f"\nClassification Report for {model_name} with Best F1 Threshold ({best_threshold_f1:.2f}):")
        y_pred_best_f1 = (y_pred_proba >= best_threshold_f1).astype(int)
        print(classification_report(y_test, y_pred_best_f1, zero_division=0))
    else:
        print(
            f"Could not determine a best threshold for {model_name} based on F1-score for the positive class (likely no true positives predicted or no positive class in test set).")

    print(f"--- End of {model_name} Model Training and Evaluation ---")
