'''Functions for feature engineering from raw appraisal data.'''
import pandas as pd
import numpy as np
from geopy.distance import geodesic

import utils
import geocoding_utils


def create_feature_dataframe(appraisals_data, geocoding_cache_obj):
    """Processes raw appraisals data to create a feature DataFrame."""
    if not appraisals_data:
        return pd.DataFrame(), geocoding_cache_obj

    feature_rows = []
    print("\nProcessing appraisals to create features...")

    for i, appraisal in enumerate(appraisals_data):
        subject = appraisal.get('subject', {})
        comps = appraisal.get('comps', [])
        properties = appraisal.get('properties', [])

        subject_eff_date = utils.parse_date(subject.get('effective_date'))
        subject_gla = utils.safe_float(subject.get('gla'))
        subject_beds = utils.safe_int(subject.get('num_beds'))
        subject_baths = utils.standardize_baths(subject.get('num_baths'))
        subject_age = utils.calculate_age(
            subject.get('year_built'), subject_eff_date)
        subject_lot_sf = utils.safe_float(subject.get('lot_size_sf'))
        subject_rooms = utils.safe_int(subject.get('room_count'))
        subject_struct_type = str(subject.get(
            'structure_type', '')).lower().strip()
        subject_style = str(subject.get('style', '')).lower().strip()
        subject_city = str(subject.get(
            'municipality_district', '')).lower().strip()
        subject_address_raw = subject.get('address')
        subject_fsa = utils.get_fsa(subject.get(
            'subject_city_province_zip') or subject.get('address'))

        subject_lat = utils.safe_float(subject.get('latitude'))
        subject_lon = utils.safe_float(subject.get('longitude'))

        if pd.isna(subject_lat) or pd.isna(subject_lon):
            if subject_address_raw:
                # print(f"Subject lat/lon missing for {subject_address_raw}, attempting geocoding...") # Moved to main
                s_lat, s_lon = geocoding_utils.geocode_address(
                    subject_address_raw, geocoding_cache_obj)
                if not pd.isna(s_lat) and not pd.isna(s_lon):
                    subject_lat = s_lat
                    subject_lon = s_lon
                # else: # Moved to main
                #     print(f"Geocoding failed or returned no result for subject: {subject_address_raw}")
            # else: # Moved to main
            #     print("Subject address is missing, cannot geocode.")

        chosen_comp_addresses = {utils.standardize_address_text(comp.get('address'))
                                 for comp in comps if comp.get('address')}

        for prop in properties:
            prop_address_raw = prop.get('address')
            prop_address_std = utils.standardize_address_text(prop_address_raw)
            is_chosen_comp = 1 if prop_address_std in chosen_comp_addresses and prop_address_std != "" else 0

            prop_close_date = utils.parse_date(prop.get('close_date'))
            days_since_sale_abs = np.nan
            prop_sold_after_eff = np.nan
            if subject_eff_date and prop_close_date:
                time_diff = (subject_eff_date - prop_close_date).days
                days_since_sale_abs = abs(time_diff)
                prop_sold_after_eff = 1 if time_diff < 0 else 0

            prop_gla = utils.safe_float(prop.get('gla'))
            gla_diff = abs(subject_gla - prop_gla) if not pd.isna(
                subject_gla) and not pd.isna(prop_gla) else np.nan
            gla_ratio = prop_gla / \
                subject_gla if subject_gla and prop_gla and subject_gla > 0 else np.nan

            prop_beds = utils.safe_int(prop.get('bedrooms'))
            bed_diff = abs(subject_beds - prop_beds) if not pd.isna(
                subject_beds) and not pd.isna(prop_beds) else np.nan

            prop_baths = utils.standardize_baths(
                None, prop.get('full_baths'), prop.get('half_baths'))
            bath_diff = abs(subject_baths - prop_baths) if not pd.isna(
                subject_baths) and not pd.isna(prop_baths) else np.nan

            prop_age = utils.calculate_age(
                prop.get('year_built'), subject_eff_date)
            age_diff = abs(subject_age - prop_age) if not pd.isna(
                subject_age) and not pd.isna(prop_age) else np.nan

            prop_lot_sf = utils.safe_float(prop.get('lot_size_sf'))
            lot_sf_diff = abs(subject_lot_sf - prop_lot_sf) if not pd.isna(
                subject_lot_sf) and not pd.isna(prop_lot_sf) else np.nan

            prop_rooms = utils.safe_int(prop.get('room_count'))
            room_diff = abs(subject_rooms - prop_rooms) if not pd.isna(
                subject_rooms) and not pd.isna(prop_rooms) else np.nan

            prop_struct_type = str(
                prop.get('structure_type', '')).lower().strip()
            struct_type_match = 1 if subject_struct_type and prop_struct_type and subject_struct_type == prop_struct_type else 0

            prop_style = str(prop.get('style', '')).lower().strip()
            style_match = 1 if subject_style and prop_style and subject_style == prop_style else 0

            prop_city = str(prop.get('city', '')).lower().strip()
            city_match = 1 if subject_city and prop_city and subject_city == prop_city else 0

            prop_fsa = utils.get_fsa(
                prop.get('postal_code') or prop_address_raw)
            fsa_match = 1 if subject_fsa and prop_fsa and subject_fsa == prop_fsa else 0

            distance_to_subject = np.nan
            prop_lat = utils.safe_float(prop.get('latitude'))
            prop_lon = utils.safe_float(prop.get('longitude'))

            if pd.isna(prop_lat) or pd.isna(prop_lon):
                if prop_address_raw:
                    p_lat, p_lon = geocoding_utils.geocode_address(
                        prop_address_raw, geocoding_cache_obj)
                    if not pd.isna(p_lat) and not pd.isna(p_lon):
                        prop_lat = p_lat
                        prop_lon = p_lon

            if (not pd.isna(subject_lat) and
                not pd.isna(subject_lon) and
                not pd.isna(prop_lat) and
                    not pd.isna(prop_lon)):
                try:
                    distance_to_subject = geodesic(
                        (subject_lat, subject_lon), (prop_lat, prop_lon)).km
                except Exception:
                    distance_to_subject = np.nan

            # Interaction and polynomial features (reverted to np.nan if components are nan)
            distance_squared = distance_to_subject**2 if not pd.isna(
                distance_to_subject) else np.nan
            age_diff_squared = age_diff**2 if not pd.isna(age_diff) else np.nan
            dist_X_age_diff = (distance_to_subject * age_diff) if not pd.isna(
                distance_to_subject) and not pd.isna(age_diff) else np.nan
            fsa_X_days_since_sale = (
                fsa_match * days_since_sale_abs) if not pd.isna(days_since_sale_abs) else np.nan

            feature_row = {
                'appraisal_orderID': appraisal.get('orderID'),
                'subject_address': subject_address_raw,
                'prop_address': prop_address_raw,
                'is_chosen_comp': is_chosen_comp,
                'days_since_sale': days_since_sale_abs,
                'prop_sold_after_eff': prop_sold_after_eff,
                'gla_diff': gla_diff,
                'gla_ratio': gla_ratio,
                'bed_diff': bed_diff,
                'bath_diff': bath_diff,
                'age_diff': age_diff,
                'lot_sf_diff': lot_sf_diff,
                'room_diff': room_diff,
                'struct_type_match': struct_type_match,
                'style_match': style_match,
                'city_match': city_match,
                'fsa_match': fsa_match,
                'distance_to_subject': distance_to_subject,
                'distance_squared': distance_squared,
                'age_diff_squared': age_diff_squared,
                'dist_X_age_diff': dist_X_age_diff,
                'fsa_X_days_since_sale': fsa_X_days_since_sale,
                # Raw values for potential later use or direct comparison if needed
                'subject_gla': subject_gla,
                'prop_gla': prop_gla,
                'subject_lot_sf': subject_lot_sf,
                'prop_lot_sf': prop_lot_sf,
                'subject_age': subject_age,
                'prop_age': prop_age,
                'subject_beds': subject_beds,
                'prop_beds': prop_beds,
                'subject_baths': subject_baths,
                'prop_baths': prop_baths,
                'subject_eff_date': subject_eff_date,
                'prop_close_date': prop_close_date,
                'subject_lat': subject_lat,
                'subject_lon': subject_lon,
                'prop_lat': prop_lat,
                'prop_lon': prop_lon
            }
            feature_rows.append(feature_row)

    df = pd.DataFrame(feature_rows)
    print(
        f"Value counts for is_chosen_comp directly after DataFrame creation:\n{df['is_chosen_comp'].value_counts()}")
    print(
        f"\nCreated DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- Imputations and Missingness Indicators ---
    print("\nImputing medians for specific diff features and creating missingness indicators...")
    for col in ['bed_diff', 'room_diff']:
        if col in df.columns and df[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(
                f"  Processed column: {col}. Missing: {df[f'{col}_missing'].sum()}, Median used: {median_val}")

    if 'distance_to_subject' in df.columns and df['distance_to_subject'].isnull().any():
        print(f"Processing column: distance_to_subject")
        df['distance_to_subject_missing'] = df['distance_to_subject'].isnull().astype(int)
        median_dist = df['distance_to_subject'].median()
        df['distance_to_subject'] = df['distance_to_subject'].fillna(
            median_dist)
        print(
            f"  Created 'distance_to_subject_missing' indicator. Median used: {median_dist}")
    else:
        print("Warning: Column 'distance_to_subject' not found or no missing values for imputation.")

    for col in ['bath_diff', 'age_diff']:
        if col in df.columns and df[col].isnull().any():
            print(f"Processing column: {col}")
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(
                f"  Created '{col}_missing' indicator column. Median used: {median_val}")
        else:
            print(
                f"Warning: Column '{col}' not found or no missing values for imputation.")

    # Impute interaction/polynomial features that might still be NaN if their *components* were NaN
    # (even if base diffs were imputed, the interaction might not have been formed)
    # XGBoost can handle NaNs, but for consistency with prior steps and other models, let's impute them if any exist.
    interaction_poly_cols = [
        'distance_squared', 'age_diff_squared', 'dist_X_age_diff', 'fsa_X_days_since_sale'
    ]
    print("\nChecking and imputing interaction/polynomial features...")
    for col in interaction_poly_cols:
        if col in df.columns and df[col].isnull().any():
            missing_count_before = df[col].isnull().sum()
            # Impute with 0 for interactions; if components were missing, interaction is likely meaningless or zero-effect
            df[col] = df[col].fillna(0.0)
            print(
                f"  Imputed {missing_count_before} NaNs in '{col}' with 0.0.")
        elif col not in df.columns:
            print(
                f"  Warning: Interaction/polynomial column '{col}' not found in DataFrame.")

    print("\n--- End of Feature Engineering ---")
    return df, geocoding_cache_obj


def describe_engineered_features(df):
    """Prints describe() and isnull().sum() for key columns after feature engineering."""
    if df.empty:
        print("DataFrame is empty. No features to describe.")
        return

    print("\n--- Engineered Features Description (Post-Imputation) ---")

    cols_to_describe = [
        'days_since_sale', 'bed_diff', 'bath_diff', 'age_diff', 'room_diff',
        'distance_to_subject', 'gla_diff', 'lot_sf_diff',
        'bath_diff_missing', 'age_diff_missing', 'distance_to_subject_missing',
        'distance_squared', 'age_diff_squared', 'dist_X_age_diff', 'fsa_X_days_since_sale'
    ]
    # Only describe columns that actually exist in the DataFrame
    existing_cols_to_describe = [
        col for col in cols_to_describe if col in df.columns]
    if existing_cols_to_describe:
        print(df[existing_cols_to_describe].describe().transpose().to_string())
    else:
        print("No specified key columns found to describe.")

    print("\nMissing values summary after all feature engineering and imputation:")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0].sort_values(
        ascending=False).to_string())
    if missing_summary.sum() == 0:
        print("No missing values remaining in the DataFrame.")

    print("\nDistribution of target 'is_chosen_comp':")
    print(df['is_chosen_comp'].value_counts(normalize=True) * 100)
    print("\nValue counts of target 'is_chosen_comp':")
    print(df['is_chosen_comp'].value_counts())

    # Matches original script's separator
    print("\n--- End of Phase 2 Data Preprocessing --- ")
