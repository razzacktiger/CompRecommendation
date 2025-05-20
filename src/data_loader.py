'''Functions for loading and performing initial EDA on the dataset.'''
import json
import pandas as pd
import os
import config
import utils


def load_appraisals_data(file_name=config.RAW_DATA_FILE):
    """Loads the appraisals dataset from a JSON file located within src/.

    Args:
        file_name (str, optional): The base name of the data file.
                                     Defaults to config.RAW_DATA_FILE.
                                     The path is constructed relative to config.py's location.
    """
    # Construct path relative to the directory of config.py (i.e., src/)
    # This ensures it works correctly whether called from src/main.py or a script in root.
    base_src_dir = os.path.dirname(config.__file__)
    full_file_path = os.path.join(base_src_dir, file_name)

    print(f"Loading {full_file_path}...")
    try:
        with open(full_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # If the loaded data is a dictionary and has an 'appraisals' key,
        # assume the actual list of appraisals is nested there.
        if isinstance(data, dict) and 'appraisals' in data:
            data = data['appraisals']
            print(
                "Note: Data was unwrapped from an 'appraisals' key in the JSON structure.")

        if isinstance(data, list):
            print(f"Successfully loaded {len(data)} appraisals.")
            return data
        else:
            # This case should be rare if the primary structure is a list or dict with 'appraisals'
            error_msg = f"Error: Loaded data from {full_file_path} is not a list of appraisals as expected. Type found: {type(data)}."
            if hasattr(data, '__len__'):
                error_msg += f" Number of top-level elements: {len(data)}."
            else:
                error_msg += " Data does not have a defined length."
            print(error_msg)
            # Decide on behavior: return None, raise error, or return data with warning
            # For now, returning None as the original error paths did.
            return None

    except FileNotFoundError:
        print(f"Error: The file {full_file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {full_file_path} is not a valid JSON file.")
        return None
    except Exception as e:  # Catch other potential errors during file/JSON processing
        print(
            f"An unexpected error occurred while loading {full_file_path}: {e}")
        return None


def perform_initial_eda(appraisals_data):
    """Prints initial exploratory data analysis insights."""
    if not appraisals_data:
        print("No data to perform EDA on.")
        return

    print("\n--- Exploratory Data Analysis ---")
    print(f"Total number of appraisals: {len(appraisals_data)}")

    if appraisals_data:
        first_appraisal = appraisals_data[0]
        print("\nStructure of the first appraisal (keys):")
        print(list(first_appraisal.keys()))

        subject_prop = first_appraisal.get('subject', {})
        print("\nKeys in the 'subject' property of the first appraisal:")
        print(list(subject_prop.keys()))
        print(f"Subject property address: {subject_prop.get('address')}")
        print(f"Subject GLA: {subject_prop.get('gla')}, "
              f"Beds: {subject_prop.get('num_beds')}, "
              f"Baths: {subject_prop.get('num_baths')}")

        comps = first_appraisal.get('comps', [])
        print(
            f"\nNumber of chosen comparables (comps) in the first appraisal: {len(comps)}")
        if comps:
            print("Keys in the first chosen 'comp':")
            print(list(comps[0].keys()))
            print(f"First chosen comp address: {comps[0].get('address')}")

        properties = first_appraisal.get('properties', [])
        print(
            f"\nNumber of potential comparables ('properties') in the first appraisal: {len(properties)}")
        if properties:
            print("Keys in the first 'property' from the potential list:")
            print(list(properties[0].keys()))

    # Statistics for 'properties' list per appraisal
    num_properties_list = [len(appraisal.get('properties', []))
                           for appraisal in appraisals_data]
    if num_properties_list:
        print("\nStatistics for 'properties' list per appraisal:")
        s_num_properties = pd.Series(num_properties_list)
        print(f"  Min: {s_num_properties.min()}")
        print(f"  Max: {s_num_properties.max()}")
        print(f"  Avg: {s_num_properties.mean():.2f}")
        print(f"  Median: {s_num_properties.median()}")

    # Distribution of chosen 'comps' (should always be 3 based on prior analysis)
    num_comps_list = [len(appraisal.get('comps', []))
                      for appraisal in appraisals_data]
    if num_comps_list:
        print("\nDistribution of number of chosen 'comps' per appraisal:")
        print(pd.Series(num_comps_list).value_counts())

    # Show subject GLA missingness
    subject_gla_missing_count = 0
    for appraisal in appraisals_data:
        if pd.isna(utils.safe_float(appraisal.get('subject', {}).get('gla'))):
            subject_gla_missing_count += 1
    print(
        f"\nNumber of appraisals with missing subject_gla: {subject_gla_missing_count} out of {len(appraisals_data)}")

    # Show subject lot_size_sf missingness
    subject_lot_sf_missing_count = 0
    for appraisal in appraisals_data:
        if pd.isna(utils.safe_float(appraisal.get('subject', {}).get('lot_size_sf'))):
            subject_lot_sf_missing_count += 1
    print(
        f"Number of appraisals with missing subject_lot_sf: {subject_lot_sf_missing_count} out of {len(appraisals_data)}")

    # Show subject lat/lon missingness
    subject_lat_lon_missing = 0
    for appraisal in appraisals_data:
        subj = appraisal.get('subject', {})
        if pd.isna(utils.safe_float(subj.get('latitude'))) or pd.isna(utils.safe_float(subj.get('longitude'))):
            subject_lat_lon_missing += 1
    print(
        f"Number of subjects missing direct lat/lon: {subject_lat_lon_missing} out of {len(appraisals_data)}")

    # Matches the original script's output separator
    print("\n--- End of Initial EDA --- ")
