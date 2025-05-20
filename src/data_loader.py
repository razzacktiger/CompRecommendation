'''Functions for loading and performing initial EDA on the dataset.'''
import json
import pandas as pd
import config
import utils


def load_appraisals_data(file_path=config.RAW_DATA_FILE):
    """Loads the appraisals dataset from a JSON file."""
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # --- DEBUGGING ---
        print(f"Type of data loaded: {type(data)}")
        if isinstance(data, dict):
            print(f"Data is a dictionary. Keys: {list(data.keys())}")
            # Attempt to access a common key if it's a dict of length 1
            if len(data) == 1:
                first_key = list(data.keys())[0]
                print(
                    f"First key is '{first_key}'. Attempting to use data[first_key].")
                # POTENTIAL FIX: If the actual list is nested, reassign data
                if first_key == 'appraisals':  # Be specific if we know the key
                    data = data[first_key]
                    print(
                        f"Reassigned data to content of '{first_key}'. New type: {type(data)}")
                else:
                    print(
                        f"Warning: Dictionary has one key, but it's not 'appraisals'. Check data structure.")
        elif isinstance(data, list):
            print(f"Data is a list. Number of items: {len(data)}")
        # --- END DEBUGGING ---

        # Original length check, might be misleading if data is a dict
        # print(f"Loaded {len(data)} appraisals.")

        # We expect data to be a list of appraisals at this point.
        # If it was a dict and we reassigned it above, this len will be correct.
        if isinstance(data, list):
            print(
                f"Loaded {len(data)} appraisals (after potential dict unwrapping).")
            return data
        else:
            print(
                "Warning: Loaded data is not a list as expected. Returning as is, but this might cause issues.")
            print(
                f"Number of top-level elements in loaded data: {len(data) if hasattr(data, '__len__') else 'N/A (not a sequence)'}")
            return data  # Still return it, but with a warning

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
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
