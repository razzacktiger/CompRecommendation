#!/usr/bin/env python3
"""
Fix Properties Extraction - Add Subject Associations & Handle Duplicates
Run this script to properly extract properties with subject links from the normalized appraisals data.
"""

import pandas as pd
import numpy as np
import ast
from datetime import datetime
import pickle

print("🔧 FIXING PROPERTIES EXTRACTION WITH SUBJECT ASSOCIATIONS")
print("=" * 60)

# Load the normalized appraisals data
print("📂 Loading normalized appraisals data...")
df_normalized = pd.read_csv('../data/processed/normalized_appraisals.csv')
print(f"✅ Loaded {len(df_normalized)} appraisal records")

# Extract properties data with SUBJECT ASSOCIATIONS + duplicate handling
print("\n🔧 Extracting properties with subject associations...")
properties_data = []
prop_idx = 0

# Iterate through subjects to preserve subject-property associations
for subject_idx, row in df_normalized.iterrows():
    # Parse properties for this subject
    properties_list = row['properties']
    if isinstance(properties_list, str):
        properties_list = ast.literal_eval(properties_list)

    # Add each property with subject association
    for prop in properties_list:
        property_record = {
            # Identifiers - WITH SUBJECT ASSOCIATION
            'property_id': prop_idx,
            'property_index': prop_idx,
            'subject_id': subject_idx,  # ✅ CRITICAL: Link to subject
            'orderID': row['orderID'],  # ✅ Also preserve original order ID

            # Core cleaned fields
            'structure_type': prop.get('structure_type'),
            'property_sub_type': prop.get('property_sub_type'),
            'prop_type_clean': prop.get('prop_type_clean'),
            'gla_sqft': prop.get('gla'),
            'bedrooms_main': prop.get('main_beds'),
            'bedrooms_additional': prop.get('additional'),
            'bedrooms_total': prop.get('total_possible'),
            'bathrooms_full': prop.get('bath_count_full'),
            'bathrooms_half': prop.get('bath_count_half'),
            'bathrooms_equivalent': prop.get('bath_count_total_equivalent'),
            'close_price': prop.get('close_price'),
            'close_date': prop.get('close_date_parsed'),

            # Location fields
            'address': prop.get('address'),
            'city': prop.get('city'),
            # ✅ FIXED: was 'state_province'
            'state_province': prop.get('province'),
            'postal_code': prop.get('postal_code'),
            'latitude': prop.get('latitude'),
            'longitude': prop.get('longitude'),

            # Additional fields
            # ✅ FIXED: was 'lot_size'
            'lot_size_sqft': prop.get('lot_size_sf'),
            'year_built': prop.get('year_built'),
            # ✅ FIXED: was 'basement_finish'
            'basement_type': prop.get('basement'),
            # ✅ NEW: additional field
            'main_level_finished_area': prop.get('main_level_finished_area'),
            # ✅ FIXED: was 'stories', using 'levels' instead
            'levels': prop.get('levels'),
            'heating': prop.get('heating'),
            'cooling': prop.get('cooling'),

            # Data quality flags
            'has_missing_bed_data': prop.get('has_missing_bed_data', False),
            'bedroom_imputation_method': prop.get('bedroom_imputation_method'),
            'has_missing_bath_data': prop.get('has_missing_bath_data', False),
            'has_missing_prop_type': prop.get('has_missing_prop_type', False),
            'prop_type_mapping_method': prop.get('mapping_method'),
            'has_missing_date_data': prop.get('close_date_is_missing', False),
        }
        properties_data.append(property_record)
        prop_idx += 1  # Increment for next property

# Create DataFrame
properties_df = pd.DataFrame(properties_data)
print(
    f"✅ Created properties DataFrame: {len(properties_df)} records × {len(properties_df.columns)} columns")

# Handle duplicate addresses
print("\n🔍 CHECKING FOR DUPLICATE ADDRESSES...")
address_counts = properties_df['address'].value_counts()
duplicates = address_counts[address_counts > 1]

if len(duplicates) > 0:
    print(f"⚠️  Found {len(duplicates)} addresses with duplicates")
    print(
        f"   Total duplicate properties: {duplicates.sum() - len(duplicates)}")

    # Add duplicate flags
    properties_df['is_duplicate_address'] = properties_df['address'].duplicated(
        keep=False)
    properties_df['duplicate_count'] = properties_df['address'].map(
        address_counts)

    print("✅ Added duplicate flags: 'is_duplicate_address', 'duplicate_count'")

    # Show top duplicates
    print("\n📋 TOP DUPLICATE ADDRESSES:")
    for addr, count in duplicates.head(5).items():
        print(f"   '{addr}': {count} properties")
else:
    properties_df['is_duplicate_address'] = False
    properties_df['duplicate_count'] = 1
    print("✅ No duplicate addresses found")

# Verify subject associations
print(f"\n📊 SUBJECT ASSOCIATION VERIFICATION:")
print(f"   Unique subjects: {properties_df['subject_id'].nunique()}")
print(
    f"   Properties per subject (avg): {len(properties_df) / properties_df['subject_id'].nunique():.1f}")
print(
    f"   Subject ID range: {properties_df['subject_id'].min()} - {properties_df['subject_id'].max()}")

# Save the corrected dataset
print(f"\n💾 SAVING CORRECTED PROPERTIES DATASET...")
properties_df.to_csv(
    'data/processed/properties_with_subjects.csv', index=False)
properties_df.to_pickle('data/processed/properties_with_subjects.pkl')

print("✅ Saved corrected datasets:")
print("   📁 data/processed/properties_with_subjects.csv")
print("   📁 data/processed/properties_with_subjects.pkl")

# Summary report
print(f"\n🎉 PROPERTIES EXTRACTION COMPLETED!")
print("=" * 50)
print(f"✅ {len(properties_df)} properties extracted")
print(f"✅ {properties_df['subject_id'].nunique()} subjects linked")
print(f"✅ {len(duplicates)} duplicate addresses flagged")
print(f"✅ Subject associations preserved")
print(f"✅ Ready for advanced cleaning & modeling!")

# Show sample with subject associations
print(f"\n📋 SAMPLE DATA WITH SUBJECT ASSOCIATIONS:")
sample_cols = ['property_id', 'subject_id', 'orderID',
               'address', 'city', 'close_price', 'gla_sqft']
print(properties_df[sample_cols].head(10).to_string(index=False))
