import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from geopy.distance import geodesic


def normalize_address(address):
    """Normalize address for better duplicate detection"""
    if pd.isna(address):
        return ''

    addr = str(address).upper().strip()

    # Remove extra spaces
    addr = re.sub(r'\s+', ' ', addr)

    # Standardize common abbreviations
    replacements = {
        r'\bSTREET\b': 'ST',
        r'\bAVENUE\b': 'AVE',
        r'\bDRIVE\b': 'DR',
        r'\bROAD\b': 'RD',
        r'\bBOULEVARD\b': 'BLVD',
        r'\bCOURT\b': 'CT',
        r'\bLANE\b': 'LN',
        r'\bPLACE\b': 'PL',
        r'\bCIRCLE\b': 'CIR',
        r'\bCIRC\b': 'CIR',
        r'\bPARKWAY\b': 'PKWY',
        r'\bTERRACE\b': 'TER',
        r'\bCRESCENT\b': 'CRES',
        r'\bHEIGHTS\b': 'HTS',
        r'\bMOUNT\b': 'MT',
        r'\bSAINT\b': 'ST',
        r'\bNORTH\b': 'N',
        r'\bSOUTH\b': 'S',
        r'\bEAST\b': 'E',
        r'\bWEST\b': 'W'
    }

    for pattern, replacement in replacements.items():
        addr = re.sub(pattern, replacement, addr)

    # Remove common prefixes that might vary
    addr = re.sub(r'^(UNIT\s+\d+\s*-\s*)', '', addr)
    addr = re.sub(r'^(APT\s+\d+\s*-\s*)', '', addr)
    addr = re.sub(r'^(#\d+\s*-\s*)', '', addr)

    return addr.strip()


def extract_unit_info(address):
    """Extract unit information from address"""
    if pd.isna(address):
        return None, address
    
    addr = str(address).upper().strip()
    
    # Look for unit patterns
    unit_patterns = [
        r'UNIT\s+(\d+[A-Z]*)\s*-\s*',
        r'APT\s+(\d+[A-Z]*)\s*-\s*',
        r'#(\d+[A-Z]*)\s*-\s*',
        r'UNIT\s+(\d+[A-Z]*)\s+',
        r'APT\s+(\d+[A-Z]*)\s+',
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, addr)
        if match:
            unit = match.group(1)
            base_addr = re.sub(pattern, '', addr).strip()
            return unit, base_addr
    
    return None, addr


def are_properties_similar_conservative(row1, row2, tolerance_pct=0.15):
    """Ultra-conservative similarity check - requires very strong evidence for duplicates"""
    
    # For condominiums, be extremely lenient (different units in same building are NOT duplicates)
    if (row1.get('structure_type') == 'Condominium' or 
        row2.get('structure_type') == 'Condominium'):
        
        # Extract unit information
        unit1, base_addr1 = extract_unit_info(row1.get('address', ''))
        unit2, base_addr2 = extract_unit_info(row2.get('address', ''))
        
        # If both have units and different unit numbers, NOT duplicates
        if unit1 and unit2 and unit1 != unit2:
            return False
        
        # If same base address but different units, NOT duplicates
        if base_addr1 == base_addr2 and unit1 != unit2:
            return False
        
        # For condos, require EXACT match on multiple criteria
        exact_matches = 0
        if pd.notna(row1['close_price']) and pd.notna(row2['close_price']) and row1['close_price'] == row2['close_price']:
            exact_matches += 1
        if pd.notna(row1['gla_sqft']) and pd.notna(row2['gla_sqft']) and row1['gla_sqft'] == row2['gla_sqft']:
            exact_matches += 1
        if pd.notna(row1['bedrooms_total']) and pd.notna(row2['bedrooms_total']) and row1['bedrooms_total'] == row2['bedrooms_total']:
            exact_matches += 1
        
        # Require at least 2 exact matches for condos to be considered duplicates
        return exact_matches >= 2
    
    # For non-condos, require EXACT price match AND other similarities
    if pd.notna(row1['close_price']) and pd.notna(row2['close_price']):
        if row1['close_price'] != row2['close_price']:
            return False  # No price tolerance for non-condos
    else:
        # If no price data, be very strict on other criteria
        pass
    
    # Check bedrooms - must be exact
    if pd.notna(row1['bedrooms_total']) and pd.notna(row2['bedrooms_total']):
        if row1['bedrooms_total'] != row2['bedrooms_total']:
            return False
    
    # Check GLA - smaller tolerance
    if pd.notna(row1['gla_sqft']) and pd.notna(row2['gla_sqft']):
        gla_diff = abs(row1['gla_sqft'] - row2['gla_sqft']) / max(row1['gla_sqft'], row2['gla_sqft'])
        if gla_diff <= tolerance_pct:
            return True
    
    # If we get here, require exact price match
    if pd.notna(row1['close_price']) and pd.notna(row2['close_price']):
        return row1['close_price'] == row2['close_price']
    
    return False


def ensure_minimum_comparables(df, duplicate_indices, min_comparables=3):
    """Ensure each subject has at least min_comparables properties after duplicate removal"""
    print(f"\n🛡️  PROTECTING SUBJECTS WITH LIMITED DATA:")
    
    protected_count = 0
    
    for subject_id in df['subject_id'].unique():
        subject_group = df[df['subject_id'] == subject_id]
        subject_indices = set(subject_group.index)
        
        # Count how many would remain after duplicate removal
        duplicates_in_subject = subject_indices.intersection(duplicate_indices)
        remaining_count = len(subject_group) - len(duplicates_in_subject)
        
        if remaining_count < min_comparables:
            # Need to protect some duplicates
            needed = min_comparables - remaining_count
            
            # Sort duplicates by some criteria (keep most complete data)
            duplicates_to_protect = []
            for idx in duplicates_in_subject:
                row = df.loc[idx]
                completeness_score = len(row) - row.isnull().sum()
                duplicates_to_protect.append((idx, completeness_score))
            
            # Sort by completeness and protect the best ones
            duplicates_to_protect.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(min(needed, len(duplicates_to_protect))):
                idx_to_protect = duplicates_to_protect[i][0]
                duplicate_indices.discard(idx_to_protect)
                protected_count += 1
            
            print(f"   Subject {subject_id}: Protected {min(needed, len(duplicates_to_protect))} properties (had {len(subject_group)}, would have {remaining_count})")
    
    print(f"   Total properties protected: {protected_count}")
    return duplicate_indices


def detect_improved_duplicates(df):
    """Improved duplicate detection with property-type specific logic and minimum comparable protection"""
    print("🔍 IMPROVED DUPLICATE DETECTION (ULTRA-CONSERVATIVE)")
    print("=" * 60)

    # 1. Address normalization
    df['address_normalized'] = df['address'].apply(normalize_address)

    print(f"📊 Initial Analysis:")
    print(f"   Total properties: {len(df):,}")
    print(f"   Original unique addresses: {df['address'].nunique():,}")
    print(f"   Normalized unique addresses: {df['address_normalized'].nunique():,}")

    # 2. Property type specific duplicate detection
    duplicate_indices = set()
    
    print(f"\n🏠 Property Type Specific Detection:")
    
    # Group by property type for different handling
    property_types = df['structure_type'].value_counts()
    print(f"   Property types: {dict(property_types)}")
    
    total_duplicates_found = 0
    
    for prop_type in property_types.index:
        if pd.isna(prop_type):
            continue
            
        type_df = df[df['structure_type'] == prop_type].copy()
        type_duplicates = 0
        
        print(f"\n   Processing {prop_type} ({len(type_df)} properties):")
        
        # Group by normalized address within property type
        for address, group in type_df.groupby('address_normalized'):
            if len(group) > 1:
                # Convert to list for easier indexing
                group_list = list(group.iterrows())
                
                # Check each pair within the same address group
                for i in range(len(group_list)):
                    for j in range(i + 1, len(group_list)):
                        idx1, row1 = group_list[i]
                        idx2, row2 = group_list[j]
                        
                        # Use conservative similarity check
                        if are_properties_similar_conservative(row1, row2):
                            # Keep the one with more complete data or lower index
                            if row1.isnull().sum() <= row2.isnull().sum():
                                duplicate_indices.add(idx2)  # Remove second one
                            else:
                                duplicate_indices.add(idx1)  # Remove first one
                            type_duplicates += 1
                            
                            print(f"     Duplicate: {row1['address']} vs {row2['address']}")
        
        print(f"     Found {type_duplicates} duplicates in {prop_type}")
        total_duplicates_found += type_duplicates

    # 3. Geographic duplicates (more conservative)
    print(f"\n🌍 Geographic Similarity Detection (Conservative):")
    geo_duplicates_found = 0

    for subject_id in df['subject_id'].unique():
        subject_group = df[df['subject_id'] == subject_id].copy()

        if len(subject_group) < 2:
            continue

        # Check each pair for potential duplicates
        for i, row1 in subject_group.iterrows():
            for j, row2 in subject_group.iterrows():
                if i >= j or i in duplicate_indices or j in duplicate_indices:
                    continue

                # Check if they're very close and very similar
                if (pd.notna(row1['latitude']) and pd.notna(row1['longitude']) and
                        pd.notna(row2['latitude']) and pd.notna(row2['longitude'])):

                    try:
                        distance = geodesic(
                            (row1['latitude'], row1['longitude']),
                            (row2['latitude'], row2['longitude'])
                        ).meters

                        # Ultra-restrictive geographic duplicates
                        if (distance < 10 and  # Within 10m (extremely close)
                            are_properties_similar_conservative(row1, row2, tolerance_pct=0.02)):  # Nearly identical
                            duplicate_indices.add(j)  # Remove second one
                            geo_duplicates_found += 1
                            print(f"   Geographic duplicate: {distance:.0f}m apart")
                            print(f"     {row1['address']} vs {row2['address']}")

                    except:
                        continue

    print(f"   Geographic duplicates found: {geo_duplicates_found}")

    # 4. PROTECT SUBJECTS WITH LIMITED COMPARABLE DATA
    duplicate_indices = ensure_minimum_comparables(df, duplicate_indices, min_comparables=3)

    # Mark duplicates in dataframe
    df['is_improved_duplicate'] = False
    df.loc[list(duplicate_indices), 'is_improved_duplicate'] = True

    print(f"\n📋 ULTRA-CONSERVATIVE DUPLICATE DETECTION SUMMARY:")
    print(f"   Total duplicates found: {df['is_improved_duplicate'].sum():,}")
    print(f"   Properties to keep: {(~df['is_improved_duplicate']).sum():,}")
    print(f"   Duplicate rate: {df['is_improved_duplicate'].mean()*100:.1f}%")
    
    # Compare with original detection
    if 'is_advanced_duplicate' in df.columns:
        original_duplicates = df['is_advanced_duplicate'].sum()
        new_duplicates = df['is_improved_duplicate'].sum()
        print(f"   Original detection: {original_duplicates:,} duplicates")
        print(f"   Improved detection: {new_duplicates:,} duplicates")
        print(f"   Difference: {original_duplicates - new_duplicates:,} fewer duplicates removed")

    # Subject-level analysis
    print(f"\n📊 SUBJECT-LEVEL ANALYSIS:")
    df_clean = df[~df['is_improved_duplicate']]
    subject_counts = df_clean.groupby('subject_id').size()
    print(f"   Subjects with ≥3 comparables: {(subject_counts >= 3).sum()}/{len(subject_counts)}")
    print(f"   Subjects with <3 comparables: {(subject_counts < 3).sum()}")
    if (subject_counts < 3).any():
        limited_subjects = subject_counts[subject_counts < 3]
        print(f"   Limited subjects: {dict(limited_subjects)}")

    return df


def remove_duplicates_and_save_improved(df):
    """Remove duplicates and save improved cleaned dataset"""
    print(f"\n💾 REMOVING DUPLICATES AND SAVING (IMPROVED)...")

    # Keep only non-duplicates
    df_clean = df[~df['is_improved_duplicate']].copy()

    # Drop the duplicate detection columns
    columns_to_drop = ['address_normalized', 'is_improved_duplicate']
    if 'is_advanced_duplicate' in df_clean.columns:
        columns_to_drop.append('is_advanced_duplicate')
    
    df_clean = df_clean.drop(columns_to_drop, axis=1, errors='ignore')

    print(f"✅ Improved cleaned dataset:")
    print(f"   Original: {len(df):,} properties")
    print(f"   After improved duplicate removal: {len(df_clean):,} properties")
    print(f"   Removed: {len(df) - len(df_clean):,} duplicates")
    
    # Check subject coverage
    subjects_covered = df_clean['subject_id'].nunique()
    print(f"   Subjects with comparables: {subjects_covered}")

    # Save improved cleaned dataset
    df_clean.to_csv('data/processed/properties_deduplicated.csv', index=False)

    return df_clean


if __name__ == "__main__":
    # Load the current dataset (before any duplicate removal)
    df = pd.read_csv('data/processed/properties_cleaned_with_subjects.csv')

    # Detect duplicates with improved algorithm
    df = detect_improved_duplicates(df)

    # Remove duplicates and save
    df_clean = remove_duplicates_and_save_improved(df)

    print(f"\n🎉 IMPROVED DUPLICATE REMOVAL COMPLETE!")
    print(f"📁 New files saved:")
    print(f"   • properties_deduplicated.csv") 