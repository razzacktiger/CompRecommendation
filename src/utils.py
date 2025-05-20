'''Utility functions for data parsing, cleaning, and calculations.'''
import pandas as pd
import numpy as np
from datetime import datetime
import re


def parse_date(date_str):
    """Parse date string in various formats."""
    if not date_str or pd.isna(date_str):
        return None
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S.%fZ",
        "%b/%d/%Y", "%b/%d/%y", "%m/%d/%y", "%Y/%m/%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except (ValueError, TypeError):
            continue
    return None


def safe_float(value, default=np.nan):
    """Safely convert value to float, return default (np.nan) if conversion fails."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return default


def safe_int(value, default=np.nan):
    """Safely convert value to int, return default (np.nan) if conversion fails."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).replace(",", "").replace("$", "")))
    except (ValueError, TypeError):
        return default


def standardize_baths(bath_str, full_baths_val=None, half_baths_val=None):
    """Standardizes bathroom counts to a single float (e.g., 2.5)."""
    if full_baths_val is not None:
        total_baths = safe_float(full_baths_val, 0.0) + \
            (safe_float(half_baths_val, 0.0) * 0.5)
        return total_baths if not pd.isna(total_baths) else np.nan
    if bath_str is None or pd.isna(bath_str):
        return np.nan
    bath_str = str(bath_str).strip()
    if ':' in bath_str:
        parts = bath_str.split(':')
        try:
            full = safe_float(parts[0], 0.0)
            half = safe_float(parts[1], 0.0)
            return full + (half * 0.5)
        except (IndexError, ValueError):
            return np.nan
    else:
        return safe_float(bath_str, np.nan)


def calculate_age(year_built_val, reference_date_val, min_age=0, max_age=120):
    """Calculates age given year_built and a reference_date (datetime object), capped within a range."""
    year_built = safe_int(year_built_val)
    if pd.isna(year_built) or reference_date_val is None:
        return np.nan
    age = reference_date_val.year - year_built
    if pd.isna(age):
        return np.nan
    return np.clip(age, min_age, max_age)


def get_fsa(address_or_postal_str):
    """Extracts the Forward Sortation Area (FSA - first 3 chars of Canadian Postal Code)."""
    if not address_or_postal_str or pd.isna(address_or_postal_str):
        return None
    match = re.search(r'\b([A-Z]\d[A-Z])(?:\s*\d[A-Z]\d)?\b',
                      str(address_or_postal_str).upper())
    if match:
        return match.group(1)
    return None


def standardize_address_text(address_text):
    """Standardizes an address string for better matching."""
    if address_text is None or pd.isna(address_text):
        return ""
    text = str(address_text).lower().strip()
    abbreviations = {
        r'\bstreets?\b': 'street', r'\bstr?\b': 'street',
        r'\broads?\b': 'road', r'\brd\b': 'road',
        r'\bavenues?\b': 'avenue', r'\bave?\b': 'avenue',
        r'\bdrives?\b': 'drive', r'\bdr\b': 'drive',
        r'\bcrescents?\b': 'crescent', r'\bcres\b': 'crescent',
        r'\blanes?\b': 'lane', r'\bln\b': 'lane',
        r'\bboulevards?\b': 'boulevard', r'\bblvd\b': 'boulevard',
        r'\bcourts?\b': 'court', r'\bct\b': 'court',
        r'\bplaces?\b': 'place', r'\bpl\b': 'place',
        r'\bterraces?\b': 'terrace', r'\bterr\b': 'terrace',
        r'\bsquares?\b': 'square', r'\bsq\b': 'square',
        r'\bpaths?\b': 'path', r'\bpt\b': 'path',
        r'\bways?\b': 'way', r'\bwy\b': 'way',
        r'\bcircus?\b': 'circus', r'\bcir\b': 'circus', r'\bcirc\b': 'circus',
        r'\bheights?\b': 'heights', r'\bhts\b': 'heights',
        r'\bgardens?\b': 'gardens', r'\bgdns\b': 'gardens',
        r'\bgroves?\b': 'grove', r'\bgr\b': 'grove',
        r'\bnorth\b': 'n', r'\bsouth\b': 's', r'\beast\b': 'e', r'\bwest\b': 'w',
        r'\bn e\b': 'ne', r'\bn w\b': 'nw', r'\bs e\b': 'se', r'\bs w\b': 'sw',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[.,#\'"!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
