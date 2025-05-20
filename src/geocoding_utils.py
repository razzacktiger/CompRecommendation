'''Utility functions for geocoding addresses.'''
import time
import json
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import config  # For GEOCODING_CACHE_FILE

# --- Geocoding Setup ---
geolocator = Nominatim(user_agent="comp_recommender_app_v1")


def load_geocoding_cache():
    """Loads the geocoding cache from a JSON file."""
    try:
        with open(config.GEOCODING_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(
            f"Loaded geocoding cache from {config.GEOCODING_CACHE_FILE} with {len(cache)} entries.")
        return cache
    except FileNotFoundError:
        print(
            f"Geocoding cache file ({config.GEOCODING_CACHE_FILE}) not found. Starting with an empty cache.")
        return {}
    except json.JSONDecodeError:
        print(
            f"Error decoding JSON from {config.GEOCODING_CACHE_FILE}. Starting with an empty cache.")
        return {}


def save_geocoding_cache(cache):
    """Saves the geocoding cache to a JSON file."""
    try:
        with open(config.GEOCODING_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
        print(
            f"Saved geocoding cache to {config.GEOCODING_CACHE_FILE} with {len(cache)} entries.")
    except IOError as e:
        print(
            f"Error saving geocoding cache to {config.GEOCODING_CACHE_FILE}: {e}")


def geocode_address(address_str, cache):
    """Geocodes an address string with caching and rate limiting.
    Uses the global 'geolocator' and updates the provided 'cache'.
    """
    if not address_str or pd.isna(address_str):
        return None, None
    address_str = str(address_str).strip()
    if not address_str:
        return None, None

    if address_str in cache:
        # print(f"Cache hit for: {address_str}") # Optional: for debugging cache hits
        return cache[address_str]

    # print(f"Geocoding: {address_str}...") # Moved print to main script for less verbose util
    try:
        time.sleep(1)  # Nominatim usage policy: 1 request per second
        location = geolocator.geocode(address_str, timeout=10)
        if location:
            cache[address_str] = (location.latitude, location.longitude)
            return location.latitude, location.longitude
        else:
            # Cache failure to avoid re-querying
            cache[address_str] = (None, None)
            return None, None
    except GeocoderTimedOut:
        print(f"Geocoder timed out for address: {address_str}")
        cache[address_str] = (None, None)
        return None, None
    except GeocoderUnavailable as e:
        print(f"Geocoder service unavailable for address: {address_str}: {e}")
        cache[address_str] = (None, None)
        return None, None
    except Exception as e:
        print(
            f"An unexpected error occurred during geocoding for {address_str}: {e}")
        cache[address_str] = (None, None)
        return None, None
