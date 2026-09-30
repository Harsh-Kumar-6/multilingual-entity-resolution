import os
import psutil
import gc
import re
import time
import pickle
import unicodedata
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from indic_transliteration import sanscript, detect

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------
NUM_CORES = cpu_count()
CACHE_DIR = "normalized_sources"

# Define the set of Indic scripts to detect and transliterate
INDIC_SCRIPTS = {
    sanscript.BENGALI, sanscript.DEVANAGARI, sanscript.GUJARATI, 
    sanscript.GURMUKHI, sanscript.KANNADA, sanscript.MALAYALAM, 
    sanscript.ORIYA, sanscript.TAMIL, sanscript.TELUGU
}

def log(msg):
    print(f"[{datetime.now().strftime('%I:%M:%S %p')}] {msg}", flush=True)

# ------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# ------------------------------------------------------------------------------
def normalize_string(text: str) -> str:
    # 1. Handle NaNs and non-strings safely
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 2. Pre-normalize Indic text so transliteration doesn't break on split characters
    text = unicodedata.normalize("NFC", text)

    # 3. If Indian language, transliterate first
    detected_script = detect.detect(text)
    if detected_script and detected_script in INDIC_SCRIPTS:
        text = sanscript.transliterate(text, detected_script, sanscript.HK)

    # 4. Standardize all resulting Latin text 
    text = unicodedata.normalize("NFKC", text).lower()

    # 4.5 THE FIX: REMOVE ACCENTS (Diacritics) for French / European data
    text = unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('utf-8')

    # 5. Remove Entity and Address Stopwords
    text = re.sub(r"\b(pvt|private|ltd|limited|corp|corporation|inc|incorporated)\b", " ", text)
    text = re.sub(r"\b(rd|road|st|street|ave|avenue|dr|drive|bldg|building)\b", " ", text)

    # 6. Remove Punctuation safely 
    text = re.sub(r"[^\w\s]", " ", text)

    # 7. Clean up redundant spaces and return
    return re.sub(r"\s+", " ", text).strip()

def _parallel_clean(series_vals):
    return [normalize_string(x) for x in series_vals]

# ------------------------------------------------------------------------------
# VISUALIZATION (Original vs Normalized)
# ------------------------------------------------------------------------------
def demonstrate_normalization(tsv_path: str):
    """Prints 5 RANDOM rows each for INDIA, US, and FRANCE showing full text Original vs Normalized."""
    print(f"\n{'='*100}")
    print(f" VISUALIZING NORMALIZATION FOR: {tsv_path}")
    print(f"{'='*100}")
    
    if not os.path.exists(tsv_path):
        print(f"File not found: {tsv_path}")
        return

    df = pd.read_csv(tsv_path, sep="\t")
    df["country_clean"] = df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    for country in ['INDIA', 'US', 'FRANCE']:
        country_df = df[df["country_clean"] == country]
        
        if country_df.empty:
            print(f"\n--- No data found for {country} in this file ---")
            continue
            
        # Get 5 random rows (or fewer if the dataframe has less than 5 rows)
        sample = country_df.sample(n=min(5, len(country_df)))
        print(f"\n{' '*30}--- RANDOM 5 ROWS FOR: {country} ---")
        
        for _, row in sample.iterrows():
            orig_name = str(row['business_name'])
            orig_addr = str(row['business_address'])
            
            # Apply normalizer purely for printing
            norm_name = normalize_string(orig_name)
            norm_addr = normalize_string(orig_addr)
            
            # Print full text on separate lines for readability
            print(f"ORIGINAL   | Name: {orig_name}")
            print(f"           | Addr: {orig_addr}")
            print(f"NORMALIZED | Name: {norm_name}")
            print(f"           | Addr: {norm_addr}")
            print("-" * 100)
    print("\n")

# ------------------------------------------------------------------------------
# DATA LOADING & PARALLEL PROCESSING
# ------------------------------------------------------------------------------
def load_and_clean_data(filepath: str, name: str) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"normalized_{name.replace(' ', '_')}_v2.pkl")
    
    if os.path.exists(cache_path):
        log(f"Loading cached {cache_path}")
        return pd.read_pickle(cache_path)

    log(f"No cache named {cache_path} found.")
    log(f"Normalizing '{name}' using {NUM_CORES} cores...")
    
    t0 = time.time()
    df = pd.read_csv(filepath, sep="\t")

    chunks = np.array_split(df["business_name"].values, NUM_CORES)
    with Pool(NUM_CORES) as p:
        df["business_name"] = [item for sublist in p.map(_parallel_clean, chunks) for item in sublist]

    chunks = np.array_split(df["business_address"].values, NUM_CORES)
    with Pool(NUM_CORES) as p:
        df["business_address"] = [item for sublist in p.map(_parallel_clean, chunks) for item in sublist]

    df["country"] = df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["full_text"] = df["business_name"] + " " + df["business_address"]
    
    log(f"Completed in {time.time() - t0:.2f}s. Saving to {cache_path}...")
    df.to_pickle(cache_path)
    return df

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    log("=== STARTING DATA NORMALIZATION (v2 - Accent Cleaned) ===")

    # 1. RUN THE VISUALIZER FIRST (Using Test Source 1 as it contains all 3 countries)
    test_s1_path = "student_resource/dataset/test/test_source2.tsv"
    if os.path.exists(test_s1_path):
        demonstrate_normalization(test_s1_path)

    # 2. RUN THE PARALLEL NORMALIZATION ON ALL FILES
    files_to_process = [
        ("student_resource/dataset/train/train_source1.tsv", "Train_Source_1"),
        ("student_resource/dataset/train/train_source2.tsv", "Train_Source_2"),
        ("student_resource/dataset/train/train_source3.tsv", "Train_Source_3"),
        ("student_resource/dataset/test/test_source1.tsv", "Test_Source_1"),
        ("student_resource/dataset/test/test_source2.tsv", "Test_Source_2"),
        ("student_resource/dataset/test/test_source3.tsv", "Test_Source_3")
    ]

    for filepath, name in files_to_process:
        if os.path.exists(filepath):
            load_and_clean_data(filepath, name)
        else:
            log(f"WARNING: File not found: {filepath}. Skipping.")

    log("=== DATA NORMALIZATION COMPLETE ===")
    log("You now have fresh '_v2.pkl' files ready for the wide-net Blocking phase!")

if __name__ == "__main__":
    main()
