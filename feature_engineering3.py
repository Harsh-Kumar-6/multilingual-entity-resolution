import os
import psutil
import time
import gc
import pickle
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------
NUM_CORES = cpu_count()
CACHE_DIR = "normalized_sources"
TEMP_DIR = "output/temp_chunks"

def log(msg):
    print(f"[{datetime.now().strftime('%I:%M:%S %p')}] {msg}", flush=True)

def log_ram_usage(step_name=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    log(f"[RAM] {step_name}: {ram_mb / 1024:.2f} GB ({ram_mb:.0f} MB)")

# ------------------------------------------------------------------------------
# 1. FEATURE ENGINEERING LOGIC (Optimized for RapidFuzz)
# ------------------------------------------------------------------------------
def get_numbers(text):
    return set(re.findall(r'\d+', str(text)))

def get_tokens(text):
    return set(str(text).split())

def _compute_features_for_chunk(df_chunk):
    n1 = df_chunk['name_s1'].tolist()
    n2 = df_chunk['name_cand'].tolist()
    a1 = df_chunk['addr_s1'].tolist()
    a2 = df_chunk['addr_cand'].tolist()

    name_jw = [distance.JaroWinkler.normalized_similarity(x, y) for x, y in zip(n1, n2)]
    name_lev = [fuzz.ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    addr_lev = [fuzz.ratio(x, y) / 100.0 for x, y in zip(a1, a2)]

    num_overlap = []
    name_jaccard = []

    for nx1, nx2, ax1, ax2 in zip(n1, n2, a1, a2):
        tok1, tok2 = get_tokens(nx1), get_tokens(nx2)
        if not tok1 and not tok2:
            name_jaccard.append(0.0)
        else:
            name_jaccard.append(len(tok1 & tok2) / max(len(tok1 | tok2), 1))

        num1, num2 = get_numbers(ax1), get_numbers(ax2)
        if not num1 and not num2:
            num_overlap.append(0.5) 
        else:
            num_overlap.append(len(num1 & num2) / max(len(num1 | num2), 1))

    df_chunk['name_jw'] = np.array(name_jw, dtype=np.float32)
    df_chunk['name_lev'] = np.array(name_lev, dtype=np.float32)
    df_chunk['addr_lev'] = np.array(addr_lev, dtype=np.float32)
    df_chunk['name_jaccard'] = np.array(name_jaccard, dtype=np.float32)
    df_chunk['num_overlap'] = np.array(num_overlap, dtype=np.float32)

    len_n1 = df_chunk['name_s1'].str.len().replace(0, 1)
    len_n2 = df_chunk['name_cand'].str.len().replace(0, 1)
    df_chunk['name_len_ratio'] = (np.minimum(len_n1, len_n2) / np.maximum(len_n1, len_n2)).astype(np.float32)

    df_chunk['name_subset'] = [
        1.0 if (len(x) > 3 and x in y) or (len(y) > 3 and y in x) else 0.0
        for x, y in zip(n1, n2)
    ]
    df_chunk['name_subset'] = df_chunk['name_subset'].astype(np.float32)

    # Drop raw text columns to save memory before returning
    df_chunk.drop(columns=['name_s1', 'addr_s1', 'name_cand', 'addr_cand'], inplace=True)
    return df_chunk

def build_features(pairs_df, s1_text, s23_text):
    # Merge strings onto this small chunk
    merged = pd.merge(pairs_df, s1_text, on='source1_entity_id', how='left')
    merged = pd.merge(merged, s23_text, on='candidate_entity_id', how='left')

    sub_chunk_size = (len(merged) // NUM_CORES) + 1
    sub_chunks = [merged.iloc[i * sub_chunk_size : (i + 1) * sub_chunk_size].copy() for i in range(NUM_CORES)]

    del merged
    gc.collect()

    with Pool(NUM_CORES) as p:
        features_df = pd.concat(p.map(_compute_features_for_chunk, sub_chunks), ignore_index=True)

    return features_df

# ------------------------------------------------------------------------------
# 3. HELPER: STREAMING EXPLODE & LABELING
# ------------------------------------------------------------------------------
def process_phase(phase, pairs_file, gt_file=None):
    log(f"\n--- STREAMING FEATURE EXTRACTION FOR {phase.upper()} ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    log("Loading normalized data caches into memory...")
    s1 = pd.read_pickle(f"{CACHE_DIR}/normalized_{phase}_Source_1_v2.pkl")
    s2 = pd.read_pickle(f"{CACHE_DIR}/normalized_{phase}_Source_2_v2.pkl")
    s3 = pd.read_pickle(f"{CACHE_DIR}/normalized_{phase}_Source_3_v2.pkl")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    s1_text = s1[['entity_id', 'business_name', 'business_address']].fillna('')
    s23_text = s23[['entity_id', 'business_name', 'business_address']].fillna('')
    s1_text.columns = ['source1_entity_id', 'name_s1', 'addr_s1']
    s23_text.columns = ['candidate_entity_id', 'name_cand', 'addr_cand']

    # We know the total number of Source 1 entities from our cache
    total_source1_entities = len(s1_text)

    # Pre-load Ground Truth for fast vectorized mapping
    gt_exploded = None
    if gt_file:
        log("Pre-loading Ground Truth...")
        gt_df = pd.read_csv(gt_file, sep='\t', dtype=str).fillna("")
        gt_df['candidate_entity_id'] = gt_df['matched_entity_ids'].str.split(',')
        gt_exploded = gt_df.explode('candidate_entity_id')[['source1_entity_id', 'candidate_entity_id']]
        gt_exploded = gt_exploded[gt_exploded['candidate_entity_id'] != ""]
        gt_exploded['candidate_entity_id'] = gt_exploded['candidate_entity_id'].str.strip()
        gt_exploded['label'] = 1
        gt_exploded = gt_exploded.drop_duplicates()
        del gt_df

    # --- TRUE OUT-OF-CORE STREAMING ---
    CHUNK_SIZE = 300_000  # Read 300k entities from the TSV at a time
    chunk_files = []
    
    total_pairs_processed = 0
    entities_processed = 0

    log(f"Streaming {pairs_file} from disk in chunks of {CHUNK_SIZE} entities (Total: ~{total_source1_entities:,})...")
    t_start = time.time()

    # pd.read_csv(chunksize=...) reads the file piecemeal without loading it all into RAM
    for i, tsv_chunk in enumerate(pd.read_csv(pairs_file, sep='\t', dtype=str, chunksize=CHUNK_SIZE)):
        t0 = time.time()
        
        # Track progress based on TSV rows (each row is 1 Source 1 entity)
        chunk_entities_count = len(tsv_chunk)
        entities_processed += chunk_entities_count

        # 1. Explode this small chunk
        tsv_chunk = tsv_chunk.fillna("")
        tsv_chunk['candidate_entity_id'] = tsv_chunk['candidate_entity_ids'].str.split(',')
        pairs_df = tsv_chunk.explode('candidate_entity_id')[['source1_entity_id', 'candidate_entity_id']]
        pairs_df = pairs_df[pairs_df['candidate_entity_id'] != ""]
        pairs_df['candidate_entity_id'] = pairs_df['candidate_entity_id'].str.strip()
        pairs_df.reset_index(drop=True, inplace=True)
        
        chunk_pairs_count = len(pairs_df)
        total_pairs_processed += chunk_pairs_count

        # 2. Vectorized Labeling (If Train)
        if gt_exploded is not None:
            pairs_df = pd.merge(pairs_df, gt_exploded, on=['source1_entity_id', 'candidate_entity_id'], how='left')
            pairs_df['label'] = pairs_df['label'].fillna(0).astype(np.int8)

        # 3. Build Features
        features_df = build_features(pairs_df, s1_text, s23_text)

        # 4. Save to temporary file & clear RAM completely
        tmp_file = f"{TEMP_DIR}/{phase.lower()}_part_{i}.parquet"
        features_df.to_parquet(tmp_file, index=False)
        chunk_files.append(tmp_file)

        del tsv_chunk, pairs_df, features_df
        gc.collect()

        # --- ETA CALCULATION ---
        elapsed_chunk = time.time() - t0
        elapsed_total = time.time() - t_start
        progress = entities_processed / total_source1_entities
        
        if progress > 0:
            total_estimated_time = elapsed_total / progress
            eta_mins = (total_estimated_time - elapsed_total) / 60
        else:
            eta_mins = 0.0

        log(f"  -> Part {i} | Extracted {chunk_pairs_count:,} pairs in {elapsed_chunk:.1f}s | "
            f"Total pairs: {total_pairs_processed:,} | "
            f"Progress: {progress*100:.1f}% | ETA: {eta_mins:.1f} mins")

    # --- STITCHING IT ALL TOGETHER ---
    out_file = f"output/{phase.lower()}_features.parquet"
    log(f"Finished streaming. Stitching {len(chunk_files)} temporary files into {out_file}...")
    
    # Concat numeric parquets (highly RAM efficient since there are no heavy strings)
    final_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    final_df.to_parquet(out_file, index=False)
    
    log(f"Cleaning up temporary files...")
    for f in chunk_files:
        os.remove(f)

    total_time = (time.time() - t_start) / 60
    log(f"Successfully saved {final_df.shape[0]:,} feature rows in {total_time:.1f} minutes.")
    
    del final_df
    gc.collect()

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    os.makedirs("output", exist_ok=True)
    log("=== ML FEATURE ENGINEERING PIPELINE ===")

    if os.path.exists("output/train_candidates_wide_keys.tsv"):
        if not os.path.exists("output/train_features.parquet"):
            process_phase(
                phase="Train",
                pairs_file="output/train_candidates_wide_keys.tsv",
                gt_file="student_resource/dataset/train/train_ground_truth.tsv"
            )
        else:
            log(f"Cached output/train_features.parquet found, skipping Train phase.")
    else:
        log("WARNING: train candidates not found!")

    if os.path.exists("output/test_candidates_wide_keys.tsv"): 
        if not os.path.exists("output/test_features.parquet"):
            process_phase(
                phase="Test",
                pairs_file="output/test_candidates_wide_keys.tsv", 
                gt_file=None
            )
        else:
            log(f"Cached output/test_features.parquet found, skipping Test phase.")
    else:
        log("WARNING: test candidates not found!")

    log("\n=== FEATURE ENGINEERING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
