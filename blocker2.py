mport os
import psutil
import gc
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Suppress pandas regex warnings for clean logs
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------
NUM_CORES = cpu_count()
CACHE_DIR = "normalized_sources"
GROUND_TRUTH_PATH = "student_resource/dataset/train/train_ground_truth.tsv"

def log(msg):
    print(f"[{datetime.now().strftime('%I:%M:%S %p')}] {msg}", flush=True)

def log_ram_usage(step_name=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    log(f"[RAM] {step_name}: {ram_mb / 1024:.2f} GB ({ram_mb:.0f} MB)")

# ------------------------------------------------------------------------------
# EVALUATION MODULE
# ------------------------------------------------------------------------------
def evaluate_blocking(candidates_df, ground_truth_path):
    """
    Evaluates the Blocking phase against the ground truth.
    Remember: Good blocking has extremely HIGH Recall (>0.98) and LOW Precision (0.01 to 0.15).
    """
    if not os.path.exists(ground_truth_path):
        log(f"Warning: Ground truth not found at {ground_truth_path}. Skipping evaluation.")
        return

    log(f"Loading Ground Truth from {ground_truth_path} for evaluation...")
    gt = pd.read_csv(ground_truth_path, sep='\t')

    def to_set(x):
        if pd.isna(x) or str(x).strip() == '':
            return set()
        return set(str(x).split(','))

    gt['matched_set'] = gt['matched_entity_ids'].apply(to_set)

    merged = pd.merge(
        candidates_df,
        gt[['source1_entity_id', 'matched_set']],
        on='source1_entity_id',
        how='left'
    )

    merged['matched_set'] = merged['matched_set'].apply(lambda x: x if isinstance(x, set) else set())
    merged['candidate_set'] = merged['candidate_entity_ids'].apply(to_set)

    log("Calculating TP, FP, and FN...")
    merged['TP'] = merged.apply(lambda r: len(r['candidate_set'].intersection(r['matched_set'])), axis=1)
    merged['FP'] = merged.apply(lambda r: len(r['candidate_set'] - r['matched_set']), axis=1)
    merged['FN'] = merged.apply(lambda r: len(r['matched_set'] - r['candidate_set']), axis=1)

    total_TP = merged['TP'].sum()
    total_FP = merged['FP'].sum()
    total_FN = merged['FN'].sum()

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    if (0.25 * precision + recall) == 0:
        f05 = 0
    else:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

    print("\n" + "="*50)
    print("=== BLOCKING PHASE EVALUATION RESULTS ===")
    print("="*50)
    print(f"Total True Positives (Matches Found) : {total_TP:,}")
    print(f"Total False Positives (Garbage Added): {total_FP:,}  <-- Needs to be much higher!")
    print(f"Total False Negatives (Matches Missed): {total_FN:,}  <-- Needs to be near zero!\n")
    print(f"Recall    : {recall:.4f}  (Aim for > 0.98)")
    print(f"Precision : {precision:.4f}  (Aim for 0.05 to 0.15)")
    print(f"F_0.5     : {f05:.4f}")
    print("="*50 + "\n")

# ------------------------------------------------------------------------------
# 1. PARALLEL BLOCKING KEY GENERATION (THE ULTIMATE 15 KEYS)
# ------------------------------------------------------------------------------
def _generate_keys_for_chunk(df_chunk):
    b_name = df_chunk['business_name'].fillna('').astype(str).str.lower()
    b_addr = df_chunk['business_address'].fillna('').astype(str).str.lower()
    country = df_chunk['country'].astype(str)

    # Clean URL suffixes and noise words
    b_name = b_name.str.replace(r'\b(com|net|org)\b', '', regex=True)
    name_no_stops = b_name.str.replace(r'\b(the|aka|nee|llc|ltd|pvt|inc|co|group|m s|dba)\b', '', regex=True)

    name_clean = name_no_stops.str.replace(r'[^a-z0-9]', '', regex=True)
    name_alpha = name_no_stops.str.replace(r'[^a-z]', '', regex=True) 
    addr_alpha = b_addr.str.replace(r'[^a-z]', '', regex=True)

    name_words = name_no_stops.str.replace(r'[^a-z0-9]', ' ', regex=True).str.split()
    name_first_clean = name_words.str[0].fillna('')
    name_last = name_words.str[-1].fillna('')
    name_sorted_first = name_words.apply(lambda x: sorted(x)[0] if isinstance(x, list) and len(x) > 0 else '')

    addr_no_stops = b_addr.str.replace(r'\b(plot|door|shop|no|flat|unit|floor|opp|near|c o|room|bldg|sector|phase|st|rd|ave|lane)\b', '', regex=True)
    addr_clean_words = addr_no_stops.str.replace(r'[^a-z0-9]', ' ', regex=True).str.split()
    addr_clean_first = addr_clean_words.str[0].fillna('')
    addr_clean_longest = addr_clean_words.apply(lambda x: max(x, key=len) if isinstance(x, list) and len(x) > 0 else '')
    addr_clean_cons = addr_no_stops.str.replace(r'[^a-z]', '', regex=True).str.replace(r'[aeiou]', '', regex=True)

    all_nums = b_addr.str.findall(r'\d+')
    first_num = all_nums.str[0].fillna('').str.lstrip('0')
    last_num = all_nums.str[-1].fillna('').str.lstrip('0')
    addr_longest_num = all_nums.apply(lambda x: max(x, key=len) if isinstance(x, list) and len(x) > 0 else '')

    name_cons = name_alpha.str.replace(r'[aeiou]', '', regex=True)
    addr_cons = addr_alpha.str.replace(r'[aeiou]', '', regex=True)

    # ================= THE 15 ARSENAL KEYS =================
    key1 = np.where(name_alpha.str.len() >= 5, country + "_k1_" + name_alpha.str[:10], np.nan)
    key2 = np.where((name_sorted_first != '') & (first_num != ''), country + "_k2_" + name_sorted_first + "_" + first_num, np.nan)
    key3 = np.where((first_num != '') & (addr_clean_longest.str.len() >= 5), country + "_k3_" + first_num + "_" + addr_clean_longest, np.nan)
    key4 = np.where((name_last != '') & (last_num != ''), country + "_k4_" + name_last + "_" + last_num, np.nan)
    key5 = np.where(addr_alpha.str.len() >= 10, country + "_k5_" + addr_alpha.str[:12], np.nan)
    key6 = np.where((first_num != '') & (last_num != '') & (first_num != last_num), country + "_k6_" + first_num + "_" + last_num, np.nan)
    key7 = np.where((name_cons.str.len() >= 4) & (addr_cons.str.len() >= 4), country + "_k7_" + name_cons.str[:5] + "_" + addr_cons.str[:5], np.nan)
    key8 = np.where((first_num != '') & (addr_clean_first != ''), country + "_k8_" + first_num + "_" + addr_clean_first, np.nan)
    key9 = np.where((name_cons.str[:1] != '') & (addr_clean_longest.str.len() >= 4), country + "_k9_" + name_cons.str[:1] + "_" + addr_clean_longest, np.nan)
    key10 = np.where(name_last.str.len() >= 4, country + "_k10_" + name_last.str[:5], np.nan)
    key11 = np.where(name_alpha.str.len() >= 6, country + "_k11_" + name_alpha.str[1:7], np.nan)
    key12 = np.where((name_cons.str.len() >= 3) & (addr_clean_cons.str.len() >= 3), country + "_k12_" + name_cons.str[:3] + "_" + addr_clean_cons.str[:3], np.nan)
    key13 = np.where(name_first_clean.str.len() >= 5, country + "_k13_" + name_first_clean, np.nan)
    key14 = np.where((addr_longest_num != '') & (addr_clean_longest.str.len() >= 4), country + "_k14_" + addr_longest_num + "_" + addr_clean_longest, np.nan)
    key15 = np.where(addr_clean_longest.str.len() >= 10, country + "_k15_" + addr_clean_longest, np.nan)

    # Assign to dataframe chunk safely
    keys = [key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15]
    for i, k in enumerate(keys, 1):
        df_chunk[f'key{i}'] = k

    return df_chunk

def apply_parallel_key_generation(df):
    log(f"Generating blocking keys in parallel using {NUM_CORES} cores for {df.shape[0]:,} rows...")
    t0 = time.time()

    chunk_size = (len(df) // NUM_CORES) + 1
    chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size].copy() for i in range(NUM_CORES)]

    with Pool(NUM_CORES) as p:
        result = pd.concat(p.map(_generate_keys_for_chunk, chunks))

    eta_mins = (time.time() - t0) / 60
    log(f"Key generation complete in {eta_mins:.2f} mins. Output shape: {result.shape}")
    return result

# ------------------------------------------------------------------------------
# 2. VECTORIZED HASH BLOCKING (RAM-OPTIMIZED & FREQUENCY CAPPED)
# ------------------------------------------------------------------------------
def run_hash_blocking(df_s1, df_s23, phase_name, max_block_size=3000):
    log(f"Starting RAM-Optimized Hash Blocking for {phase_name}...")

    log("Mapping string entity_ids to int32 to save RAM...")
    all_unique_ids = pd.concat([df_s1['entity_id'], df_s23['entity_id']]).unique()
    id_to_int = {str_id: i for i, str_id in enumerate(all_unique_ids)}
    int_to_id = {i: str_id for str_id, i in id_to_int.items()}

    df_s1['int_id'] = df_s1['entity_id'].map(id_to_int).astype(np.int32)
    df_s23['int_id'] = df_s23['entity_id'].map(id_to_int).astype(np.int32)
    
    del all_unique_ids
    gc.collect()

    all_pairs = []
    
    # Loop now checks all 15 keys
    for key_idx in [f'key{i}' for i in range(1, 16)]:
        s1_valid = df_s1.dropna(subset=[key_idx])[['int_id', key_idx]]
        s23_valid = df_s23.dropna(subset=[key_idx])[['int_id', key_idx]]

        s1_counts = s1_valid[key_idx].value_counts()
        s23_counts = s23_valid[key_idx].value_counts()

        valid_keys_s1 = s1_counts[s1_counts <= max_block_size].index
        valid_keys_s23 = s23_counts[s23_counts <= max_block_size].index
        
        safe_keys = valid_keys_s1.intersection(valid_keys_s23)

        dropped_s1 = len(s1_counts) - len(valid_keys_s1)
        log(f"[{key_idx}] Dropped {dropped_s1:,} Super-Node keys exceeding {max_block_size} entities.")

        s1_filtered = s1_valid[s1_valid[key_idx].isin(safe_keys)]
        s23_filtered = s23_valid[s23_valid[key_idx].isin(safe_keys)]

        merged = pd.merge(s1_filtered, s23_filtered, on=key_idx, how='inner', suffixes=('_s1', '_cand'))
        pairs = merged[['int_id_s1', 'int_id_cand']]
        all_pairs.append(pairs)
        
        del s1_valid, s23_valid, s1_counts, s23_counts, s1_filtered, s23_filtered, merged, pairs
        gc.collect()

    log(f"Concatenating and deduplicating integer pairs...")
    final_pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    
    del all_pairs
    gc.collect()
    
    log(f"Total Unique Candidate Pairs Generated: {len(final_pairs):,}")

    log("Mapping int32 back to string IDs for submission format...")
    final_pairs['entity_id_s1'] = final_pairs['int_id_s1'].map(int_to_id)
    final_pairs['entity_id_cand'] = final_pairs['int_id_cand'].map(int_to_id)
    
    final_pairs.drop(columns=['int_id_s1', 'int_id_cand'], inplace=True)

    log(f"Grouping candidate pairs into lists...")
    grouped = final_pairs.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ','.join(x)).reset_index()
    grouped.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_ids'}, inplace=True)

    s1_all = df_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    final_output = pd.merge(s1_all, grouped, on='source1_entity_id', how='left')
    final_output['candidate_entity_ids'] = final_output['candidate_entity_ids'].fillna("")

    return final_output

# ------------------------------------------------------------------------------
# 3. DATA LOADING HELPERS
# ------------------------------------------------------------------------------
def load_cached_data(phase, source_num):
    path = os.path.join(CACHE_DIR, f"normalized_{phase}_Source_{source_num}_v2.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        raise FileNotFoundError(f"Cache file {path} not found.")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    os.makedirs("output", exist_ok=True)
    log("=== MULTI-PASS HEURISTIC BLOCKING PIPELINE ===")

    # ==========================================
    # PHASE 1: TRAIN DATA BLOCKING
    # ==========================================
    log("\n--- PROCESSING TRAIN DATA ---")
    train_s1 = load_cached_data("Train", "1")
    train_s2 = load_cached_data("Train", "2")
    train_s3 = load_cached_data("Train", "3")
    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)
    del train_s2, train_s3
    gc.collect()

    train_s1 = apply_parallel_key_generation(train_s1)
    train_s23 = apply_parallel_key_generation(train_s23)

    train_candidates = run_hash_blocking(train_s1, train_s23, "TRAIN")

    # --- EVALUATE BLOCKING QUALITY ---
    evaluate_blocking(train_candidates, GROUND_TRUTH_PATH)

    # UPDATED OUTPUT FILE NAME
    out_path_train = "output/train_candidates_wide_keys.tsv"
    train_candidates.to_csv(out_path_train, sep='\t', index=False)
    log(f"Saved Train Candidates to {out_path_train}")

    del train_s1, train_s23, train_candidates
    gc.collect()

    # ==========================================
    # PHASE 2: TEST DATA BLOCKING
    # ==========================================
    log("\n--- PROCESSING TEST DATA ---")
    try:
        test_s1 = load_cached_data("Test", "1")
        test_s2 = load_cached_data("Test", "2")
        test_s3 = load_cached_data("Test", "3")

        test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
        del test_s2, test_s3
        gc.collect()

        test_s1 = apply_parallel_key_generation(test_s1)
        test_s23 = apply_parallel_key_generation(test_s23)

        test_candidates = run_hash_blocking(test_s1, test_s23, "TEST")

        # UPDATED OUTPUT FILE NAME
        out_path_test = "output/test_candidates_wide_keys.tsv"
        test_candidates.to_csv(out_path_test, sep='\t', index=False)
        log(f"Saved Test Candidates to {out_path_test}")

    except FileNotFoundError as e:
        log(f"Skipping Test Blocking: {e}")

    log("\n=== BLOCKING SCRIPT COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
