import os
import pickle
from datetime import datetime
import pandas as pd
import xgboost as xgb

def log(msg):
    print(f"[{datetime.now().strftime('%I:%M:%S %p')}] {msg}", flush=True)

def main():
    log("=== RUNNING INFERENCE ON TEST SET ===")
    
    # 1. Load the Model and Metadata
    meta_path = "model_artifacts/model_meta.pkl"
    model_path = "model_artifacts/xgb_final_model.json"
    
    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        log("ERROR: Model artifacts not found. Run train.py first.")
        return

    log("Loading XGBoost model and optimal threshold...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    best_thresh = meta["best_threshold"]
    feature_cols = meta["features"]
    log(f"Optimal Threshold to apply: {best_thresh}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 2. Load the Test Features
    test_features_path = "output/test_features.parquet"
    if not os.path.exists(test_features_path):
        log("ERROR: test_features.parquet not found. Please run feature_engineer.py to generate test features.")
        return
        
    log("Loading Test Candidate features...")
    df_test = pd.read_parquet(test_features_path)
    X_test = df_test[feature_cols]

    # 3. Predict Probabilities
    log("Scoring candidate pairs...")
    probs = model.predict_proba(X_test)[:, 1]
    
    # 4. Apply Threshold
    # Only keep candidates where the model is confident it's a match
    df_test['is_match'] = (probs >= best_thresh).astype(int)
    
    # Filter out the false positives
    matches_df = df_test[df_test['is_match'] == 1][['source1_entity_id', 'candidate_entity_id']]
    
    # 5. Format for Submission
    log("Formatting matches per Source 1 entity...")
    # Group by S1 and join the candidate IDs with commas
    grouped_matches = matches_df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    grouped_matches.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    
    # 6. Ensure ALL Source 1 Entities are present (The "Singleton" Rule)
    # The competition strictly requires every S1 ID from the test set to be in the final output.
    log("Injecting singletons (entities with no matches)...")
    s1_test_raw = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
    s1_test_raw.rename(columns={"entity_id": "source1_entity_id"}, inplace=True)
    
    # Left merge ensures every S1 ID is kept. Those without matches will get NaN
    final_output = pd.merge(s1_test_raw, grouped_matches, on="source1_entity_id", how="left")
    
    # Fill NaN with empty string
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna("")
    
    # 7. Save Final Submission
    out_file = "output/matching_results.tsv"
    final_output.to_csv(out_file, sep="\t", index=False)
    log(f"SUCCESS: Final matches saved to {out_file}")
    
    log("\n=== PIPELINE COMPLETE ===")
    log("Your final submission files are ready:")
    log("1. output/matching_results.tsv")
    log("2. output/candidate_pairs.tsv")
    log("\nYou can now run the validation script: python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test")

if __name__ == "__main__":
    main()
