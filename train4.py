import os
import time
import pickle
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%I:%M:%S %p')}] {msg}", flush=True)

# Custom callback to save model checkpoints during training
class CheckpointCallback(xgb.callback.TrainingCallback):
    def __init__(self, checkpoint_dir, save_every=10):
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.save_every == 0:
            filepath = os.path.join(self.checkpoint_dir, f"xgb_checkpoint_tree_{epoch+1}.json")
            model.save_model(filepath)
        return False

# ------------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------------------------
def main():
    log("=== XGBOOST MODEL TRAINING ===")

    # 1. Load the Features
    file_path = "output/train_features.parquet"
    log(f"Loading features from {file_path}...")
    df = pd.read_parquet(file_path)

    # 2. Define Features and Target
    feature_cols = [
        'name_jw', 'name_lev', 'addr_lev',
        'name_jaccard', 'num_overlap',
        'name_len_ratio', 'name_subset'
    ]
    X = df[feature_cols]
    y = df['label']
    groups = df['source1_entity_id']

    # 3. Train / Validation Split
    log("Splitting data into 80% Train and 20% Validation (grouped by S1 Entity)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- RAM OPTIMIZATION ---
    log("Clearing original dataframe from RAM...")
    del df, X, y, groups
    gc.collect()

    log(f"Training set: {len(X_train):,} pairs")
    log(f"Validation set: {len(X_val):,} pairs")

    # 4. Train the Model
    log("Initializing XGBoost Classifier...")
    
    ckpt_callback = CheckpointCallback(checkpoint_dir="model_checkpoints", save_every=10)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        # Removed scale_pos_weight to allow natural probability calibration for F0.5
        eval_metric="logloss",
        early_stopping_rounds=20, # Stops if validation score doesn't improve for 20 rounds
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        callbacks=[ckpt_callback]
    )

    log("Training model... (Saving checkpoints to 'model_checkpoints/')")
    t0 = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25
    )

    log(f"Training complete in {(time.time() - t0)/60:.2f} mins.")

    # 5. Threshold Tuning for F_0.5 on Validation Set
    log("\n--- TUNING THRESHOLD FOR F_0.5 SCORE ---")
    log("Predicting probabilities on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_f05 = 0.0

    print(f"\n{'Threshold':<12} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F_0.5 Score':<12}")
    print("-" * 70)

    # Added higher thresholds (0.90+) because the model will be very confident now
    thresholds = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.98, 0.99]

    for t in thresholds:
        preds = (val_probs >= t).astype(int)

        acc = accuracy_score(y_val, preds)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)

        # F_0.5 Formula
        if (0.25 * p + r) == 0:
            f05 = 0
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)

        print(f"{t:<12.2f} | {acc:<12.4f} | {p:<12.4f} | {r:<12.4f} | {f05:<12.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t

    log(f"\nOptimal Threshold Found: {best_thresh} (F_0.5 = {best_f05:.4f})")

    # 6. Save Final Model and Metadata
    os.makedirs("model_artifacts", exist_ok=True)

    final_model_path = "model_artifacts/xgb_final_model.json"
    model.save_model(final_model_path)

    meta_path = "model_artifacts/model_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({"best_threshold": best_thresh, "features": feature_cols}, f)

    log(f"Final Model saved to {final_model_path}")
    log("=== ML PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
