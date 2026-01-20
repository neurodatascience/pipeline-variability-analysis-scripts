import pandas as pd
import numpy as np
import argparse
import sys
import os

def analyze_and_export(file_path, output_csv="analysis_summary.csv"):
    print("[Thinking] Initializing analysis...")
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"[Thinking] Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        print(f"[Thinking] Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        sys.exit(1)

    if df.empty:
        print("[Error] The CSV file is empty.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Find Best Learner
    # ---------------------------------------------------------
    print("[Thinking] Scanning for the best performing model (lowest MAE) across all rows...")
    best_row_idx = df['mean_mae'].idxmin()
    
    best_learner = df.loc[best_row_idx, 'model']
    best_sex = df.loc[best_row_idx, 'sex']
    best_overall_mae = df.loc[best_row_idx, 'mean_mae']
    
    print(f"[Thinking] Identified best learner: '{best_learner}' in group '{best_sex}' with MAE {best_overall_mae:.4f}.")

    # ---------------------------------------------------------
    # 3. Get Top 2 Pipelines within that Learner/Sex
    # ---------------------------------------------------------
    print(f"[Thinking] Fetching all pipelines for model '{best_learner}' and sex '{best_sex}'...")
    
    # Filter and Copy
    learner_df = df[(df['model'] == best_learner) & (df['sex'] == best_sex)].copy()
    
    # Sort by MAE (ascending)
    learner_df = learner_df.sort_values('mean_mae')
    
    if len(learner_df) < 2:
        print("[Error] Not enough pipelines to compare (found fewer than 2).")
        sys.exit(1)

    # Select Top 2
    best_pipeline_row = learner_df.iloc[0]
    second_best_pipeline_row = learner_df.iloc[1]

    name_1 = best_pipeline_row['pipeline']
    mae_1 = best_pipeline_row['mean_mae']
    
    name_2 = second_best_pipeline_row['pipeline']
    mae_2 = second_best_pipeline_row['mean_mae']

    print(f"[Thinking] Top pipeline:      '{name_1}' (MAE: {mae_1:.4f})")
    print(f"[Thinking] Runner-up pipeline: '{name_2}' (MAE: {mae_2:.4f})")
    
    # ---------------------------------------------------------
    # 4. Perform Permutation Test
    # ---------------------------------------------------------
    print("[Thinking] Preparing data for paired permutation testing...")
    
    fold_cols = [f'mae_fold_{i}' for i in range(1, 8)]
    
    # Verify columns exist
    if not all(col in df.columns for col in fold_cols):
        print(f"[Error] Missing fold columns: {fold_cols}")
        sys.exit(1)

    scores_A = best_pipeline_row[fold_cols].values.astype(float)
    scores_B = second_best_pipeline_row[fold_cols].values.astype(float)
    
    # Difference: Model B (Worse) - Model A (Better). Expected > 0.
    differences = scores_B - scores_A
    obs_mean_diff = np.mean(differences)
    
    print(f"[Thinking] Observed Mean Difference (Model B - Model A): {obs_mean_diff:.4f}")
    
    n_permutations = 10000
    print(f"[Thinking] Generating {n_permutations} null permutations (flipping signs of differences)...")
    
    # Reproducibility
    rng = np.random.default_rng(42)
    
    # Generate random signs (-1 or 1)
    random_signs = rng.choice([-1, 1], size=(n_permutations, len(differences)))
    
    # Calculate permuted means
    permuted_diffs = random_signs * differences
    permuted_means = np.mean(permuted_diffs, axis=1)
    
    print("[Thinking] Calculating P-value...")
    
    # Two-sided P-value
    p_value = np.mean(np.abs(permuted_means) >= np.abs(obs_mean_diff))
    is_significant = p_value < 0.05
    
    print(f"[Thinking] P-value: {p_value:.5f}. Significant (< 0.05)? {is_significant}")

    # ---------------------------------------------------------
    # 5. Export Results
    # ---------------------------------------------------------
    print(f"[Thinking] Saving results to '{output_csv}'...")
    
    results = {
        'best_learner': [best_learner],
        'sex': [best_sex],
        'best_pipeline': [name_1],
        'best_pipeline_mae': [mae_1],
        'second_best_pipeline': [name_2],
        'second_best_pipeline_mae': [mae_2],
        'mean_observed_diff': [obs_mean_diff],
        'p_value': [p_value],
        'is_significant': [is_significant]
    }
    
    results_df = pd.DataFrame(results)
    
    try:
        results_df.to_csv(output_csv, index=False)
        print(f"[Thinking] Successfully saved file.")
    except Exception as e:
        print(f"[Error] Could not save CSV: {e}")

    print("-" * 60)
    print("DONE. Summary:")
    print(f"Comparison: {name_1} vs {name_2}")
    print(f"P-Value:    {p_value:.5f}")
    print(f"Conclusion: {'Statistically Significant' if is_significant else 'Not Significant'}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze brain age ML results with logging.")
    parser.add_argument("file_path", nargs='?', 
                        default="brain_age_results_sex_split_and_correct_ensembles.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", default="analysis_summary.csv",
                        help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    analyze_and_export(args.file_path, args.output)
