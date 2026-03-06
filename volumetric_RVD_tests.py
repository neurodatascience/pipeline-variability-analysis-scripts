import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import json
from itertools import combinations

def run_rvd_analysis(input_csv):
    """
    Analyzes Relative Volume Difference (RVD) using the statistical logic 
    from the Dice overlap script.
    """
    try:
        df_vol = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

    # 1. DATA INTERSECTION: Only sessions available across ALL pipelines
    pipelines = df_vol['pipeline'].unique()
    num_pipelines = len(pipelines)
    
    # Identify keys available in ALL pipelines
    counts = df_vol.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline'].nunique()
    valid_keys = counts[counts == num_pipelines].index
    df_clean = df_vol.set_index(['dataset', 'subject', 'session', 'structure']).loc[valid_keys].reset_index()
    
    # 2. PAIRWISE DATA CONSTRUCTION
    pipeline_pairs = list(combinations(pipelines, 2))
    pairwise_list = []
    
    for p1, p2 in pipeline_pairs:
        df1 = df_clean[df_clean['pipeline'] == p1]
        df2 = df_clean[df_clean['pipeline'] == p2]
        
        merged = pd.merge(df1, df2, on=['dataset', 'subject', 'session', 'structure'], suffixes=('_p1', '_p2'))
        merged['mean_vol'] = (merged['volume_mm3_p1'] + merged['volume_mm3_p2']) / 2.0
        merged['abs_diff'] = (merged['volume_mm3_p1'] - merged['volume_mm3_p2']).abs()
        merged['rel_diff'] = merged['abs_diff'] / merged['mean_vol']
        merged['p1'], merged['p2'] = p1, p2
        pairwise_list.append(merged)
        
    df_all = pd.concat(pairwise_list)
    
    # 3. ANALYSIS 1: Volumetric Scaling (Correlation of Medians across structures)
    pairs = df_all.groupby(['p1', 'p2'])
    pair_results = []
    p_vals_h1 = []
    
    for name, group in pairs:
        # Aggregate to find median Volume and median RVD per structure (Logic from Dice script)
        agg = group.groupby('structure').agg({
            'rel_diff': 'median',
            'mean_vol': 'median'
        })
        
        if len(agg) > 2:
            rho, p = spearmanr(agg['mean_vol'], agg['rel_diff'])
            if not np.isnan(rho):
                pair_label = f"{name[0]} vs {name[1]}"
                pair_results.append({
                    "pair": pair_label,
                    "rho": float(rho),
                    "p": float(p)
                })
                p_vals_h1.append(p)

    # Bonferroni correction per-comparison
    n_tests = len(p_vals_h1)
    if n_tests > 0:
        for i in range(len(pair_results)):
            p_adj = min(pair_results[i]["p"] * n_tests, 1.0)
            pair_results[i]["p_adj"] = float(p_adj)
            pair_results[i]["significant"] = bool(p_adj < 0.05)
            
    avg_rho = np.mean([r['rho'] for r in pair_results]) if pair_results else np.nan

    # 4. ANALYSIS 2: FSL Bias Comparison (Distribution comparison with Z-score Effect Size)
    df_all['is_fsl'] = df_all['p1'].str.contains('fsl', case=False) | \
                       df_all['p2'].str.contains('fsl', case=False)
    
    fsl_rvd = df_all[df_all['is_fsl']]['rel_diff']
    other_rvd = df_all[~df_all['is_fsl']]['rel_diff']
    
    # One-sided test: Is FSL RVD 'greater' (worse) than others?
    u, p_h2 = mannwhitneyu(fsl_rvd, other_rvd, alternative='greater')
    
    # Calculate Z-score and Effect Size r (Logic from Dice script)
    n1, n2 = len(fsl_rvd), len(other_rvd)
    z = (u - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    r_val = abs(z) / np.sqrt(n1 + n2)
    
    # 5. ANATOMICAL HIERARCHY (Sorted by Median Volume)
    hierarchy = df_all.groupby('structure')['mean_vol'].median().sort_values()
    
    # 6. CLI REPORTING
    print("\n" + "="*95)
    print(f"VOLUMETRIC DISAGREEMENT (RVD) ANALYSIS REPORT")
    print(f"(N = {df_clean[['dataset','subject','session']].drop_duplicates().shape[0]} sessions, Intersection of {num_pipelines} pipelines)")
    print("="*95)
    
    print(f"{'Structure (Sorted by Median Volume)':<40} | {'Volume (mm3)':>15}")
    print("-" * 60)
    for s, v in hierarchy.items():
        print(f"{s:<40} | {v:>15.2f}")
        
    print("\n" + "-"*95)
    print("HYPOTHESIS 1: Is there a volumetric bias (Size vs. RVD)?")
    print(f"Mean Rho across pairs = {avg_rho:.3f}")
    print(f"Number of comparisons = {n_tests}")
    
    print("\nPer-Pair Results (Correlation of structure medians):")
    print(f"{'Pipeline Pair':<50} | {'Rho':>6} | {'Raw p':>10} | {'Adj p':>10} | Sig")
    print("-" * 95)
    for r in pair_results:
        print(f"{r['pair']:<50} | "
              f"{r['rho']:>6.3f} | "
              f"{r['p']:>10.3e} | "
              f"{r['p_adj']:>10.3e} | "
              f"{'YES' if r['significant'] else 'NO'}")

    print("-" * 95)
    print("HYPOTHESIS 2: Does FSL introduce systematic volumetric bias?")
    print(f"FSL Median RVD = {fsl_rvd.median():.4f} | "
          f"Non-FSL Median RVD = {other_rvd.median():.4f}")
    print(f"p = {p_h2:.3e} (Effect Size r = {r_val:.3f})")
    
    conclusion_h2 = "Significant difference detected (FSL pairings show higher RVD)." if p_h2 < 0.05 else "No significant difference detected."
    print(f"CONCLUSION: {conclusion_h2}")
    print("="*95)
    
    # 7. SAVE TO JSON
    results_json = {
        "metadata": {
            "n_sessions": int(df_clean[['dataset','subject','session']].drop_duplicates().shape[0]),
            "n_pipelines": int(num_pipelines)
        },
        "anatomical_hierarchy": hierarchy.to_dict(),
        "hypothesis_1_scaling": {
            "mean_rho": float(avg_rho),
            "n_tests": n_tests,
            "pairwise_results": pair_results
        },
        "hypothesis_2_fsl_bias": {
            "fsl_median": float(fsl_rvd.median()),
            "other_median": float(other_rvd.median()),
            "p_val": float(p_h2),
            "r_val": float(r_val),
            "conclusion": conclusion_h2
        }
    }
    
    with open('volumetric_rvd_stats.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    print("\nResults successfully saved to volumetric_rvd_stats.json")

if __name__ == "__main__":
    run_rvd_analysis('df_tidy.csv')
