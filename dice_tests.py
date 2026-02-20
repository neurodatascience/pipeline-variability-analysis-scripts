import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import json
import os

# 1. Load Data
df_dice = pd.read_csv('dice_overlap_tidy.csv')
df_vol = pd.read_csv('df_tidy.csv')

# 2. Harmonization
ds_map = {
    'NKI_nipoppy': 'Rockland',
    'ds002345_nipoppy': 'ds002345',
    'ds003592_nipoppyy': 'ds003592',
    'ds005752_nipoppy': 'ds005752',
    'preventAD': 'PreventAD'
}
df_dice['dataset'] = df_dice['dataset'].replace(ds_map)

struct_map = {
    'Left-Thalamus-Proper': 'Left-Thalamus',
    'Right-Thalamus-Proper': 'Right-Thalamus',
    'Brain-Stem/4thVentricle': 'Brainstem'
}
df_dice['struct_key'] = df_dice['structure'].replace(struct_map)
df_vol['struct_key'] = df_vol['structure']

merge_keys = ['dataset', 'subject', 'session', 'struct_key']
for col in merge_keys:
    df_dice[col] = df_dice[col].astype(str).str.strip()
    df_vol[col] = df_vol[col].astype(str).str.strip()

# 3. Merge
df = pd.merge(
    df_dice,
    df_vol[merge_keys + ['pipeline', 'volume_mm3']],
    left_on=merge_keys + ['pipeline1'],
    right_on=merge_keys + ['pipeline'],
    how='inner'
).rename(columns={'volume_mm3': 'vol1'}).drop(columns='pipeline')

df = pd.merge(
    df,
    df_vol[merge_keys + ['pipeline', 'volume_mm3']],
    left_on=merge_keys + ['pipeline2'],
    right_on=merge_keys + ['pipeline'],
    how='inner'
).rename(columns={'volume_mm3': 'vol2'}).drop(columns='pipeline')

df['mean_vol'] = df[['vol1', 'vol2']].mean(axis=1)

# 4. Hypothesis 1: Size Bias (Per-Pair Breakdown)
pairs = df.groupby(['pipeline1', 'pipeline2'])
pair_results = []
p_vals_h1 = []

print("\n" + "="*95)
print("DETAILED BREAKDOWN: HYPOTHESIS 1 (SIZE VS OVERLAP)")
print("-" * 95)
print(f"{'Pipeline Pair':<50} | {'Rho':>6} | {'Raw p':>10}")
print("-" * 72)

for name, group in pairs:
    agg = group.groupby('struct_key').agg({
        'dice': 'median',
        'mean_vol': 'median'
    })

    if len(agg) > 2:
        rho, p = spearmanr(agg['mean_vol'], agg['dice'])

        if not np.isnan(rho):
            pair_label = f"{name[0]} vs {name[1]}"
            pair_results.append({
                "pair": pair_label,
                "rho": float(rho),
                "p": float(p)
            })
            p_vals_h1.append(p)

            print(f"{pair_label:<50} | {rho:>6.3f} | {p:>10.3e}")

# Proper Bonferroni per-comparison correction
n_tests = len(p_vals_h1)

if n_tests > 0:
    adj_p_vals = [min(p * n_tests, 1.0) for p in p_vals_h1]

    for i in range(len(pair_results)):
        pair_results[i]["p_adj"] = float(adj_p_vals[i])
        pair_results[i]["significant"] = bool(adj_p_vals[i] < 0.05)

avg_rho = np.mean([r['rho'] for r in pair_results]) if pair_results else np.nan

# 5. Hypothesis 2: FSL Bias
df['is_fsl'] = df['pipeline1'].str.contains('fsl', case=False) | \
               df['pipeline2'].str.contains('fsl', case=False)

fsl_dice = df[df['is_fsl']]['dice']
other_dice = df[~df['is_fsl']]['dice']

u, p_h2 = mannwhitneyu(fsl_dice, other_dice, alternative='less')

n1, n2 = len(fsl_dice), len(other_dice)
z = (u - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
r_val = abs(z) / np.sqrt(n1 + n2)

# 6. Anatomical Hierarchy
hierarchy = df.groupby('struct_key')['mean_vol'].median().sort_values()

# --- FINAL REPORT ---
print("\n" + "="*95)
print(f"SUB-CORTICAL SEGMENTATION ANALYSIS REPORT "
      f"(N = {df[['dataset','subject','session']].drop_duplicates().shape[0]})")
print("="*95)

print(f"{'Structure (Sorted by Median Volume)':<40} | {'Volume (mm3)':>15}")
print("-" * 60)
for s, v in hierarchy.items():
    print(f"{s:<40} | {v:>15.2f}")

print("\n" + "-"*95)
print("HYPOTHESIS 1: Is there a volumetric bias? (Bonferroni corrected)")
print(f"Mean Rho across pairs = {avg_rho:.3f}")
print(f"Number of comparisons = {n_tests}")

print("\nPer-Pair Results:")
print(f"{'Pipeline Pair':<50} | {'Rho':>6} | {'Raw p':>10} | {'Adj p':>10} | Sig")
print("-" * 95)

for r in pair_results:
    print(f"{r['pair']:<50} | "
          f"{r['rho']:>6.3f} | "
          f"{r['p']:>10.3e} | "
          f"{r['p_adj']:>10.3e} | "
          f"{'YES' if r['significant'] else 'NO'}")

print("-" * 95)
print("HYPOTHESIS 2: Does FSL introduce systematic bias?")
print(f"FSL Median = {fsl_dice.median():.4f} | "
      f"Non-FSL Median = {other_dice.median():.4f}")
print(f"p = {p_h2:.3e} (Effect Size r = {r_val:.3f})")

if p_h2 < 0.05:
    conclusion_h2 = "Significant difference detected."
else:
    conclusion_h2 = "No significant difference detected."

print(f"CONCLUSION: {conclusion_h2}")
print("="*95)

# --- SAVE TO JSON ---
results_json = {
    "n_sessions": int(
        df[['dataset','subject','session']].drop_duplicates().shape[0]
    ),
    "anatomical_hierarchy": hierarchy.to_dict(),
    "hypothesis_1": {
        "mean_rho": float(avg_rho),
        "n_tests": n_tests,
        "correction": "bonferroni_per_comparison",
        "pairwise_results": pair_results
    },
    "hypothesis_2": {
        "fsl_median": float(fsl_dice.median()),
        "other_median": float(other_dice.median()),
        "p_val": float(p_h2),
        "r_val": float(r_val),
        "conclusion": conclusion_h2
    }
}

with open('subcortical_analysis_results.json', 'w') as f:
    json.dump(results_json, f, indent=4)

print("\nResults and conclusions successfully saved to JSON.")
