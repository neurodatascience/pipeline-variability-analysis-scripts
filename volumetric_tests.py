import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ks_2samp
import itertools
import json

def run_scientific_stats(file_path='df_tidy.csv', output_file='volumetric_stat_results.json'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return

    # Data cleaning
    df['volume_mm3'] = pd.to_numeric(df['volume_mm3'], errors='coerce')
    df = df.dropna(subset=['volume_mm3'])
    
    pipelines = sorted(df['pipeline'].unique())
    structures = df['structure'].unique()
    
    # Calculate total comparisons for Bonferroni multiplier
    n_pairs = len(list(itertools.combinations(pipelines, 2)))
    m_total = n_pairs * len(structures)

    print("\n" + "="*130)
    print(f"{'FORMAL STATISTICAL ANALYSIS REPORT':^130}")
    print(f"{'(Bonferroni Correction: p_adj = p_raw * ' + str(m_total) + ')':^130}")
    print("="*130)

    # Prepare JSON storage
    all_results = {'ks_tests': [], 'mwu_tests': [], 'summary': []}

    # --- SECTION 1: DISTRIBUTIONAL OVERLAP (K-S TEST) ---
    print(f"\n[SECTION 1] DISTRIBUTIONAL SIMILARITY (Kolmogorov-Smirnov)")
    print(f"H0: Pipelines draw from identical volume distributions. HA: Distributions differ in shape/location.")
    print("-" * 130)
    print(f"{'Structure':<25} | {'Mean Overlap (1-D)':<20} | {'Avg Raw p':<12} | {'Bonf. p (min)':<12} | {'Agreement'}")
    print("-" * 110)
    
    for struct in structures:
        struct_df = df[df['structure'] == struct]
        scores, ks_pvals = [], []
        struct_ks_details = []
        print(f"\nSTRUCTURE: {struct}")
        for p1, p2 in itertools.combinations(pipelines, 2):
            v1 = struct_df[struct_df['pipeline'] == p1]['volume_mm3']
            v2 = struct_df[struct_df['pipeline'] == p2]['volume_mm3']
            if len(v1) > 1 and len(v2) > 1:
                d_stat, p_ks = ks_2samp(v1, v2)
                overlap = 1 - d_stat
                scores.append(overlap)
                ks_pvals.append(p_ks)
                # CLI print
                print(f"  KS Test: {p1} vs {p2} | D = {d_stat:.4f} | p_raw = {p_ks:.2e} | 1-D overlap = {overlap:.4f}")
                # Save for JSON
                struct_ks_details.append({
                    'pipeline_pair': f"{p1} vs {p2}",
                    'D': d_stat,
                    'p_raw': p_ks,
                    'overlap': overlap
                })
        avg_ovl = np.mean(scores) if scores else 0
        avg_p = np.mean(ks_pvals) if ks_pvals else 1.0
        min_p_adj = np.clip(min(ks_pvals) * m_total, 0, 1.0) if ks_pvals else 1.0
        status = "HIGH" if avg_ovl > 0.7 else "LOW" if avg_ovl < 0.4 else "MODERATE"
        print(f"{struct:<25} | {avg_ovl:<20.4f} | {avg_p:<12.2e} | {min_p_adj:<12.2e} | {status}")

        all_results['ks_tests'].append({
            'structure': struct,
            'avg_overlap': avg_ovl,
            'avg_p': avg_p,
            'min_p_bonf': min_p_adj,
            'agreement': status,
            'details': struct_ks_details
        })

# --- SECTION 2: MAGNITUDE COMPARISONS (WILCOXON SIGNED-RANK) ---
    print(f"\n[SECTION 2] MAGNITUDE COMPARISONS (Wilcoxon Signed-Rank Test)")
    print(f"H0: The median of the differences between pipelines is zero. HA: There is a systematic shifting.")
    print("-" * 130)

    raw_results = []
    for struct in structures:
        struct_df = df[df['structure'] == struct]
        print(f"\nSTRUCTURE: {struct}")
        
        for p1, p2 in itertools.combinations(pipelines, 2):
            # Isolate data for both pipelines
            df_p1 = struct_df[struct_df['pipeline'] == p1]
            df_p2 = struct_df[struct_df['pipeline'] == p2]
            
            # Align pairs by subject identifier to guarantee correct matching.
            # NOTE: Change 'subject' to your actual subject/ID column name if it differs.
            id_col = 'subject' if 'subject' in df.columns else df.columns[0] 
            
            paired_df = pd.merge(
                df_p1[[id_col, 'volume_mm3']], 
                df_p2[[id_col, 'volume_mm3']], 
                on=id_col, 
                suffixes=('_p1', '_p2')
            ).dropna()
            
            v1 = paired_df['volume_mm3_p1']
            v2 = paired_df['volume_mm3_p2']
            
            # Wilcoxon requires at least some non-zero differences and matching lengths
            if len(paired_df) > 0 and not (v1 == v2).all():
                w_stat, p_raw = wilcoxon(v1, v2, alternative='two-sided')
                med1, med2 = v1.median(), v2.median()
                
                raw_results.append({
                    'structure': struct, 'p1': p1, 'p2': p2,
                    'med1': med1, 'med2': med2,
                    'w_stat': w_stat, 'p_raw': p_raw
                })
                # CLI print
                print(f"  Wilcoxon Sign-Rank: {p1} vs {p2} | W = {w_stat:.2f} | p_raw = {p_raw:.2e} | med1 = {med1:.2f} | med2 = {med2:.2f}")
            else:
                print(f"  Wilcoxon Sign-Rank: {p1} vs {p2} | Skipped (insufficient paired data or identical values)")

    if raw_results:
        res_df = pd.DataFrame(raw_results)
        res_df['p_bonf'] = np.clip(res_df['p_raw'] * m_total, 0, 1.0)
        res_df['sig'] = res_df['p_bonf'] < 0.05

        header = f"{'Structure':<20} | {'Comparison':<35} | {'Raw p':<12} | {'Bonf. p':<12} | {'Result'}"
        print("\n" + header)
        print("-" * len(header))

        for _, row in res_df.iterrows():
            higher = 'A' if row['med1'] > row['med2'] else 'B'
            lower = 'B' if row['med1'] > row['med2'] else 'A'
            res_text = f"Reject H0 ({higher} > {lower})" if row['sig'] else "Fail to Reject"
            print(f"{row['structure']:<20} | {row['p1']+' vs '+row['p2']:<35} | {row['p_raw']:<12.2e} | {row['p_bonf']:<12.2e} | {res_text}")

            all_results['mwu_tests'].append({
                'structure': row['structure'],
                'pipeline_pair': f"{row['p1']} vs {row['p2']}",
                'medians': [row['med1'], row['med2']],
                'W': row['w_stat'],
                'p_raw': row['p_raw'],
                'p_bonf': row['p_bonf'],
                'significant': row['sig'],
                'result': res_text
            })
    else:
        # Fallback if no tests were run
        res_df = pd.DataFrame(columns=['p1', 'p2', 'sig'])

    # --- SECTION 3: SUPERVISOR METRIC ---
    print(f"\n[SECTION 3] FRACTION OF TESTS REJECTED (Alpha = 0.05)")
    print("-" * 80)
    for p in pipelines:
        rel = res_df[(res_df['p1'] == p) | (res_df['p2'] == p)]
        frac = rel['sig'].mean()
        print(f"Pipeline: {p:<25} | Fraction Rejected: {rel['sig'].sum()}/{len(rel)} ({frac:.2%})")
        all_results['summary'].append({
            'pipeline': p,
            'tests_rejected': int(rel['sig'].sum()),
            'total_tests': len(rel),
            'fraction': frac
        })

    # Save all results to JSON
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll detailed results saved to '{output_file}'.")

if __name__ == "__main__":
    run_scientific_stats()
