import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "df_tidy.csv"
OUTPUT_FILE = "pipeline_correlation_heatmap.png"

def filter_complete_pipelines(df_tidy):
    """Ensure strict Session-Level Complete Case Analysis."""
    required_pipelines = {
        'fslanat6071ants243', 'freesurfer741ants243', 
        'freesurfer8001ants243', 'samseg8001ants243'
    }
    counts = (
        df_tidy.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )
    incomplete_structures = counts[counts['n_pipelines'] < len(required_pipelines)]
    bad_sessions = incomplete_structures[['dataset', 'subject', 'session']].drop_duplicates()
    
    if len(bad_sessions) > 0:
        merged = df_tidy.merge(bad_sessions, on=['dataset', 'subject', 'session'], how='left', indicator=True)
        return merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    return df_tidy.copy()

def get_sorted_structures(structures):
    """Sort structures according to the predefined anatomical order."""
    TARGET_ORDER = [
        'Left-Thalamus', 'Right-Thalamus', 'Left-Caudate', 'Right-Caudate',
        'Left-Putamen', 'Right-Putamen', 'Left-Pallidum', 'Right-Pallidum',
        'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala',
        'Left-Accumbens-area', 'Right-Accumbens-area', 'Brainstem'
    ]
    present_structures = set(structures)
    sorted_structures = [s for s in TARGET_ORDER if s in present_structures]
    non_target_structures = sorted([s for s in present_structures if s not in TARGET_ORDER])
    sorted_structures.extend(non_target_structures)
    return sorted_structures

def main():
    file_path = Path(INPUT_FILE)
    if not file_path.exists():
        print(f"Error: Could not find {INPUT_FILE} in the current directory.")
        return

    print(f"Loading {INPUT_FILE}...")
    df_tidy = pd.read_csv(file_path)

    # 1. Clean and Sanitize Data
    df_tidy['volume_mm3'] = pd.to_numeric(df_tidy['volume_mm3'], errors='coerce')
    df_tidy = df_tidy.dropna(subset=['volume_mm3'])
    df_tidy = df_tidy[df_tidy['volume_mm3'] > 0]

    # 2. Apply Strict Session Filtering
    df_complete = filter_complete_pipelines(df_tidy)
    
    # Calculate N (Total Unique Scans/Sessions remaining)
    n_scans = df_complete[['dataset', 'subject', 'session']].drop_duplicates().shape[0]

    # 3. Map Pipelines to Short Names
    pipeline_mapping = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001', 
        'samseg8001ants243': 'SAMSEG8'
    }
    df_complete['pipeline_short'] = df_complete['pipeline'].map(pipeline_mapping)
    pipeline_order = list(pipeline_mapping.values())

    # Create unique scan IDs for pivoting
    df_complete['scan_id'] = df_complete[['dataset', 'subject', 'session']].astype(str).agg('_'.join, axis=1)

    # 4. Calculate Pairwise Correlations per Structure
    correlation_records = []
    structures = df_complete['structure'].unique()

    for structure in structures:
        struct_data = df_complete[df_complete['structure'] == structure]
        pivot_df = struct_data.pivot_table(index='scan_id', columns='pipeline_short', values='volume_mm3')
        
        # Verify all pipelines are present for correlation calculation
        if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
            corr_matrix = pivot_df.corr(method='spearman')
            
            # Extract unique pipeline pairs (upper triangle of the matrix)
            for i in range(len(pipeline_order)):
                for j in range(i + 1, len(pipeline_order)):
                    p1, p2 = pipeline_order[i], pipeline_order[j]
                    pair_name = f"{p1}_vs_{p2}"
                    corr_value = corr_matrix.loc[p1, p2]
                    
                    correlation_records.append({
                        'structure': structure,
                        'pipeline_pair': pair_name,
                        'correlation': corr_value
                    })

    corr_df = pd.DataFrame(correlation_records)

    # 5. Pivot into Matrix for Heatmap
    heatmap_data = corr_df.pivot(index='structure', columns='pipeline_pair', values='correlation')

    # 6. Sort Rows (Structures)
    sorted_structs = get_sorted_structures(heatmap_data.index)
    heatmap_data = heatmap_data.reindex(sorted_structs)

    # 7. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("white")

    # Format annotations safely to 2 decimal places
    annot_data = heatmap_data.apply(lambda col: col.map(lambda x: f"{x:.2f}" if pd.notna(x) else ""))

    # Create Heatmap (Using divergent palette suited for correlation coefficients)
    ax = sns.heatmap(
        heatmap_data, 
        annot=annot_data, 
        fmt='', 
        cmap='RdBu_r', 
        vmin=-1.0, vmax=1.0, center=0,
        cbar_kws={'label': "Spearman's Rank Correlation ($\\rho$)"},
        linewidths=0.5
    )

    # Title and Labels
    plt.title(f'Inter-Pipeline Volume Correlation by Structure (n={n_scans})', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('')
    plt.xlabel('')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    

    print("\n" + "="*50)
    print("STATISTICS FOR THE RESULTS SECTION")
    print("="*50)
    
    # 1. Overall Dataset Metrics
    print(f"Total fully complete scans analyzed (n): {n_scans}")
    print(f"Total brain structures evaluated: {len(heatmap_data)}")
    print(f"Total pipeline pairs evaluated: {len(heatmap_data.columns)}")
    
    # 2. Global Correlation Ranges
    all_corrs = corr_df['correlation'].dropna()
    print(f"\nGlobal Spearman correlation range: [{all_corrs.min():.4f}, {all_corrs.max():.4f}]")
    print(f"Global mean Spearman correlation: {all_corrs.mean():.4f}")
    
    # 3. Best and Worst Performing Structure-Pair Combinations
    max_idx = corr_df['correlation'].idxmax()
    min_idx = corr_df['correlation'].idxmin()
    
    print(f"\nHighest Correlation:")
    print(f"  Structure: {corr_df.loc[max_idx, 'structure']}")
    print(f"  Pipeline Pair: {corr_df.loc[max_idx, 'pipeline_pair']}")
    print(f"  Value: {corr_df.loc[max_idx, 'correlation']:.4f}")
    
    print(f"\nLowest Correlation:")
    print(f"  Structure: {corr_df.loc[min_idx, 'structure']}")
    print(f"  Pipeline Pair: {corr_df.loc[min_idx, 'pipeline_pair']}")
    print(f"  Value: {corr_df.loc[min_idx, 'correlation']:.4f}")
    
    # 4. Average Correlation Per Pipeline Pair (Which pipelines agree most/least generally?)
    print("\nMean Correlation Grouped by Pipeline Pair:")
    pair_means = corr_df.groupby('pipeline_pair')['correlation'].mean().sort_values(ascending=False)
    for pair, val in pair_means.items():
        print(f"  {pair}: {val:.4f}")
        
    # 5. Average Correlation Per Structure (Which brain regions are most/least stable?)
    print("\nTop 3 Most Consistent Structures (Highest Mean Correlation):")
    struct_means = corr_df.groupby('structure')['correlation'].mean().sort_values(ascending=False)
    for struct, val in struct_means.head(3).items():
        print(f"  {struct}: {val:.4f}")
        
    print("\nBottom 3 Least Consistent Structures (Lowest Mean Correlation):")
    for struct, val in struct_means.tail(3).iloc[::-1].items():
        print(f"  {struct}: {val:.4f}")
    print("="*50 + "\n")

    output_path = Path(OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved correlation heatmap to: {output_path}")

if __name__ == "__main__":
    main()
