import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = "dice_overlap_tidy.csv"
OUTPUT_FILE = "dice_overlap_summary_heatmap.png"

# -----------------------------------------------------------------------------
# Mappings & Helpers
# -----------------------------------------------------------------------------
def shorten_pipeline_name(pipeline_name):
    """Maps long pipeline names to short acronyms matching userscript3.py"""
    mapping = {
        'freesurfer741ants243': 'FS741',
        'samseg8001ants243': 'Samseg8', 
        'freesurfer8001ants243': 'FS8001',
        'fslanat6071ants243': 'FSL6071'
    }
    return mapping.get(pipeline_name, pipeline_name)

def clean_structure_names(df):
    """
    Normalizes structure names from the raw CSV (e.g., 'Left-Thalamus-Proper')
    to the cleaner style used in userscript3 (e.g., 'Left-Thalamus').
    """
    # Remove '-Proper' suffix
    df['structure'] = df['structure'].str.replace('-Proper', '', regex=False)
    # Rename Brain-Stem
    df['structure'] = df['structure'].replace('Brain-Stem/4thVentricle', 'Brainstem')
    return df

def get_structure_order():
    """Defines the standard anatomical order for the Y-axis."""
    base_structures = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens-area']
    ordered_structures = []
    for struct in base_structures:
        ordered_structures.extend([f'Left-{struct}', f'Right-{struct}'])
    ordered_structures.append('Brainstem')
    return ordered_structures

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    file_path = Path(INPUT_FILE)
    if not file_path.exists():
        print(f"Error: Could not find {INPUT_FILE} in the current directory.")
        return

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(file_path)

    # --- Calculate N (Total Scans/Sessions) ---
    if 'session' in df.columns:
        n_scans = df[['subject', 'session']].drop_duplicates().shape[0]
    else:
        print("Warning: 'session' column not found. Counting unique 'subject' entries.")
        n_scans = df['subject'].nunique()

    # 1. Clean Data
    df = clean_structure_names(df)
    
    # 2. Create Short Pipeline Pair Names (X-Axis)
    df['p1_short'] = df['pipeline1'].apply(shorten_pipeline_name)
    df['p2_short'] = df['pipeline2'].apply(shorten_pipeline_name)
    df['pipeline_pair'] = df['p1_short'] + "_vs_" + df['p2_short']

    # 3. Aggregate: Calculate Mean Dice per Structure per Pair
    mean_dice_df = df.groupby(['structure', 'pipeline_pair'])['dice'].mean().reset_index()

    # 4. Pivot for Heatmap
    heatmap_data = mean_dice_df.pivot(index='structure', columns='pipeline_pair', values='dice')

    # 5. Sort Rows (Structures)
    structure_order = get_structure_order()
    existing_structures = [s for s in structure_order if s in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(existing_structures)

    # 6. Plotting
    # --- MODIFIED: Reduced height from 8 to 6 to compress vertical axis ---
    plt.figure(figsize=(10, 6))
    sns.set_style("white")

    # Format annotations to 2 decimal places
    annot_data = heatmap_data.applymap(lambda x: f"{x:.2f}")

    # Create Heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=annot_data, 
        fmt='', 
        cmap='viridis', 
        vmin=0.5, vmax=1.0,
        cbar_kws={'label': 'Mean Dice Score'},
        linewidths=0.5
    )

    # Title with Mean and Correct N
    plt.title(f'Mean Pairwise Dice Overlap by Structure (n={n_scans})', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('')
    plt.xlabel('')
    
    # Rotate X-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = Path(OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved heatmap to: {output_path}")

if __name__ == "__main__":
    main()
