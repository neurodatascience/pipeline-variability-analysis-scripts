import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('brain_age_results_sex_split_and_correct_ensembles.csv')

# Clean up model names by removing 'ensemble_' prefix
df['model_clean'] = df['model'].str.replace('ensemble_', '')

# Create simplified pipeline names
def simplify_pipeline(name):
    if 'freesurfer8001ants243' in name:
        return 'FS8001'
    elif 'freesurfer741ants243' in name:
        return 'FS741'
    elif 'samseg8001ants243' in name:
        return 'SAMSEG8'
    elif 'fslanat6071ants243' in name:
        return 'FSL6071'
    elif 'all_pipelines' in name:
        return 'Data Aggregation'
    elif 'ensemble' in name:
        return 'Ensemble'
    else:
        return name

df['pipeline_simple'] = df['pipeline'].apply(simplify_pipeline)

# Set up global plotting style and colors
sns.set_style("whitegrid")
pipelines = df['pipeline_simple'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
color_dict = dict(zip(pipelines, colors))

def plot_sex_results(ax, data, sex_full_name):
    """Function to ensure identical plotting logic for every subplot."""
    models = data['model_clean'].unique()
    bar_width = 0.8
    
    # Get metadata for title
    n_sub = data['n_subjects'].iloc[0]
    n_samp = data['n_samples'].iloc[0]
    
    for model_idx, model in enumerate(models):
        model_data = data[data['model_clean'] == model]
        
        # Special handling for baseline
        if model == 'baseline':
            row = model_data.iloc[0]
            x_pos = model_idx
            ax.bar(x_pos, row['mean_mae'], width=bar_width, color='#7f7f7f', alpha=0.8)
            ax.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'], color='black', 
                         capsize=4, capthick=1, linewidth=1.5)
            ax.text(x_pos, row['mean_mae']/2, 'Baseline', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white', rotation=90)
        else:
            n_bars = len(model_data)
            for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
                x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
                ax.bar(x_pos, row['mean_mae'], width=bar_width/n_bars, 
                        color=color_dict[row['pipeline_simple']], alpha=0.8)
                ax.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'], color='black', 
                             capsize=4, capthick=1, linewidth=1.5)
                ax.text(x_pos, row['mean_mae']/2, row['pipeline_simple'], ha='center', 
                        va='center', fontsize=9, fontweight='bold', color='white', rotation=90)

    ax.set_title(f'{sex_full_name} Subjects - Brain Age Prediction MAE (n={int(n_sub)} subjects, k={int(n_samp)} scans)', 
                  fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0)
    ax.set_xlim(-0.5, len(models) - 0.5)

# Separate data
male_df = df[df['sex'] == 'M']
female_df = df[df['sex'] == 'F']

# --- VERSION 1: Side-by-Side ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
plot_sex_results(ax1, male_df, 'Male')
plot_sex_results(ax2, female_df, 'Female')
plt.tight_layout()
plt.savefig('brain_age_side_by_side_sexes.png', dpi=300, bbox_inches='tight')
plt.close()

# --- VERSION 2: Vertically Stacked ---
# Adjusted figsize for vertical aspect ratio
fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 16))
plot_sex_results(ax_top, male_df, 'Male')
plot_sex_results(ax_bottom, female_df, 'Female')
plt.tight_layout()
plt.savefig('brain_age_stacked_sexes.png', dpi=300, bbox_inches='tight')
plt.close()

print("Images saved: 'brain_age_side_by_side_sexes.png' and 'brain_age_stacked_sexes.png'")
