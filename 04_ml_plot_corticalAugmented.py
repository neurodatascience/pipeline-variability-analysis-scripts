import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('brain_age_results_sex_split_and_correct_ensembles.csv')

# Clean up model names
df['model_clean'] = df['model'].str.replace('ensemble_', '')

def simplify_pipeline(name):
    if 'ensemble' in name:
        return 'Ensemble'
    elif 'all_pipelines' in name:
        return 'Data Aggregation'
    
    tool = 'FS8.0' if 'fs800' in name else 'FS7.4' if 'fs741' in name else name
    
    if 'DKT_a2009s' in name:
        return f'{tool} (Comb)'
    elif 'a2009s' in name:
        return f'{tool} (Destr)'
    elif 'DKT' in name:
        return f'{tool} (DKT)'
    
    return tool

df['pipeline_simple'] = df['pipeline'].apply(simplify_pipeline)

# Set up global plotting parameters
sns.set_style("whitegrid")
pipelines = sorted(df['pipeline_simple'].unique())
color_map = plt.get_cmap('tab10')
color_dict = {pipe: color_map(i) for i, pipe in enumerate(pipelines)}

def plot_sex_data(ax, data, sex_label):
    """Function to render the specific bar logic for a given axis and sex."""
    models = data['model_clean'].unique()
    bar_width = 0.8
    
    n_sub = data['n_subjects'].iloc[0]
    n_samp = data['n_samples'].iloc[0]
    
    for model_idx, model in enumerate(models):
        model_data = data[data['model_clean'] == model]
        
        if model == 'baseline':
            row = model_data.iloc[0]
            x_pos = model_idx
            ax.bar(x_pos, row['mean_mae'], width=bar_width, color='#7f7f7f', alpha=0.8)
            ax.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'], color='black', capsize=4, capthick=1, linewidth=1.5)
            ax.text(x_pos, row['mean_mae']/2, 'Baseline', ha='center', va='center', fontsize=9, fontweight='bold', color='white', rotation=90)
        else:
            n_bars = len(model_data)
            for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
                x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
                pipeline_name = row['pipeline_simple']
                
                ax.bar(x_pos, row['mean_mae'], width=bar_width/n_bars, color=color_dict[pipeline_name], alpha=0.8)
                ax.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'], color='black', capsize=4, capthick=1, linewidth=1.5)
                ax.text(x_pos, row['mean_mae']/2, pipeline_name, ha='center', va='center', fontsize=8, fontweight='bold', color='white', rotation=90)

    ax.set_title(f'{sex_label} Subjects - Brain Age Prediction MAE (n={int(n_sub)} subjects, k={int(n_samp)} scans)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0)
    ax.set_xlim(-0.5, len(models) - 0.5)

# Prepare data subsets
male_df = df[df['sex'] == 'M']
female_df = df[df['sex'] == 'F']

# --- GENERATE VERSION 1: SIDE-BY-SIDE ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
plot_sex_data(ax1, male_df, 'Male')
plot_sex_data(ax2, female_df, 'Female')
plt.tight_layout()
plt.savefig('brain_age_side_by_side_sexes_BOTH.png', dpi=300, bbox_inches='tight')
plt.close()

# --- GENERATE VERSION 2: VERTICALLY STACKED ---
fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 16))
plot_sex_data(ax_top, male_df, 'Male')
plot_sex_data(ax_bottom, female_df, 'Female')
plt.tight_layout()
plt.savefig('brain_age_stacked_sexes_BOTH.png', dpi=300, bbox_inches='tight')
plt.close()

print("Both images saved: side-by-side and stacked.")
