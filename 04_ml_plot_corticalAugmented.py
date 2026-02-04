import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

# Load the data
df = pd.read_csv('brain_age_results_sex_split_and_correct_ensembles.csv')

# Clean up model names by removing 'ensemble_' prefix
df['model_clean'] = df['model'].str.replace('ensemble_', '')

# --- FIXED: Updated logic to match the actual CSV filenames ---
def simplify_pipeline(name):
    # Handle Ensembles and Aggregation first
    if 'ensemble' in name:
        return 'Ensemble'
    elif 'all_pipelines' in name:
        return 'Data Aggregation'
    
    # Determine Base Tool
    tool = 'FS8.0' if 'fs800' in name else 'FS7.4' if 'fs741' in name else name
    
    # Determine Atlas/Features to make labels distinct but short
    if 'DKT_a2009s' in name:
        return f'{tool} (Comb)'
    elif 'a2009s' in name:
        return f'{tool} (Destr)'
    elif 'DKT' in name:
        return f'{tool} (DKT)'
    
    return tool

df['pipeline_simple'] = df['pipeline'].apply(simplify_pipeline)

# Set up the plotting style
sns.set_style("whitegrid")

# Create side-by-side plots for Male and Female
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

# Get both n_subjects and n_samples for titles
male_n_sub = df[df['sex'] == 'M']['n_subjects'].iloc[0]
male_n_samp = df[df['sex'] == 'M']['n_samples'].iloc[0]

female_n_sub = df[df['sex'] == 'F']['n_subjects'].iloc[0]
female_n_samp = df[df['sex'] == 'F']['n_samples'].iloc[0]

# --- FIXED: Dynamic color generation to prevent KeyError ---
pipelines = sorted(df['pipeline_simple'].unique())
# Use a colormap that supports enough distinct categories (tab10 has 10 colors)
color_map = plt.get_cmap('tab10')
color_dict = {pipe: color_map(i) for i, pipe in enumerate(pipelines)}

# Male subjects - manual plotting
male_data = df[df['sex'] == 'M']
models = male_data['model_clean'].unique()

bar_width = 0.8  # Thick bars
for model_idx, model in enumerate(models):
    model_data = male_data[male_data['model_clean'] == model]
    
    # Special handling for baseline
    if model == 'baseline':
        row = model_data.iloc[0]
        x_pos = model_idx
        
        ax1.bar(x_pos, row['mean_mae'], width=bar_width, 
                color='#7f7f7f', alpha=0.8)
        
        ax1.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                     color='black', capsize=4, capthick=1, linewidth=1.5)
        
        ax1.text(x_pos, row['mean_mae']/2, 'Baseline',
                 ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                 rotation=90)
    
    else:
        n_bars = len(model_data)
        for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
            # Calculate offset for grouped bars
            x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
            
            pipeline_name = row['pipeline_simple']
            
            # Plot bar
            ax1.bar(x_pos, row['mean_mae'], width=bar_width/n_bars, 
                    color=color_dict[pipeline_name], alpha=0.8)
            
            # Error bars
            ax1.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                         color='black', capsize=4, capthick=1, linewidth=1.5)
            
            # Pipeline label
            ax1.text(x_pos, row['mean_mae']/2, pipeline_name,
                     ha='center', va='center', fontsize=8, fontweight='bold', color='white',
                     rotation=90)

ax1.set_title(f'Male Subjects - Brain Age Prediction MAE (n={int(male_n_sub)} subjects, k={int(male_n_samp)} scans)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Model Type', fontsize=12)
ax1.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=0)
ax1.set_xlim(-0.5, len(models) - 0.5)

# Female subjects - manual plotting
female_data = df[df['sex'] == 'F']
for model_idx, model in enumerate(models):
    model_data = female_data[female_data['model_clean'] == model]
    
    if model == 'baseline':
        row = model_data.iloc[0]
        x_pos = model_idx
        
        ax2.bar(x_pos, row['mean_mae'], width=bar_width,
                color='#7f7f7f', alpha=0.8)
        
        ax2.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                     color='black', capsize=4, capthick=1, linewidth=1.5)
        
        ax2.text(x_pos, row['mean_mae']/2, 'Baseline',
                 ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                 rotation=90)
    
    else:
        n_bars = len(model_data)
        for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
            x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
            pipeline_name = row['pipeline_simple']
            
            ax2.bar(x_pos, row['mean_mae'], width=bar_width/n_bars,
                    color=color_dict[pipeline_name], alpha=0.8)
            
            ax2.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                         color='black', capsize=4, capthick=1, linewidth=1.5)
            
            ax2.text(x_pos, row['mean_mae']/2, pipeline_name,
                     ha='center', va='center', fontsize=8, fontweight='bold', color='white',
                     rotation=90)

ax2.set_title(f'Female Subjects - Brain Age Prediction MAE (n={int(female_n_sub)} subjects, k={int(female_n_samp)} scans)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Model Type', fontsize=12)
ax2.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=0)
ax2.set_xlim(-0.5, len(models) - 0.5)

plt.tight_layout()

# Save as PNG
plt.savefig('brain_age_side_by_side_sexes_BOTH.png', dpi=300, bbox_inches='tight')
plt.close()

print("PNG image saved: brain_age_side_by_side_sexes.png")
