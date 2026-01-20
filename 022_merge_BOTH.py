import pandas as pd

def main():
    # File names
    cortical_file = 'ml_dataset_with_age_sex_CORTICAL.csv'
    volumetric_file = 'ml_dataset_with_age_sex.csv'
    output_file = 'ml_dataset_with_age_sex_BOTH.csv'

    # 1. Load the CORTICAL dataset
    print(f"Loading {cortical_file}...")
    try:
        df_cortical = pd.read_csv(cortical_file)
    except FileNotFoundError:
        print(f"Error: Could not find {cortical_file}")
        return

    # Delete all columns with '__volume__' in their name from the CORTICAL dataset
    # We filter list comprehension to find matching columns
    cols_to_drop = [col for col in df_cortical.columns if '__volume__' in col]
    print(f"Dropping {len(cols_to_drop)} columns containing '__volume__' from cortical dataset.")
    df_cortical.drop(columns=cols_to_drop, inplace=True)

    # 2. Load the Volumetric dataset
    print(f"Loading {volumetric_file}...")
    try:
        df_vol = pd.read_csv(volumetric_file)
    except FileNotFoundError:
        print(f"Error: Could not find {volumetric_file}")
        return

    # Define the renaming logic
    def rename_column(col_name):
        # List of key columns that should NOT be renamed
        keys = ['dataset', 'subject', 'session', 'age', 'sex']
        if col_name in keys:
            return col_name
        
        # Split on the first double underscore to separate tool name from label
        # Example: freesurfer741ants243__label_10__volume
        parts = col_name.split('__', 1)
        
        if len(parts) == 2:
            tool_name = parts[0]  # e.g., freesurfer741ants243
            rest = parts[1]       # e.g., label_10__volume
            
            # Construct the new base name by inserting '__volume__'
            new_name = f"{tool_name}__volume__{rest}"
            
            # Special case: if it ends with '__volume', replace it with '__mm'
            if new_name.endswith('__volume'):
                new_name = new_name[:-8] + '__mm'
                
            return new_name
        
        return col_name

    # Apply renaming to the volumetric columns
    original_cols = df_vol.columns.tolist()
    new_cols = {col: rename_column(col) for col in original_cols}
    df_vol.rename(columns=new_cols, inplace=True)
    print("Renamed columns in volumetric dataset.")

    # 3. Merge the datasets
    # Merging on dataset, subject, session, age, sex
    merge_keys = ['dataset', 'subject', 'session', 'age', 'sex']
    
    print("Merging datasets...")
    # using 'inner' merge to keep only rows that exist in both files
    df_merged = pd.merge(df_cortical, df_vol, on=merge_keys, how='inner')

    # 4. Save the result
    df_merged.to_csv(output_file, index=False)
    print(f"Successfully merged data saved to {output_file}")
    print(f"Final shape: {df_merged.shape}")

if __name__ == "__main__":
    main()
