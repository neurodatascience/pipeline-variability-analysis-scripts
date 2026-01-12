import json
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT
EXPERIMENT_STATE_ROOT = ROOT

# ----------------------------
# Age/Sex extraction logic - FIXED PREVENT-AD (MULTI-SOURCE AGE)
# ----------------------------
def load_tabular_data(dataset_path: Path):
    """Load all TSVs in the tabular/ directory."""
    tabular = dataset_path / "tabular"
    if not tabular.exists():
        return {}

    dfs = {}
    for tsv in tabular.glob("*.tsv"):
        try:
            dfs[tsv.stem] = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"Could not read {tsv}: {e}")
    return dfs

def extract_demographics(dataset_name: str, dfs: dict):
    """Dataset-specific logic to extract age and sex information."""
    df_demo = None
    dataset_lower = dataset_name.lower()

    # PREVENT-AD logic (UPDATED: Dual Age Sources + Sex)
    if dataset_lower == 'preventad':
        # --- 1. Extract Sex (Subject-Level) ---
        df_sex = pd.DataFrame(columns=['subject', 'sex'])
        if 'demographics' in dfs:
            d = dfs['demographics'].copy()
            if 'participant_id' in d.columns and 'Sex' in d.columns:
                # Normalize ID
                d['participant_id'] = d['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
                df_sex = d[['participant_id', 'Sex']].rename(columns={'participant_id': 'subject', 'Sex': 'sex'})

        # --- 2. Extract Age (Session-Level) ---
        age_dfs = []

        # Source A: mci_status.tsv
        if 'mci_status' in dfs:
            mci = dfs['mci_status'].copy()
            
            # Normalize ID
            if 'participant_id' in mci.columns:
                mci['subject'] = mci['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            # Calculate Age (Candidate_Age in Months -> Years)
            age_col = None
            if 'Candidate_Age' in mci.columns: age_col = 'Candidate_Age'
            elif 'Age' in mci.columns: age_col = 'Age'
            elif 'age_months' in mci.columns: age_col = 'age_months'
            elif len(mci.columns) > 3 and mci.iloc[:, 3].dtype.kind in 'fi':
                 if 'Unnamed: 3' in mci.columns: age_col = 'Unnamed: 3'
            
            if age_col and 'subject' in mci.columns and 'visit_id' in mci.columns:
                mci['age'] = pd.to_numeric(mci[age_col], errors='coerce') / 12.0
                
                # Generate Session Variations
                # V1: Strict (NAPFU12 -> ses-NAPFU12)
                v1 = mci[['subject', 'visit_id', 'age']].copy()
                v1['session'] = v1['visit_id'].astype(str).str.strip().apply(
                    lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                )
                
                # V2: PRE->FU (PREFU12 -> ses-FU12)
                v2 = mci[['subject', 'visit_id', 'age']].copy()
                v2['session'] = v2['visit_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('PREFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                # V3: NAP->FU (NAPFU12 -> ses-FU12)
                v3 = mci[['subject', 'visit_id', 'age']].copy()
                v3['session'] = v3['visit_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('NAPFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)

                age_dfs.extend([v1[['subject', 'session', 'age']], 
                                v2[['subject', 'session', 'age']], 
                                v3[['subject', 'session', 'age']]])

        # Source B: mri_sessions-phase1.tsv
        if 'mri_sessions-phase1' in dfs:
            mri = dfs['mri_sessions-phase1'].copy()
            
            # Normalize ID
            if 'participant_id' in mri.columns:
                mri['subject'] = mri['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            # Calculate Age (Column 'age' is in Months -> Years)
            if 'age' in mri.columns and 'subject' in mri.columns and 'session_id' in mri.columns:
                mri['age_years'] = pd.to_numeric(mri['age'], errors='coerce') / 12.0
                
                # Generate Session Variations
                # V1: Strict
                v1 = mri[['subject', 'session_id', 'age_years']].copy()
                v1['session'] = v1['session_id'].astype(str).str.strip().apply(
                    lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                )
                
                # V2: PRE->FU
                v2 = mri[['subject', 'session_id', 'age_years']].copy()
                v2['session'] = v2['session_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('PREFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                # V3: NAP->FU
                v3 = mri[['subject', 'session_id', 'age_years']].copy()
                v3['session'] = v3['session_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('NAPFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                age_dfs.extend([v1[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v2[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v3[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'})])

        # --- 3. Combine & Finalize ---
        if age_dfs:
            # Concat all age records
            df_combined_ages = pd.concat(age_dfs, ignore_index=True)
            # Deduplicate: If same session in both files, keep first (mci_status)
            df_combined_ages = df_combined_ages.drop_duplicates(subset=['subject', 'session'], keep='first')
            
            # Merge with Sex
            if not df_sex.empty:
                df_demo = df_combined_ages.merge(df_sex, on='subject', how='left')
            else:
                df_demo = df_combined_ages
                df_demo['sex'] = np.nan
        else:
            # Fallback if no age data found
            df_demo = pd.DataFrame(columns=['subject', 'session', 'age', 'sex'])

        # Final Cleanup
        for col in ['subject', 'session', 'age', 'sex']:
            if col not in df_demo.columns:
                df_demo[col] = np.nan
        df_demo = df_demo[['subject', 'session', 'age', 'sex']]

    elif dataset_lower == 'ds005752':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    elif dataset_lower == 'ds003592':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    
    # Rockland / NKI logic
    elif 'rockland' in dataset_lower or 'nki' in dataset_lower:
        key = next((k for k in dfs.keys() if 'participants' in k), None)
        if key:
            df = dfs[key]
            if 'participant_id' in df.columns:
                df_demo = df.rename(columns={'participant_id': 'subject'})
                
                # Normalize Subject ID
                if 'subject' in df_demo.columns:
                    df_demo['subject'] = df_demo['subject'].astype(str).apply(
                        lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                    )

                # Capture Session ID if available and normalize
                if 'session_id' in df_demo.columns:
                    df_demo = df_demo.rename(columns={'session_id': 'session'})
                    df_demo['session'] = df_demo['session'].astype(str).apply(
                        lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                    )
                
                # Select available columns
                cols = ['subject']
                if 'session' in df_demo.columns: cols.append('session')
                if 'age' in df_demo.columns: cols.append('age')
                if 'sex' in df_demo.columns: cols.append('sex')
                df_demo = df_demo[cols]

    else:
        if 'participants' in dfs:
            df_demo = dfs['participants'].rename(columns={'participant_id': 'subject'})
            if 'age' not in df_demo.columns:
                df_demo['age'] = np.nan
            if 'sex' not in df_demo.columns:
                df_demo['sex'] = np.nan
    
    # Data Cleaning: Handle comma-separated values
    if df_demo is not None and not df_demo.empty:
        if 'age' in df_demo.columns:
            df_demo['age'] = df_demo['age'].astype(str).str.split(',').str[0]
            df_demo['age'] = df_demo['age'].replace('n/a', np.nan)
            df_demo['age'] = pd.to_numeric(df_demo['age'], errors='coerce')
        
        if 'sex' in df_demo.columns:
            df_demo['sex'] = df_demo['sex'].astype(str).str.split(',').str[0]
            df_demo['sex'] = df_demo['sex'].replace('n/a', np.nan)

            # Standardization logic
            non_nan_mask = df_demo['sex'].notna()
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].astype(str).str.upper()
            )
            standardization_map = {'MALE': 'M', 'FEMALE': 'F'}
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].replace(standardization_map)
            )
    
    # Final Return: Ensure 'session' column exists (fill NaN if missing)
    if df_demo is not None and not df_demo.empty:
        if 'session' not in df_demo.columns:
            df_demo['session'] = np.nan
        return df_demo[['subject', 'session', 'age', 'sex']]
    else:
        return pd.DataFrame(columns=['subject', 'session', 'age', 'sex'])

# ----------------------------
# Discover subjects and sessions - UNCHANGED
# ----------------------------
def discover_subjects(state_dir: Path):
    """Discover all subjects and sessions without parsing volume files."""
    results = []

    for dataset_outer in state_dir.iterdir():
        if not dataset_outer.is_dir():
            continue

        for dataset_dir in dataset_outer.iterdir():
            if not dataset_dir.is_dir():
                continue

            derivatives = dataset_dir / "derivatives"
            if not derivatives.exists():
                continue

            for pipeline_root in derivatives.iterdir():
                if not pipeline_root.is_dir():
                    continue

                for version_dir in pipeline_root.iterdir():
                    if not version_dir.is_dir():
                        continue

                    output_dir = version_dir / "output"
                    if not output_dir.exists():
                        continue

                    for subj_dir in output_dir.iterdir():
                        if not subj_dir.is_dir():
                            continue
                        subj = subj_dir.name

                        for ses_dir in subj_dir.iterdir():
                            if not ses_dir.is_dir():
                                continue
                            ses = ses_dir.name

                            # Just record the subject/session info
                            results.append({
                                "dataset": dataset_dir.name,
                                "subject": subj,
                                "session": ses,
                            })

    return results

# ----------------------------
# Main - STRICT MAPPING LOGIC (UNCHANGED from last successful version)
# ----------------------------
if __name__ == "__main__":
    print(f"Dataset root: {STATE_DIR}")
    print(f"Output CSV directory: {EXPERIMENT_STATE_ROOT}")

    # Discover all subjects and sessions
    subjects_meta = discover_subjects(STATE_DIR)
    print(f"Discovered {len(subjects_meta)} subject-session combinations")
    
    # Create initial DataFrame
    df_demo = pd.DataFrame(subjects_meta)
    
    # Add age AND sex column
    df_demo["age"] = np.nan
    df_demo["sex"] = np.nan 
    
    # Build age AND sex mapping
    age_mapping = {}
    sex_mapping = {} 
    
    for dataset in df_demo["dataset"].unique():
        dataset_path = None

        # Find the real dataset folder
        for outer in ROOT.iterdir():
            if outer.is_dir():
                inner = outer / dataset
                if inner.exists():
                    dataset_path = inner
                    break

        if dataset_path is None:
            continue

        dfs = load_tabular_data(dataset_path)
        df_demo_dataset = extract_demographics(dataset, dfs)

        if df_demo_dataset is None or df_demo_dataset.empty:
            continue

        # Build mappings for this dataset
        for _, row in df_demo_dataset.iterrows():
            subj = row["subject"]
            sess = row["session"]
            
            # --- STRICT LOGIC ---
            
            if pd.notna(sess):
                # We have session info: Map strictly to the session
                age_mapping[(dataset, subj, sess)] = row["age"]
                sex_mapping[(dataset, subj, sess)] = row["sex"]
                
                # Sex is constant per subject, so we CAN set the fallback (d,s)
                sex_mapping[(dataset, subj)] = row["sex"]
                
                # AGE: DO NOT set fallback here. 
                # This ensures that if a session is NOT in the TSV, it stays NaN.
            
            else:
                # No session info (e.g. ds005752): Set fallback mapping for everything
                age_mapping[(dataset, subj)] = row["age"]
                sex_mapping[(dataset, subj)] = row["sex"]

    # Apply the age AND sex mapping with session awareness
    def get_age(row):
        # 1. Try exact session match
        val = age_mapping.get((row["dataset"], row["subject"], row["session"]), np.nan)
        # 2. If not found, try fallback (dataset, subject)
        if pd.isna(val):
            val = age_mapping.get((row["dataset"], row["subject"]), np.nan)
        return val
    
    def get_sex(row):
        val = sex_mapping.get((row["dataset"], row["subject"], row["session"]), np.nan)
        if pd.isna(val):
            val = sex_mapping.get((row["dataset"], row["subject"]), np.nan)
        return val
    
    df_demo["age"] = df_demo.apply(get_age, axis=1)
    df_demo["sex"] = df_demo.apply(get_sex, axis=1) 
    
    # Remove duplicates
    df_demo = df_demo.drop_duplicates(subset=["dataset", "subject", "session"]).reset_index(drop=True)
    
    print(f"Demographics DataFrame shape: {df_demo.shape}")
    print(f"Subjects with age data: {df_demo['age'].notna().sum()}")
    print(f"Subjects with sex data: {df_demo['sex'].notna().sum()}")
    
    # Load morphological features and join
    features_path = ROOT / "morphological_features_mni.csv"
    if features_path.exists():
        print(f"Loading morphological features from: {features_path}")
        df_features = pd.read_csv(features_path)
        print(f"Morphological features shape: {df_features.shape}")
        
        # FIX THE TYPO: Change ds003592_nipoppyy to ds003592_nipoppy
        df_features['dataset'] = df_features['dataset'].replace('ds003592_nipoppyy', 'ds003592_nipoppy')
        
        # Join with demographics on dataset, subject, session
        df_merged = df_features.merge(
            df_demo[["dataset", "subject", "session", "age", "sex"]], 
            on=["dataset", "subject", "session"],
            how="left"
        )
        
        # Reorder columns
        demo_columns = ["dataset", "subject", "session", "age", "sex"]
        feature_columns = [col for col in df_merged.columns if col not in demo_columns]
        df_merged = df_merged[demo_columns + feature_columns]
        
        print(f"Merged DataFrame shape: {df_merged.shape}")
        print(f"Subjects with age data in merged set: {df_merged['age'].notna().sum()}")
        print(f"Subjects with sex data in merged set: {df_merged['sex'].notna().sum()}")
        
        # Save the merged result
        merged_output_path = EXPERIMENT_STATE_ROOT / "ml_dataset_with_age_sex.csv"
        df_merged.to_csv(merged_output_path, index=False)
        print(f"Saved merged ML dataset to: {merged_output_path}")
        
        # Show sample
        print("\nSample of merged data:")
        print(df_merged[["dataset", "subject", "session", "age", "sex"]].head(10))
    else:
        print(f"Morphological features file not found at: {features_path}")
        # Save just demographics as fallback
        output_path = EXPERIMENT_STATE_ROOT / "demographics_with_sex.csv"
        df_demo.to_csv(output_path, index=False)
        print(f"Saved demographics DataFrame to: {output_path}")
