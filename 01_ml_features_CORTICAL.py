import pandas as pd
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
# Set output directory to current working directory
EXPERIMENT_STATE_ROOT = Path.cwd() 
STATE_DIR = Path.cwd()

# Structures to exclude from ASEG (Subcortical)
FREESURFER_EXCLUDE = {
    "WM-hypointensities",
    "Optic-Chiasm",
    "Right-vessel",
    "Left-vessel",
    "non-WM-hypointensities",
    "5th-Ventricle",
    "Unknown",
}

# ----------------------------
# File parsers
# ----------------------------

def parse_freesurfer_aseg(path: Path):
    """
    Parses FreeSurfer aseg.stats (Subcortical Volumes).
    """
    results = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                struct_name = parts[4]
                try:
                    volume = float(parts[3])
                except ValueError:
                    continue
                
                # Normalize name
                if struct_name == "Brain-Stem":
                    struct_name = "Brainstem"
                
                # Exclude unwanted structures
                if struct_name in FREESURFER_EXCLUDE:
                    continue
                
                if volume >= 0.0: 
                    results[struct_name] = volume
    except Exception as e:
        print(f"Error parsing ASEG {path.name}: {e}")
        
    return results

def parse_freesurfer_cortical(path: Path):
    """
    Parses FreeSurfer cortical stats (DKTatlas, a2009s).
    Extracts both ThickAvg (col 4) and SurfArea (col 2).
    """
    results = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # cortical stats headers: 
                # StructName(0) NumVert(1) SurfArea(2) GrayVol(3) ThickAvg(4) ...
                struct_name = parts[0]
                
                if "???" in struct_name:
                    continue

                # 1. Extract Surface Area
                try:
                    surf_area = float(parts[2])
                    # Append suffix so downstream logic sees a unique feature name
                    results[f"{struct_name}_SurfArea"] = surf_area
                except ValueError:
                    pass

                # 2. Extract Average Thickness
                try:
                    thickness = float(parts[4])
                    results[f"{struct_name}_ThickAvg"] = thickness
                except ValueError:
                    pass
                    
    except Exception as e:
        print(f"Error parsing Cortical Stats {path.name}: {e}")

    return results

# Map file_type to the specific parser
PARSERS = {
    "aseg": parse_freesurfer_aseg,
    "cortical": parse_freesurfer_cortical,
}

# ----------------------------
# Discovery
# ----------------------------
def discover_files(state_dir: Path):
    """
    Recursively discovers FreeSurfer stats files.
    Updated to detect datasets by the presence of a 'derivatives' folder
    rather than a specific folder suffix.
    """
    results = []
    
    # Iterate over top-level containers
    for top_level_dir in state_dir.iterdir():
        if not top_level_dir.is_dir() or top_level_dir.name.startswith('.'):
            continue
        
        # --- MODIFIED LOGIC START ---
        # We look inside the top_level_dir for any folder that looks like a dataset.
        # A folder is considered a dataset if it contains a "derivatives" subdirectory.
        dataset_dir = None
        dataset_name = None
        
        for inner_dir in top_level_dir.iterdir():
            if inner_dir.is_dir():
                # Check if this inner_dir has a derivatives folder
                possible_derivatives = inner_dir / "derivatives"
                if possible_derivatives.exists() and possible_derivatives.is_dir():
                    dataset_dir = inner_dir
                    dataset_name = inner_dir.name
                    break # Assuming one dataset per top-level folder
        # --- MODIFIED LOGIC END ---
        
        if dataset_dir is None:
            continue
            
        derivatives = dataset_dir / "derivatives"
            
        # Check pipelines (e.g., freesurfer741ants243)
        for pipeline_root in derivatives.iterdir():
            if not pipeline_root.is_dir():
                continue
            pipeline_name = pipeline_root.name
            
            # Check versions
            for version_dir in pipeline_root.iterdir():
                if not version_dir.is_dir():
                    continue
                version = version_dir.name
                output_dir = version_dir / "output"
                if not output_dir.exists():
                    continue
                    
                # Iterate subjects
                for subj_dir in output_dir.iterdir():
                    if not subj_dir.is_dir():
                        continue
                    subj = subj_dir.name
                    
                    # Iterate sessions
                    for ses_dir in subj_dir.iterdir():
                        if not ses_dir.is_dir():
                            continue
                        ses = ses_dir.name
                        
                        # --- Stats Directory ---
                        stats_dir = ses_dir / subj / "stats"
                        if not stats_dir.exists():
                            continue

                        # 1. Subcortical Volumes (aseg)
                        aseg_path = stats_dir / "aseg.stats"
                        if aseg_path.exists():
                            results.append({
                                "dataset": dataset_name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "aseg",
                                "feature_group": "volume",
                                "hemi": None,
                                "path": aseg_path,
                            })

                        # 2. Cortical Thickness (DKTatlas & a2009s)
                        atlases = ["aparc.DKTatlas", "aparc.a2009s"]
                        hemispheres = ["lh", "rh"]
                        
                        for atlas in atlases:
                            group_name = atlas.replace("aparc.", "") 
                            
                            for hemi in hemispheres:
                                fname = f"{hemi}.{atlas}.stats"
                                fpath = stats_dir / fname
                                
                                if fpath.exists():
                                    results.append({
                                        "dataset": dataset_name,
                                        "pipeline": pipeline_name,
                                        "version": version,
                                        "subject": subj,
                                        "session": ses,
                                        "file_type": "cortical",
                                        "feature_group": group_name,
                                        "hemi": hemi,
                                        "path": fpath,
                                    })

    return results

# ----------------------------
# Build tidy DataFrame
# ----------------------------
def build_tidy_dataframe(files_meta):
    tidy_rows = []
    for r in files_meta:
        parser = PARSERS.get(r["file_type"])
        if parser is None:
            continue
        
        try:
            features = parser(r["path"])
            for struct, value in features.items():
                if r["hemi"]:
                    final_struct_name = f"{r['hemi']}_{struct}"
                else:
                    final_struct_name = struct

                tidy_rows.append({
                    "dataset": r["dataset"],
                    "pipeline": r["pipeline"],
                    "version": r["version"],
                    "subject": r["subject"],
                    "session": r["session"],
                    "feature_group": r["feature_group"],
                    "structure": final_struct_name,
                    "value": value,
                })
        except Exception as e:
            print(f"Error processing {r['path']}: {e}")
            continue
            
    return pd.DataFrame(tidy_rows)

# ----------------------------
# Wide pivot
# ----------------------------
def pivot_wide(df_tidy: pd.DataFrame):
    if df_tidy.empty:
        return pd.DataFrame()

    df_wide = df_tidy.pivot_table(
        index=["dataset", "subject", "session"],
        columns=["pipeline", "feature_group", "structure"],
        values="value"
    )
    
    df_wide = df_wide.dropna(axis=0, how='all')
    df_wide = df_wide.dropna(axis=1, how='all')
    
    new_cols = []
    for pipe, group, struct in df_wide.columns:
        col_name = f"{pipe}__{group}__{struct}"
        new_cols.append(col_name)
    
    df_wide.columns = new_cols
    df_wide = df_wide.reset_index()
    return df_wide

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"Output directory: {EXPERIMENT_STATE_ROOT.absolute()}")
    
    files_meta = discover_files(STATE_DIR)
    print(f"Discovered {len(files_meta)} stats files")

    df_tidy = build_tidy_dataframe(files_meta)
    print(f"Tidy DataFrame shape: {df_tidy.shape}")
    
    tidy_path = EXPERIMENT_STATE_ROOT / "df_tidy_cortical.csv"
    df_tidy.to_csv(tidy_path, index=False)
    
    if not df_tidy.empty:
        df_wide = pivot_wide(df_tidy)
        print(f"Wide DataFrame shape: {df_wide.shape}")
        
        wide_path = EXPERIMENT_STATE_ROOT / "morphological_features_CORTICAL.csv"
        df_wide.to_csv(wide_path, index=False)
        print(f"Saved wide DataFrame to: {wide_path.absolute()}")
        
        print("\nFeature Groups found:")
        print(df_tidy['feature_group'].value_counts())
        
        print("\nPipelines found:")
        print(df_tidy['pipeline'].value_counts())
    else:
        print("No data found. Check directory structure.")
