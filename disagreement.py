#!/usr/bin/env python3
"""
Pipeline disagreement heatmap generator - Single-Pass Global Intersection
- REMOVED: Separate Integrity Check phase (starts computations immediately).
- FIXED: Global Intersection enforced during computation.
  (If ANY pipeline fails for a subject, that subject is dropped from ALL maps).
- ARCHITECTURE: Subject-centric processing (Reads files once, computes all pairs).
"""

import os, argparse, numpy as np, nibabel as nib, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR_PATH = os.path.expanduser("~/Desktop/duckysets")
MNI_TEMPLATE_PATH = os.path.join(ROOT_DIR_PATH, "MNI152_T1_1mm_Brain.nii.gz")
# NOTE: The order of keys here determines the index for the 4-tuple loaded in the worker
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz",
    "freesurfer8001ants243": "aseg_MNI.nii.gz",
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
}
# Define the pairs we want to compute
PAIRS_TO_COMPUTE = [
    ("freesurfer741ants243", "freesurfer8001ants243"),
    ("freesurfer741ants243", "fslanat6071ants243"),
    ("freesurfer741ants243", "samseg8001ants243"),
    ("freesurfer8001ants243", "fslanat6071ants243"),
    ("freesurfer8001ants243", "samseg8001ants243"),
    ("fslanat6071ants243", "samseg8001ants243")
]

FSL_LABELS = [10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
TEST_SUBJECTS = None  
OUTPUT_CSV = os.path.join(ROOT_DIR_PATH, "pipeline_disagreement_statistics.csv")
OUTPUT_HIST_PNG = os.path.join(ROOT_DIR_PATH, "pipeline_disagreement_histograms.png")
SLICES_TO_SHOW = 3
BEST_SLICES = [59, 74, 87]

# -----------------------------
# Helper functions
# -----------------------------
def extract_ids(path, root_dir):
    path_parts = Path(path).parts
    root_parts = Path(root_dir).parts
    for i in range(len(path_parts) - len(root_parts) + 1):
        if path_parts[i:i+len(root_parts)] == root_parts:
            dataset = path_parts[i + len(root_parts)] if i + len(root_parts) < len(path_parts) else "unknown"
            break
    else: dataset = "unknown"
    sub = next((p for p in path_parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in path_parts if p.startswith("ses-")), "nosess")
    return dataset, sub, ses

def load_and_resample(img_path, ref_img):
    """Robust loading with header validation and resampling."""
    if not os.path.exists(img_path): raise FileNotFoundError(f"Missing: {img_path}")
    if os.path.getsize(img_path) == 0: raise EOFError(f"Empty: {img_path}")
    img = nib.load(img_path)
    if img.shape == ref_img.shape and np.allclose(img.affine, ref_img.affine, atol=1e-5):
        return img.get_fdata(dtype=np.float32).astype(np.int16)
    return resample_from_to(img, (ref_img.shape, ref_img.affine), order=0).get_fdata(dtype=np.float32).astype(np.int16)

def find_files_for_pipeline(root, pipeline, target_name):
    pattern = os.path.join(root, f"**/{pipeline}/**/{target_name}")
    files = sorted(glob(pattern, recursive=True))
    if TEST_SUBJECTS: files = files[:TEST_SUBJECTS]
    return files

# -----------------------------
# Subject-Centric Worker (The Global Intersection Fix)
# -----------------------------
def process_subject_all_pipelines(uid, pipeline_paths, ref_img, pipeline_names):
    """
    1. Loads ALL 4 pipeline images for this subject.
    2. Performs Integrity Check on ALL 4.
    3. If any fail -> Returns None (Subject dropped from Global Intersection).
    4. If all pass -> Computes and returns ALL 6 pairwise XOR maps.
    """
    loaded_masks = {}
    
    try:
        # Step 1: Load and Validate ALL pipelines first
        for pl_name in pipeline_names:
            fpath = pipeline_paths.get(pl_name)
            if not fpath: return None # Should not happen given initial intersection, but safe
            
            # Load
            data = load_and_resample(fpath, ref_img)
            
            # --- INTEGRITY CHECK ---
            # A. Check for empty mask
            if np.sum(data) == 0:
                return None
            
            # B. Check for missing labels (partial failure)
            unique_vals = np.unique(data)
            # Efficient subset check
            if not set(FSL_LABELS).issubset(set(unique_vals)):
                return None
            # -----------------------
            
            # Convert to boolean mask for XOR operations
            loaded_masks[pl_name] = np.isin(data, FSL_LABELS)

        # Step 2: Compute All Pairs (only if we survived Step 1)
        results = {}
        for (p1, p2) in PAIRS_TO_COMPUTE:
            # XOR and cast to float32 for accumulation
            xor_map = np.logical_xor(loaded_masks[p1], loaded_masks[p2]).astype(np.float32)
            results[f"{p1}_vs_{p2}"] = xor_map
            
        return results

    except Exception:
        # Any file read error or crash -> Subject is dropped
        return None

# -----------------------------
# Visualization Function
# -----------------------------
def create_visualization(disagreement_maps, vmax_mode, ref_img, mni_template, global_n, max_dis_values):
    all_vals = np.concatenate([dm.ravel() for dm in disagreement_maps.values()])
    if vmax_mode == 'percentile99':
        vmax = np.percentile(all_vals, 99)
        vmax_label, title_suffix = f"99th percentile: {vmax:.3f}", " (99th Percentile Normalization)"
    else:
        vmax = np.max(all_vals)
        vmax_label, title_suffix = f"full range: {vmax:.3f}", " (Full Range)"
    
    print(f"\nCreating {vmax_mode} visualization: using vmax={vmax:.3f}")
    n_comp = len(disagreement_maps)
    fig, axes = plt.subplots(SLICES_TO_SHOW, n_comp, figsize=(4 * n_comp, 4 * SLICES_TO_SHOW))
    
    for r, sl_idx in enumerate(BEST_SLICES[::-1]):
        mni_z = nib.affines.apply_affine(ref_img.affine, [[0, 0, sl_idx]])[0][2]
        slice_label = f"Slice {sl_idx} z={mni_z:.1f}mm"
        
        # Use a fixed order for consistency
        sorted_keys = sorted(disagreement_maps.keys())
        
        for c, comp_name in enumerate(sorted_keys):
            dm = disagreement_maps[comp_name]
            ax = axes[r, c]
            ax.imshow(np.rot90(mni_template[:, :, sl_idx]), cmap='gray', alpha=0.7)
            im = ax.imshow(np.rot90(dm[:, :, sl_idx]), cmap="hot", vmin=0, vmax=vmax, alpha=0.8)
            
            if c == 0: 
                ax.text(-0.15, 0.5, slice_label, fontsize=9, rotation=90, va='center', ha='center', transform=ax.transAxes)
            
            if r == 0:
                p1, p2 = comp_name.split("_vs_")
                mapping = {'freesurfer741ants243': 'FS741', 'freesurfer8001ants243': 'FS8001', 'fslanat6071ants243': 'FSL6071', 'samseg8001ants243': 'Samseg8'}
                
                # Global N is used here
                ax.set_title(f"{mapping.get(p1,p1)} vs {mapping.get(p2,p2)}\nn={global_n}", fontsize=10, pad=10)
            
            ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("Proportion of subjects with pipeline disagreement", rotation=270, labelpad=15)
    
    ann_text = f"Color scale: {vmax_label}\nMax observed: {max(max_dis_values.values()):.3f}"
    fig.text(0.92, 0.05, ann_text, fontsize=9, ha='center', transform=fig.transFigure, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplots_adjust(right=0.9, hspace=0.2, wspace=0.1)
    plt.suptitle(f"Pipeline Disagreement Overlaid on MNI Template{title_suffix}", fontsize=14, y=0.95)
    out = os.path.join(ROOT_DIR_PATH, f"pipeline_disagreement_heatmap_{vmax_mode}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved to: {out}")
    return out

# -----------------------------
# Main Execution
# -----------------------------
def main(max_threads=10):
    pipeline_subj_map = {}
    ordered_pipeline_names = list(PIPELINES.keys())
    
    print("Indexing files...")
    for pl in ordered_pipeline_names:
        fname = PIPELINES[pl]
        files = find_files_for_pipeline(ROOT_DIR_PATH, pl, fname)
        subj_map = {f"{extract_ids(f, ROOT_DIR_PATH)[0]}__{extract_ids(f, ROOT_DIR_PATH)[1]}__{extract_ids(f, ROOT_DIR_PATH)[2]}": f for f in files}
        pipeline_subj_map[pl] = subj_map
        print(f"  {pl}: found {len(files)} total files")

    # 1. Initial Match (File Existence Only)
    matched_uids = sorted(list(set.intersection(*[set(m.keys()) for m in pipeline_subj_map.values()])))
    print(f"\nFound {len(matched_uids)} candidates (files exist in all 4). Starting computations...")
    
    if not matched_uids: raise RuntimeError("No matched scans found.")
    
    # Load Reference
    ref_img = nib.load(pipeline_subj_map[ordered_pipeline_names[0]][matched_uids[0]])
    
    print("Loading MNI template...")
    mni_template_img = nib.load(MNI_TEMPLATE_PATH)
    mni_template_resampled = resample_from_to(mni_template_img, (ref_img.shape, ref_img.affine))
    mni_template = mni_template_resampled.get_fdata()

    # Initialize Accumulators for all 6 pairs
    accumulators = {f"{p1}_vs_{p2}": np.zeros(ref_img.shape, dtype=np.float32) for p1, p2 in PAIRS_TO_COMPUTE}
    valid_global_n = 0
    
    print("\nComputing Disagreement Maps (Subject-by-Subject)...")
    
    # ---------------------------------------------------------
    # MAIN LOOP - Subject Centric (Enforces Global Intersection)
    # ---------------------------------------------------------
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        active_futures = set()
        uid_iter = iter(matched_uids)
        
        # Helper to submit a task
        def submit_next():
            try:
                uid = next(uid_iter)
                # Pack all paths for this subject
                paths = {pl: pipeline_subj_map[pl][uid] for pl in ordered_pipeline_names}
                return executor.submit(process_subject_all_pipelines, uid, paths, ref_img, ordered_pipeline_names)
            except StopIteration:
                return None

        # Seed the queue
        for _ in range(max_threads * 2):
            f = submit_next()
            if f: active_futures.add(f)

        pbar = tqdm(total=len(matched_uids), desc="Processing Subjects")
        
        while active_futures:
            done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
            
            for f in done:
                res = f.result()
                if res is not None:
                    # If res is not None, the subject passed integrity checks for ALL 4 pipelines
                    valid_global_n += 1
                    # Add results to the global accumulators
                    for pair_key, xor_map in res.items():
                        accumulators[pair_key] += xor_map
                
                pbar.update(1)
                
                # Submit new task
                new_f = submit_next()
                if new_f: active_futures.add(new_f)
                
        pbar.close()

    print(f"\nComputation Complete.")
    print(f"Final Valid Global N: {valid_global_n} (subjects valid in ALL pipelines)")
    print(f"Dropped: {len(matched_uids) - valid_global_n} subjects during processing.")
    
    if valid_global_n == 0:
        raise RuntimeError("All subjects failed validation! Check your data/labels.")

    # -----------------------------
    # Post-Processing & Saving
    # -----------------------------
    disagreement_maps = {}
    disagreement_stats = []
    max_dis_values = {}

    for comp_name, acc_map in accumulators.items():
        # Normalize by the GLOBAL valid N
        acc_map /= valid_global_n
        disagreement_maps[comp_name] = acc_map
        max_val = float(np.max(acc_map))
        max_dis_values[comp_name] = max_val
        
        stats = {
            'comparison': comp_name,
            'n_subjects': valid_global_n, # Identical for all
            'mean_disagreement': float(np.mean(acc_map)),
            'max_disagreement': max_val,
            'median_disagreement': float(np.median(acc_map)),
            'std_disagreement': float(np.std(acc_map)),
            'n_voxels_high_disagreement': int(np.sum(acc_map >= 0.5)),
            'n_voxels_medium_disagreement': int(np.sum(acc_map >= 0.25)),
            'n_voxels_any_disagreement': int(np.sum(acc_map > 0)),
            'total_voxels': int(np.prod(acc_map.shape))
        }
        disagreement_stats.append(stats)

    pd.DataFrame(disagreement_stats).to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*100 + "\nPIPELINE DISAGREEMENT SUMMARY (Global Intersection)\n" + "="*100)
    for s in disagreement_stats:
        print(f"{s['comparison']:45} | Mean: {s['mean_disagreement']:5.3f} | N: {s['n_subjects']}")

    print("\n" + "="*80 + "\nCREATING VISUALIZATIONS\n" + "="*80)
    
    create_visualization(disagreement_maps, 'percentile99', ref_img, mni_template, valid_global_n, max_dis_values)
    create_visualization(disagreement_maps, 'fullrange', ref_img, mni_template, valid_global_n, max_dis_values)

    print("\nCreating histogram figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    all_nonzero = []
    
    sorted_keys = sorted(disagreement_maps.keys())
    for idx, name in enumerate(sorted_keys):
        dm = disagreement_maps[name]
        ax = axes[idx]
        nonzero = dm.ravel()[dm.ravel() > 0]
        all_nonzero.extend(nonzero)
        ax.hist(nonzero, bins=50, alpha=0.7, color='steelblue', log=True, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='25% threshold')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='50% threshold')
        ax.set_title(f"{name.split('_vs_')[0][:10]} vs {name.split('_vs_')[1][:10]}", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    if len(disagreement_maps) < len(axes):
        ax = axes[len(disagreement_maps)]
        ax.hist(all_nonzero, bins=100, alpha=0.7, color='darkgreen', edgecolor='black', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_title("All Comparisons Combined", fontsize=10)
    
    plt.suptitle("Distribution of Pipeline Disagreement Proportions (Non-Zero Voxels Only)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_HIST_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved histogram figure to: {OUTPUT_HIST_PNG}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=10, help="Number of worker threads (cores to use)")
    main(parser.parse_args().threads)
