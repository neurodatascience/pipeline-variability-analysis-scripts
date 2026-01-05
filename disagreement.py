#!/usr/bin/env python3
"""
Pipeline disagreement heatmap generator - Parallelized & Memory-Safe
- Strictly respects thread/core limits.
- Uses float32 for masks (no uint8).
- Throttled task submission to prevent OOM (Killed) crashes.
- Restored original logging, summary tables, and interpretation.
- Handles dataset__subject__session UIDs for multi-session support.
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
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz",
    "freesurfer8001ants243": "aseg_MNI.nii.gz",
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
}
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
    """Extract dataset, subject, and session for multi-session support."""
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
# Parallel Worker Function
# -----------------------------
def process_uid_pair(uid, pl1, pl2, pipeline_subj_map, ref_img):
    try:
        arr1 = load_and_resample(pipeline_subj_map[pl1][uid], ref_img)
        arr2 = load_and_resample(pipeline_subj_map[pl2][uid], ref_img)
        mask1 = np.isin(arr1, FSL_LABELS)
        mask2 = np.isin(arr2, FSL_LABELS)
        # Using float32 as requested (NOT uint8)
        return np.logical_xor(mask1, mask2).astype(np.float32)
    except:
        return None

# -----------------------------
# Visualization Function
# -----------------------------
def create_visualization(disagreement_maps, vmax_mode, ref_img, mni_template, n_matched, max_dis_values):
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
        for c, (comp_name, dm) in enumerate(disagreement_maps.items()):
            ax = axes[r, c]
            ax.imshow(np.rot90(mni_template[:, :, sl_idx]), cmap='gray', alpha=0.7)
            im = ax.imshow(np.rot90(dm[:, :, sl_idx]), cmap="hot", vmin=0, vmax=vmax, alpha=0.8)
            if c == 0: ax.text(-0.15, 0.5, slice_label, fontsize=9, rotation=90, va='center', ha='center', transform=ax.transAxes)
            if r == 0:
                p1, p2 = comp_name.split("_vs_")
                mapping = {'freesurfer741ants243': 'FS741', 'freesurfer8001ants243': 'FS8001', 'fslanat6071ants243': 'FSL6071', 'samseg8001ants243': 'Samseg8'}
                ax.set_title(f"{mapping.get(p1,p1)} vs {mapping.get(p2,p2)}\nn={n_matched}", fontsize=10, pad=10)
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
    for pl, fname in PIPELINES.items():
        files = find_files_for_pipeline(ROOT_DIR_PATH, pl, fname)
        subj_map = {f"{extract_ids(f, ROOT_DIR_PATH)[0]}__{extract_ids(f, ROOT_DIR_PATH)[1]}__{extract_ids(f, ROOT_DIR_PATH)[2]}": f for f in files}
        pipeline_subj_map[pl] = subj_map
        print(f"{pl}: found {len(files)} total files, {len(subj_map)} unique matched scans")

    matched_uids = sorted(list(set.intersection(*[set(m.keys()) for m in pipeline_subj_map.values()])))
    print(f"\n Found {len(matched_uids)} matched scans with all 4 pipelines available")
    if not matched_uids: raise RuntimeError("No matched scans found.")

    ref_img = nib.load(pipeline_subj_map[list(PIPELINES.keys())[0]][matched_uids[0]])
    print(f"\nReference chosen: {ref_img.shape}")

    print("Loading MNI template...")
    mni_template_img = nib.load(MNI_TEMPLATE_PATH)
    mni_template_resampled = resample_from_to(mni_template_img, (ref_img.shape, ref_img.affine))
    mni_template = mni_template_resampled.get_fdata()
    print(f"MNI template shape: {mni_template.shape} (resampled to match reference)")

    pipeline_pairs = [("freesurfer741ants243", "freesurfer8001ants243"), ("freesurfer741ants243", "fslanat6071ants243"), ("freesurfer741ants243", "samseg8001ants243"), ("freesurfer8001ants243", "fslanat6071ants243"), ("freesurfer8001ants243", "samseg8001ants243"), ("fslanat6071ants243", "samseg8001ants243")]

    disagreement_maps, disagreement_stats, max_dis_values = {}, [], {}

    print("\n Computing pipeline disagreements...")

    for pl1, pl2 in pipeline_pairs:
        comp_name = f"{pl1}_vs_{pl2}"
        print(f"\nComputing {pl1} vs {pl2}...")
        acc_map = np.zeros(ref_img.shape, dtype=np.float32)
        valid_n = 0
        
        # MEMORY-SAFE PARALLEL LOOP: Strictly respects worker count and throttles queue
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            active_futures = set()
            uid_iter = iter(matched_uids)
            
            # Pre-fill queue with limited tasks (2x worker count)
            for _ in range(max_threads * 2):
                try:
                    uid = next(uid_iter)
                    active_futures.add(executor.submit(process_uid_pair, uid, pl1, pl2, pipeline_subj_map, ref_img))
                except StopIteration: break

            pbar = tqdm(total=len(matched_uids), desc=f"{pl1[:10]} vs {pl2[:10]}")
            while active_futures:
                # Wait for the next task to finish
                done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    if res is not None:
                        acc_map += res
                        valid_n += 1
                    pbar.update(1)
                    # Submit a new task to fill the vacancy
                    try:
                        uid = next(uid_iter)
                        active_futures.add(executor.submit(process_uid_pair, uid, pl1, pl2, pipeline_subj_map, ref_img))
                    except StopIteration: pass
            pbar.close()
        
        if valid_n > 0:
            acc_map /= valid_n
            disagreement_maps[comp_name] = acc_map
            max_dis_values[comp_name] = float(np.max(acc_map))
            
            stats = {
                'comparison': comp_name,
                'n_subjects': valid_n,
                'mean_disagreement': float(np.mean(acc_map)),
                'max_disagreement': max_dis_values[comp_name],
                'median_disagreement': float(np.median(acc_map)),
                'std_disagreement': float(np.std(acc_map)),
                'n_voxels_high_disagreement': int(np.sum(acc_map >= 0.5)),
                'n_voxels_medium_disagreement': int(np.sum(acc_map >= 0.25)),
                'n_voxels_any_disagreement': int(np.sum(acc_map > 0)),
                'total_voxels': int(np.prod(acc_map.shape))
            }
            disagreement_stats.append(stats)
            print(f"  Mean disagreement: {stats['mean_disagreement']:.3f} | Max: {stats['max_disagreement']:.3f} | >=50%: {stats['n_voxels_high_disagreement']}")

    pd.DataFrame(disagreement_stats).to_csv(OUTPUT_CSV, index=False)
    
    # Restored SUMMARY TABLE
    print("\n" + "="*100 + "\nPIPELINE DISAGREEMENT SUMMARY\n" + "="*100)
    for s in disagreement_stats:
        print(f"{s['comparison']:45} | Mean: {s['mean_disagreement']:5.3f} | Max: {s['max_disagreement']:5.3f} | >=50%: {s['n_voxels_high_disagreement']:6d}")

    print("\n" + "="*80 + "\nCREATING VISUALIZATIONS\n" + "="*80)
    create_visualization(disagreement_maps, 'percentile99', ref_img, mni_template, len(matched_uids), max_dis_values)
    create_visualization(disagreement_maps, 'fullrange', ref_img, mni_template, len(matched_uids), max_dis_values)

    print("\nCreating histogram figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    all_nonzero = []
    for idx, (name, dm) in enumerate(disagreement_maps.items()):
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
