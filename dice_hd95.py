#!/usr/bin/env python3
"""
Fast Dice and HD95 overlap computation between subcortical segmentations.
- Handles multiple datasets and parallelization.
- Resilient to missing/empty files (skips sessions gracefully).
- Generates plots immediately after each dataset is processed.
- Plot footers show total scan (session) counts.
"""
import os, argparse, numpy as np, nibabel as nib, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from glob import glob
from collections import OrderedDict
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    print("Warning: SimpleITK not available. Install with: pip install SimpleITK")
    SITK_AVAILABLE = False

# === OUTPUT IN CURRENT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

PIPELINES = {
    "freesurfer741ants243": "aseg.nii.gz",
    "freesurfer8001ants243": "aseg.nii.gz",
    "fslanat6071ants243": "T1_subcort_seg.nii.gz",
    "samseg8001ants243": "seg.nii.gz",
}

STRUCTURE_LABELS = OrderedDict([
    ("Left-Thalamus-Proper", 10),
    ("Right-Thalamus-Proper", 49),
    ("Left-Caudate", 11),
    ("Right-Caudate", 50),
    ("Left-Putamen", 12),
    ("Right-Putamen", 51),
    ("Left-Pallidum", 13),
    ("Right-Pallidum", 52),
    ("Left-Hippocampus", 17),
    ("Right-Hippocampus", 53),
    ("Left-Amygdala", 18),
    ("Right-Amygdala", 54),
    ("Left-Accumbens-area", 26),
    ("Right-Accumbens-area", 58),
    ("Brain-Stem/4thVentricle", 16),
])
LABELS = list(STRUCTURE_LABELS.values())

def find_files(root, pipeline, filename):
    return sorted(glob(os.path.join(root, f"**/{pipeline}/**/{filename}"), recursive=True))

def extract_ids(path, root_dir):
    """Extract dataset, subject, and session from path."""
    path_parts = Path(path).parts
    root_parts = Path(root_dir).parts
    for i in range(len(path_parts) - len(root_parts) + 1):
        if path_parts[i:i+len(root_parts)] == root_parts:
            if i + len(root_parts) < len(path_parts):
                dataset = path_parts[i + len(root_parts)]
            else:
                dataset = "unknown"
            break
    else:
        dataset = "unknown"
    sub = next((p for p in path_parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in path_parts if p.startswith("ses-")), "nosess")
    return dataset, sub, ses

def dice_per_label(arr1, arr2, labels):
    """Real Dice overlap computation."""
    dices = {}
    for lbl in labels:
        m1 = (arr1 == lbl)
        m2 = (arr2 == lbl)
        inter = np.logical_and(m1, m2).sum()
        v1, v2 = m1.sum(), m2.sum()
        dices[lbl] = 1.0 if v1 + v2 == 0 else 2 * inter / (v1 + v2)
    return dices

def fast_hd95_per_label(arr1, arr2, labels, voxel_spacing):
    """Real HD95 computation using SimpleITK."""
    if not SITK_AVAILABLE:
        return {lbl: np.nan for lbl in labels}
    hd95s = {}
    spacing_sitk = voxel_spacing[::-1]
    for lbl in labels:
        m1, m2 = (arr1 == lbl), (arr2 == lbl)
        if m1.sum() == 0 or m2.sum() == 0:
            hd95s[lbl] = np.nan
            continue
        try:
            sitk1 = sitk.GetImageFromArray(m1.astype(np.uint8))
            sitk2 = sitk.GetImageFromArray(m2.astype(np.uint8))
            sitk1.SetSpacing(spacing_sitk)
            sitk2.SetSpacing(spacing_sitk)
            contour1 = sitk.BinaryContour(sitk1, fullyConnected=False)
            contour2 = sitk.BinaryContour(sitk2, fullyConnected=False)
            dt1 = sitk.SignedMaurerDistanceMap(contour1, useImageSpacing=True, squaredDistance=False)
            dt2 = sitk.SignedMaurerDistanceMap(contour2, useImageSpacing=True, squaredDistance=False)
            dist1_to_2 = np.abs(sitk.GetArrayFromImage(dt2)[sitk.GetArrayFromImage(contour1) > 0])
            dist2_to_1 = np.abs(sitk.GetArrayFromImage(dt1)[sitk.GetArrayFromImage(contour2) > 0])
            if len(dist1_to_2) == 0 or len(dist2_to_1) == 0:
                hd95s[lbl] = np.nan
                continue
            hd95s[lbl] = max(np.percentile(dist1_to_2, 95), np.percentile(dist2_to_1, 95))
        except:
            hd95s[lbl] = np.nan
    return hd95s

def load_segmentation_as_int(path, ref_img=None):
    """Real data loading with existence and empty checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if os.path.getsize(path) == 0:
        raise EOFError(f"Empty file: {path}")
    img = nib.load(path)
    if ref_img is not None and (img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine, atol=1e-3)):
        img = resample_from_to(img, ref_img, order=0)
    data = img.get_fdata(dtype=np.float32).astype(np.int16)
    return data, img.affine, img

def get_heatmap_range(df, metric):
    """Calculate mean-based range for heatmap."""
    means = [group[metric].mean() for _, group in df.groupby(['pipeline1', 'pipeline2', 'structure']) if not np.isnan(group[metric].mean())]
    if not means: return 0, 1
    mmin, mmax = min(means), max(means)
    pad = (mmax - mmin) * 0.1
    return max(0, mmin - pad), (min(1, mmax + pad) if metric == 'dice' else mmax + pad)

def compute_for_subject(uid, ref_img, downsample_factor, metrics, pairs, dataset, STRUCTURE_LABELS, LABELS, PIPELINES, subj_maps):
    """Process a single session (uid) for all pipeline pairs."""
    try:
        _, sub, ses = uid.split("__")
        high_res_imgs = {}
        high_res_img_objects = {} 
        high_res_spacing = np.sqrt(np.sum(ref_img.affine[:3,:3]**2, axis=0))
        
        for pl in PIPELINES:
            data, _, img_obj = load_segmentation_as_int(subj_maps[pl][uid], ref_img)
            high_res_imgs[pl] = data
            high_res_img_objects[pl] = img_obj
            
        low_res_imgs, low_res_spacing = None, None
        if downsample_factor > 1.0 and 'hd95' in metrics:
            ds_affine = ref_img.affine.copy(); ds_affine[:3,:3] *= downsample_factor
            ds_shape = (np.array(ref_img.shape[:3]) // downsample_factor).astype(int)
            ds_ref = nib.Nifti1Image(np.zeros(ds_shape), ds_affine)
            low_res_spacing = np.sqrt(np.sum(ds_ref.affine[:3,:3]**2, axis=0))
            low_res_imgs = {pl: resample_from_to(high_res_img_objects[pl], ds_ref, order=0).get_fdata().astype(np.int16) for pl in PIPELINES}
        
        local_results = []
        for p1, p2 in pairs:
            m_res = {}
            if 'dice' in metrics:
                m_res['dice'] = dice_per_label(high_res_imgs[p1], high_res_imgs[p2], LABELS)
            if 'hd95' in metrics:
                a1 = low_res_imgs[p1] if low_res_imgs else high_res_imgs[p1]
                a2 = low_res_imgs[p2] if low_res_imgs else high_res_imgs[p2]
                sp = low_res_spacing if low_res_spacing is not None else high_res_spacing
                m_res['hd95'] = fast_hd95_per_label(a1, a2, LABELS, sp)
            
            for name, lbl in STRUCTURE_LABELS.items():
                row = {"dataset": dataset, "subject": sub, "session": ses, "pipeline1": p1, "pipeline2": p2, "structure": name, "label": lbl}
                for m in metrics: row[m] = m_res[m][lbl] if m in m_res else np.nan
                local_results.append(row)
        return local_results
    except Exception as e:
        print(f"\n[SKIP SESSION] Error processing {uid}: {e}")
        return []

def plot_dataset_grid(df_subset, dataset_name, metrics):
    """Generate heatmaps for the given dataset results."""
    short = {'fslanat6071ants243': 'FSL6071', 'freesurfer741ants243': 'FS741', 'freesurfer8001ants243': 'FS8001', 'samseg8001ants243': 'SAMSEG8'}
    pipelines = [short[p] for p in PIPELINES]
    n_sessions = df_subset[['subject', 'session']].drop_duplicates().shape[0]

    for metric in metrics:
        vmin, vmax = get_heatmap_range(df_subset, metric)
        structs = list(STRUCTURE_LABELS.keys())
        ncols, nrows = 5, int(np.ceil(len(structs)/5))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)
        cmap = sns.color_palette("viridis", as_cmap=True) if metric == 'dice' else sns.color_palette("rocket_r", as_cmap=True)

        for i, s in enumerate(structs):
            ax = axes[i//ncols,i%ncols]
            d = df_subset[df_subset.structure==s]
            mat = pd.DataFrame(np.nan, index=pipelines, columns=pipelines)
            for (p1,p2), g in d.groupby(['pipeline1','pipeline2']):
                mat.at[short[p1],short[p2]] = mat.at[short[p2],short[p1]] = g[metric].mean()
            np.fill_diagonal(mat.values, 1 if metric == 'dice' else 0)
            sns.heatmap(mat, vmin=vmin, vmax=vmax, annot=True, fmt=".2f", square=True, cmap=cmap, cbar=False, ax=ax)
            ax.set_title(f"{s}", fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        for k in range(len(structs), ncols*nrows): fig.delaxes(axes[k//ncols, k%ncols])

        cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        metric_label = "Mean Dice" if metric == 'dice' else f"Mean {metric.upper()}"
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin,vmax)), cax=cax).set_label(metric_label, rotation=270, labelpad=15)
        
        fig.text(0.5, 0.01, f"n = {n_sessions} scans | {metric_label} = mean across subjects", ha='center', fontsize=12)
        fig.suptitle(f"Inter-pipeline {metric.upper()} per structure – {dataset_name}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])
        
        out_path = Path(OUTPUT_DIR)/f"structure_{metric}_grid_{dataset_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")

def main(n_subjects=10, parallel=False, max_threads=10, metrics=None, downsample_factor=1.0):
    if metrics is None: metrics = ['dice']
    ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")

    subj_maps = {pl: {"__".join(extract_ids(f, ROOT_DIR)): f for f in find_files(ROOT_DIR, pl, fname)} for pl, fname in PIPELINES.items()}
    all_datasets = sorted(set(uid.split("__")[0] for pl_map in subj_maps.values() for uid in pl_map.keys()))
    
    total_results = []
    for dataset in all_datasets:
        subjects_in_dataset = [set(uid for uid in pl_map.keys() if uid.startswith(dataset)) for pl_map in subj_maps.values()]
        common = sorted(set.intersection(*subjects_in_dataset))
        if n_subjects > 0: common = common[:n_subjects]
        if not common: continue
        
        pairs = [(a,b) for i,a in enumerate(PIPELINES) for b in list(PIPELINES)[i+1:]]
        ref_img = nib.load(subj_maps[list(PIPELINES.keys())[0]][common[0]]) 
        
        dataset_results = []
        compute_func = lambda uid: compute_for_subject(uid, ref_img, downsample_factor, metrics, pairs, dataset, STRUCTURE_LABELS, LABELS, PIPELINES, subj_maps)
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_threads) as exe:
                for r in tqdm(exe.map(compute_func, common), total=len(common), desc=f"Processing {dataset}"):
                    dataset_results.extend(r)
        else:
            for uid in tqdm(common, desc=f"Processing {dataset}"):
                dataset_results.extend(compute_func(uid))
        
        if dataset_results:
            plot_dataset_grid(pd.DataFrame(dataset_results), dataset, metrics)
            total_results.extend(dataset_results)

    if total_results:
        df_all = pd.DataFrame(total_results)
        df_all.to_csv(Path(OUTPUT_DIR)/"dice_overlap_tidy.csv", index=False)
        plot_dataset_grid(df_all, "ALL", metrics)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_subjects", type=int, default=0, help="0 for all subjects")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--max_threads", type=int, default=10)
    p.add_argument("--metrics", nargs="+", choices=['dice', 'hd95'], default=['dice'])
    p.add_argument("--downsample_factor", type=float, default=1.0, help="Resolution factor for HD95")
    a = p.parse_args()
    main(a.n_subjects, a.parallel, a.max_threads, a.metrics, a.downsample_factor)
