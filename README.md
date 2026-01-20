### This repository contains several scripts for analyzing inter-pipeline discrepancies and leveraging this variability to predict brain age using scikit-learn models.
* These scripts were too resource-intensive to run on NeuroCI. For demographic analyses and volumetric computations (e.g., volumetric correlations, volume distributions, SVD), please refer to the NeuroCI repository: https://github.com/neurodatascience/NeuroCI/ . In the NeuroCI repository, you can also reproducibly see how all datasets used in the analyses were processed to obtain segmentations.

* The scripts are meant to run in a file directory that looks as follows:
```
. (pwd)
├── 01_ml_features_ASEG.py
├── 01_ml_features_MNI.py
├── 02_ml_age_sex.py
├── 03_ml_models.py
├── 04_ml_plot.py
├── dice_hd95.py
├── disagreement.py
├── YourDataset1
│   └── dataset1_nipoppy
├── YourDataset2
│   └── dataset2_nipoppy
├── etc...
```
Where the datasetX_nipoppy directory is a Nipoppy-formatted dataset (you only need the derivatives and tabular folders for the purpose of running these scripts).

* **01_ml_features_MNI.py** - Feature Extraction:
    * What it does: Extracts morphological and geometric features from MNI-space brain segmentations. It utilizes SimpleITK for high-fidelity volume and surface area calculations and Scikit-Learn for PCA-based shape analysis.
    * Inputs: NIfTI segmentation files (.nii.gz) across multiple neuroimaging pipelines (FreeSurfer, FSL, SAMSEG).
    * Outputs: morphological_features_mni.csv containing a high-dimensional feature matrix (Volumes, Surface Area, Sphericity, Compactness, and PCA Eigenvalues) indexed by subject and session.
    * Implements multi-core parallel processing with robust error handling to skip corrupt files without crashing the execution queue (eg: python3 01_ml_features_MNI.py --parallel --max_threads 4 --features volume pca-eigenvalue-1 pca-eigenvalue-2)
 
* **01_ml_features_CORTICAL.py** – FreeSurfer Stats Parser & Aggregator
    * What it does: Recursively traverses experiment directories to identify and parse native FreeSurfer statistics files. It extracts Subcortical Volumes (ASEG) and Cortical metrics (Surface Area and Average Thickness) from two atlases (DKT and a2009s).
    * Inputs: A directory structure containing derivatives/folders with FreeSurfer outputs (supports multiple pipelines and versions, e.g., freesurfer741, freesurfer8001).
    * Outputs: morphological_features_CORTICAL.csv (Wide format) and df_tidy_cortical.csv (Long format).
    * Custom text parsing logic for .stats files (ignoring header comments and parsing columnar data). Automatically detects datasets based on directory hierarchy and pivots data into a machine-learning-ready feature matrix where columns represent specific structures (e.g., rh_SuperiorFrontal_ThickAvg).

* **01_ml_features_ASEG.py and 01_ml_features_NATIVE.py**: Old scripts I don't really use now. ASEG ingests from the volume outputs (df_wide.csv) of NeuroCI, and NATIVE is similar to MNI but instead ingests the native-space scans from the Nipoppy datasets.

* **02_ml_age_sex.py** - Demographic Integration: This script is meant to be run after any of the 01 scripts.
    *  What it does: Performs complex data wrangling to align neuroimaging features with phenotypic data. It features custom regex and mapping logic to handle inconsistent session naming conventions (e.g., matching NAPFU12 to ses-FU12) across diverse datasets like PREVENT-AD and NKI.
    *  Inputs: The features CSV and dataset-specific tabular metadata (BIDS .tsv files).
    *  Outputs: ml_dataset_with_age_sex.csv, a unified "master" dataset ready for machine learning.
    *  Uses strict mapping logic to maintain data integrity across longitudinal sessions, ensuring age and sex are accurately assigned per scan (eg: python3 02_ml_age_sex.py)
 
* **022_merge_BOTH.py** – Feature Set Harmonization. Meant to be run after 02_ml_age_sex.py and before 03_ml_models_HPC.py.
    * What it does: Merges the specialized Cortical feature set with the existing Volumetric feature set to create a unified dataset for analysis. It handles namespace collisions by removing redundant volume columns from the cortical input and standardizing column naming conventions.
    * Inputs: ml_dataset_with_age_sex_CORTICAL.csv and ml_dataset_with_age_sex.csv (subcortical).
    * Outputs: ml_dataset_with_age_sex_BOTH.csv (BOTH as it has the cortical AND subcortical data).
    * Performs an inner join on metadata keys (dataset, subject, session, age, sex). Includes a renaming utility to normalize feature names (converting __volume suffixes to __mm where appropriate) to ensure downstream compatibility with the ML pipeline.

* **03_ml_models.py** - Machine Learning Modeling and Anomaly Detection: This script is meant to be run after 02_ml_age_sex.py. It works on the subcortical data across all pipelines.
    * What it does: Executes a rigorous Nested Group K-Fold Cross-Validation framework to predict age while preventing data leakage from longitudinal subjects.
    * Inputs: The unified master dataset.
    * Outputs: brain_age_results_sex_split_and_correct_ensembles.csv containing performance metrics (MAE, STD) for individual pipelines and ensembles.
    * Anomaly Detection & Outlier Removal: Implements a robust Median Absolute Deviation (MAD) filtering system that identifies and removes participants if more than X% of their brain features are extreme outliers. Parameters for this inside code.
    * Evaluates multiple regressors (ElasticNet, SVR, HistGradientBoosting, etc.) against a MeanPredictor baseline. It includes a custom DatasetPipelineScaler to handle site-specific variance and implements sex-stratified training to capture biological dimorphism (eg: python3 03_ml_models.py)
 
* **03_ml_models_HPC.py** – Similar to 03_ml_models.py but meant for the cortical or cortical+subcortical experiments, and optimized for parallel execution on an HPC (as the many cortical features take a long time to compute on).
    * What it does: A robust machine learning training and evaluation script optimized for SLURM-based High-Performance Computing (HPC) environments. It benchmarks multiple regressors (ElasticNet, SVR, RandomForest, HistGradientBoosting, MLP, ExtraTrees) and ensemble methods on brain age prediction tasks.
    * Inputs: ml_dataset_with_age_sex_BOTH.csv.
    * Outputs: brain_age_results_cortical_experiments.csv containing Mean Absolute Error (MAE) scores for every fold, broken down by Sex, Model, and Pipeline.
    * Dynamic resource allocation using SLURM_CPUS_PER_TASK and joblib to parallelize experiments across available CPU cores.

* **04_ml_plot.py** - Results Visualization: This script is meant to be run after 03_ml_models.py.
    * What it does: Generates publication-ready visualizations of the model performance.
    * Inputs: The results CSV from the modeling script.
    * Outputs: A side-by-side comparison plot (brain_age_side_by_side_sexes.png) showing Mean Absolute Error (MAE) across all models, pipelines, and sexes.
    * Automatically parses metadata to display sample sizes ($n$) and scan counts ($k$) in plot headers.

* **05_permutation_testing.py** – Statistical Significance Testing for ML models.
    * What it does: Analyzes the results from the ML pipeline to determine the best-performing learner and sex group. It then performs a paired permutation test (10,000 permutations) to assess if the difference in performance between the top two processing pipelines is statistically significant.
    * Inputs: The results CSV generated by the 03 ML pipeline(s) (e.g., brain_age_results_sex_split_and_correct_ensembles.csv).
    * Outputs: analysis_summary.csv containing the observed mean difference, p-value, and significance boolean.
    * Vectorized numpy operations for efficient null distribution generation (random sign flipping of paired differences). Calculates a two-sided p-value to rigorously compare model performance.

These scripts are for measuring disagreement, not for the ML models:

* **dice_hd95.py** - Segmentation Overlap Metrics:
    * What it does: Computes standard geometric validation metrics—Dice Similarity Coefficient (DSC) and 95th Percentile Hausdorff Distance (HD95)—to assess the reliability of segmentations across pipelines.
    * Inputs: Native-space segmentation files across diverse datasets.
    * Outputs: Tidy-format CSVs (dice_overlap_tidy.csv) and structured grid heatmaps for every subcortical structure, broken down by dataset.
    * Uses SimpleITK for high-precision distance map calculations (eg: python3 dice_hd95.py --parallel --max_threads 4 --metrics dice)

* **disagreement.py** - Pipeline Disagreement Analysis:
    * What it does: Quantifies and visualizes spatial "disagreement" between different neuroimaging pipelines by calculating voxel-wise XOR (Exclusive OR) maps of subcortical segmentations.
    * Inputs: MNI-space segmentation files from multiple pipelines and a standard MNI152 template.
    * Outputs: Global disagreement heatmaps (pipeline_disagreement_heatmap.png) overlaid on the MNI template and a statistical summary (pipeline_disagreement_statistics.csv).
    * python3 disagreement.py --threads 4
