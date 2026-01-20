import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning

# ----------------------------
# 🔇 SILENCE WARNINGS
# ----------------------------
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# 🎯 REQUIREMENT 1: BASELINE MEAN PREDICTOR CLASS
# ----------------------------
class MeanPredictor:
    """A baseline model that always predicts the mean age of the training set."""
    def __init__(self):
        self.mean_age = None

    def fit(self, X_train, y_train):
        self.mean_age = y_train.mean()
        return self

    def predict(self, X_test):
        if self.mean_age is None:
            raise ValueError("Model must be fitted before prediction.")
        return np.full(len(X_test), self.mean_age)

# ----------------------------
# Configuration: Feature Sets & Models
# ----------------------------
EXPERIMENTS = {
    # --- Freesurfer 7.4.1 ---
    'fs741_vol_DKT': {
        'pipelines': ['freesurfer741ants243'], 
        'groups': ['volume', 'DKTatlas']
    },
    'fs741_vol_a2009s': {
        'pipelines': ['freesurfer741ants243'], 
        'groups': ['volume', 'a2009s']
    },
    'fs741_vol_DKT_a2009s': {
        'pipelines': ['freesurfer741ants243'], 
        'groups': ['volume', 'DKTatlas', 'a2009s']
    },

    # --- Freesurfer 8.0.0.1 ---
    'fs800_vol_DKT': {
        'pipelines': ['freesurfer8001ants243'], 
        'groups': ['volume', 'DKTatlas']
    },
    'fs800_vol_a2009s': {
        'pipelines': ['freesurfer8001ants243'], 
        'groups': ['volume', 'a2009s']
    },
    'fs800_vol_DKT_a2009s': {
        'pipelines': ['freesurfer8001ants243'], 
        'groups': ['volume', 'DKTatlas', 'a2009s']
    },

    # --- Aggregation (All Features) ---
    'aggregated_all': {
        'pipelines': ['freesurfer741ants243', 'freesurfer8001ants243'],
        'groups': ['volume', 'DKTatlas', 'a2009s']
    }
}

ENSEMBLE_COMPONENTS = [
    'fs741_vol_DKT', 
    'fs741_vol_a2009s', 
    'fs800_vol_DKT', 
    'fs800_vol_a2009s'
]

MODELS_TO_EVALUATE = ['baseline', 'elasticnet', 'kneighbors', 'histgradientboosting','extratrees', 'svm'] 

# --- TOGGLE FOR OUTLIER REMOVAL ---
RUN_OUTLIER_REMOVAL = False

# --- HYPERPARAMETER GRIDS ---
ELASTIC_NET_PARAMS = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

SVM_PARAMS = {
    'regressor__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__kernel': ['linear', 'rbf'],
    'regressor__gamma': ['scale', 'auto', 0.01, 0.1]
}

MLP_PARAMS = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'regressor__learning_rate_init': [0.001, 0.01],
    'regressor__early_stopping': [True]
}

RANDOM_FOREST_PARAMS = {
    'regressor__n_estimators': [100, 200],  
    'regressor__max_depth': [10, 20, None],  
    'regressor__min_samples_split': [2, 10], 
    'regressor__min_samples_leaf': [1, 5]    
}

EXTRA_TREES_PARAMS = {
    'regressor__n_estimators': [100, 200], 
    'regressor__max_depth': [10, 20, None], 
    'regressor__min_samples_split': [2, 5], 
    'regressor__min_samples_leaf': [1, 2]   
}

HIST_GBM_PARAMS = {
    'regressor__max_iter': [100, 200], 
    'regressor__learning_rate': [0.01, 0.1], 
    'regressor__max_leaf_nodes': [31, 63], 
    'regressor__max_depth': [5, None] 
}

K_NEIGHBORS_PARAMS = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2] 
}

# ----------------------------
# Data Preparation
# ----------------------------

def clean_age_value(age):
    if pd.isna(age): return np.nan
    if isinstance(age, str):
        age = age.replace('+', '').strip()
        try: return float(age)
        except ValueError: return np.nan
    return float(age)

def get_feature_cols(df):
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    return [col for col in df.columns if col not in metadata_cols]

def remove_outliers_mad(df_in, threshold_percent=0.05, mad_threshold=4.0): 
    df = df_in.copy()
    all_feature_cols = get_feature_cols(df)
    
    if len(all_feature_cols) == 0:
        return df
    
    if len(df) < 10:
        return df
        
    print(f"  Removing participants with > {threshold_percent * 100:.0f}% extreme outliers (MAD method)...")
    initial_count = len(df)
    
    medians = df[all_feature_cols].median()
    mad = (df[all_feature_cols] - medians).abs().median()
    
    mad = mad.replace(0, 1e-9)
    
    lower_bound = medians - mad_threshold * mad
    upper_bound = medians + mad_threshold * mad
    
    outlier_mask = (df[all_feature_cols] < lower_bound) | (df[all_feature_cols] > upper_bound)
    df['outlier_count'] = outlier_mask.sum(axis=1)
    
    max_allowed_outliers = int(len(all_feature_cols) * threshold_percent)
    df_clean = df[df['outlier_count'] <= max_allowed_outliers].drop(columns=['outlier_count']).copy()
    
    removed_count = initial_count - len(df_clean)
    print(f"  Removed {removed_count} samples ({(removed_count/initial_count)*100:.1f}%) based on MAD outlier threshold.")
        
    return df_clean

def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    if 'age' not in df.columns or 'sex' not in df.columns:
        raise KeyError("REQUIRED: 'age' and 'sex' columns not found.")
    
    print("Cleaning age data...")
    df['age'] = df['age'].apply(clean_age_value)
    
    # Create Subject ID
    df['subject_id'] = df['dataset'] + '_' + df['subject']
    
    # Filter Sex
    df['sex'] = df['sex'].astype(str).str.upper().str.strip()
    initial_count_sex = len(df)
    df_clean = df[df['sex'].isin(['M', 'F'])].copy()
    print(f"After removing non-M/F sex entries: {len(df_clean)} samples (removed {initial_count_sex - len(df_clean)})")

    # Filter Missing Age
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['age']).copy()
    print(f"After removing missing/invalid age: {len(df_clean)} samples (removed {initial_count - len(df_clean)})")
    
    # Identify All Possible Features
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    all_feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    
    # Filter Missing Features (Strict: Drop row if ANY feature is missing)
    initial_count_age_clean = len(df_clean)
    df_clean = df_clean.dropna(subset=all_feature_cols)
    print(f"After removing missing feature data: {len(df_clean)} samples (removed {initial_count_age_clean - len(df_clean)})")

    # Ensure numeric
    for col in all_feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Final check for NaNs introduced by numeric conversion
    final_count = len(df_clean)
    df_clean = df_clean.dropna(subset=all_feature_cols)
    print(f"After final numeric cleanup: {len(df_clean)} samples (removed {final_count - len(df_clean)})")

    # --- RESTORED SPARSE COLUMN DROP (Running AFTER cleaning) ---
    print("Scanning for sparse/empty columns (>50% 0.0 or NaN)...")
    cols_to_drop = []
    n_samples = len(df_clean)
    threshold = 0.5 * n_samples 
    
    for col in all_feature_cols:
        # Note: No NaNs exist here due to strict cleaning above, so this effectively checks for zeros
        n_bad = (df_clean[col] == 0).sum() + df_clean[col].isna().sum()
        if n_bad > threshold:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns that have >50% 0.0 or NaN values.")
        df_clean = df_clean.drop(columns=cols_to_drop)
        # Re-update the feature list
        all_feature_cols = [c for c in all_feature_cols if c not in cols_to_drop]
    # -------------------------------------------------------------

    if RUN_OUTLIER_REMOVAL:
        df_clean = remove_outliers_mad(df_clean)
    else:
        print("Outlier removal skipped.")
        
    print(f"Data Loaded: {len(df_clean)} samples available.")
    print(f"Sex distribution - M: {len(df_clean[df_clean['sex']=='M'])}, F: {len(df_clean[df_clean['sex']=='F'])}")
    
    return df_clean

def get_features_for_experiment(df, experiment_name):
    if experiment_name not in EXPERIMENTS:
        return []
    
    config = EXPERIMENTS[experiment_name]
    target_pipelines = config['pipelines']
    target_groups = config['groups']
    
    selected_features = []
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    for col in feature_cols:
        parts = col.split('__')
        if len(parts) < 3: continue
        
        pipeline_part = parts[0]
        group_part = parts[1]
        
        if pipeline_part in target_pipelines:
            if group_part in target_groups:
                selected_features.append(col)
                
    return selected_features

# ----------------------------
# Normalization Functions
# ----------------------------
class DatasetPipelineScaler:
    def __init__(self):
        self.scalers_ = {}  
        self.feature_means_ = {}
        self.feature_stds_ = {}
    
    def fit(self, X, features, datasets, pipeline_name):
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            if len(X_dataset) > 1:
                scaler = StandardScaler()
                scaler.fit(X_dataset)
                self.scalers_[(dataset, pipeline_name)] = scaler
                self.feature_means_[(dataset, pipeline_name)] = scaler.mean_
                self.feature_stds_[(dataset, pipeline_name)] = scaler.scale_
            else:
                self.scalers_[(dataset, pipeline_name)] = None
        
        return self
    
    def transform(self, X, features, datasets, pipeline_name):
        X_normalized = np.zeros_like(X)
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            if (dataset, pipeline_name) in self.scalers_ and self.scalers_[(dataset, pipeline_name)] is not None:
                X_normalized[dataset_mask] = self.scalers_[(dataset, pipeline_name)].transform(X_dataset)
            else:
                X_normalized[dataset_mask] = X_dataset
        
        return X_normalized

def normalize_features(df, features, experiment_name, datasets, fit_scaler=None):
    if len(features) == 0:
        return df, fit_scaler
    
    X = df[features].values
    
    if fit_scaler is None:
        scaler = DatasetPipelineScaler()
        scaler.fit(X, features, datasets, experiment_name)
        X_normalized = scaler.transform(X, features, datasets, experiment_name)
    else:
        X_normalized = fit_scaler.transform(X, features, datasets, experiment_name)
        scaler = fit_scaler
    
    df_normalized = df.copy()
    df_normalized[features] = X_normalized
    return df_normalized, scaler

# ----------------------------
# Modeling Functions
# ----------------------------

def create_model_pipeline(model_type):
    # OPTIMIZATION: Set n_jobs=1 for internal model parallelism to allow outer-loop parallelization
    if model_type == 'baseline':
        return MeanPredictor(), {}, None
    elif model_type == 'elasticnet':
        # Increased max_iter to help convergence, warning silenced globally
        return Pipeline([('regressor', ElasticNet(random_state=42, max_iter=100000))]), ELASTIC_NET_PARAMS, RandomizedSearchCV
    elif model_type == 'randomforest':
        return Pipeline([('regressor', RandomForestRegressor(random_state=42, n_jobs=1))]), RANDOM_FOREST_PARAMS, RandomizedSearchCV
    elif model_type == 'extratrees':
        return Pipeline([('regressor', ExtraTreesRegressor(random_state=42, n_jobs=1))]), EXTRA_TREES_PARAMS, RandomizedSearchCV
    elif model_type == 'histgradientboosting':
        return Pipeline([('regressor', HistGradientBoostingRegressor(random_state=42, verbose=0))]), HIST_GBM_PARAMS, RandomizedSearchCV
    elif model_type == 'svm':
        return Pipeline([('regressor', SVR())]), SVM_PARAMS, RandomizedSearchCV
    elif model_type == 'mlp':
        return Pipeline([('regressor', MLPRegressor(random_state=42, max_iter=2000, early_stopping=True))]), MLP_PARAMS, RandomizedSearchCV
    elif model_type == 'kneighbors':
        return Pipeline([('regressor', KNeighborsRegressor(n_jobs=1))]), K_NEIGHBORS_PARAMS, RandomizedSearchCV
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def nested_cv_evaluation(df, model_type, experiment_name):
    """
    Standard evaluation for a single experiment (feature set).
    """
    # Use simple print format for logs since we are running in parallel
    print(f"  [START] {model_type} on {experiment_name}", flush=True)
    
    features = get_features_for_experiment(df, experiment_name)
    if len(features) == 0 and model_type != 'baseline':
        print(f"    WARNING: No features found for {experiment_name}")
        return []
    
    # --- IMPORTANT: Do NOT dropna() here. Use df as-is to ensure consistent N across models ---
    # The df passed in has already been strictly cleaned in load_and_preprocess_data
    cols_needed = features + ['dataset', 'sex', 'age', 'subject_id']
    df_exp = df[cols_needed].copy()
    
    if len(df_exp) < 10:
        return []

    X_full = df_exp[features + ['dataset', 'sex']] if model_type != 'baseline' else df_exp[['dataset', 'sex']]
    y_full = df_exp['age']
    groups_full = df_exp['subject_id']
    
    n_splits = min(7, len(df_exp))
    if n_splits < 2: return []
    
    outer_cv = GroupKFold(n_splits=n_splits)
    
    sex_results = {'M': {'scores': [], 'train_size': 0}, 'F': {'scores': [], 'train_size': 0}}
    model_pipeline, param_grid, search_class = create_model_pipeline(model_type)

    search_kwargs = {}
    if search_class == RandomizedSearchCV:
        total_param_combos = np.prod([len(v) for v in param_grid.values()])
        n_iter = min(20, total_param_combos)
        # CRITICAL: n_jobs=1 to prevent oversubscription in parallel execution
        search_kwargs = {'n_iter': n_iter, 'random_state': 42, 'n_jobs': 1}

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups_full)):
        # --- VERBOSE LOGGING START (Matched to 03_ml_models.py) ---
        print(f"    Fold {fold + 1} ({experiment_name}/{model_type}):", end=" ", flush=True)
        
        X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_full, y_test_full = y_full.iloc[train_idx], y_full.iloc[test_idx]
        groups_train = groups_full.iloc[train_idx]
        
        for sex in ['M', 'F']:
            train_mask = X_train_full['sex'] == sex
            X_train_sex = X_train_full[train_mask].drop(columns=['sex'])
            y_train_sex = y_train_full[train_mask]
            groups_train_sex = groups_train[train_mask]
            datasets_train_sex = X_train_sex['dataset']
            
            test_mask = X_test_full['sex'] == sex
            X_test_sex = X_test_full[test_mask].drop(columns=['sex'])
            y_test_sex = y_test_full[test_mask]
            datasets_test_sex = X_test_sex['dataset']

            if len(y_train_sex) < 2 or len(y_test_sex) == 0:
                print(f"({sex} - Too few samples)", end=" ", flush=True)
                continue
                
            try:
                if model_type != 'baseline':
                    X_train_feats = X_train_sex.drop(columns=['dataset'])
                    X_test_feats = X_test_sex.drop(columns=['dataset'])
                    
                    train_normalized, scaler = normalize_features(
                        X_train_feats, features, experiment_name, datasets_train_sex
                    )
                    test_normalized, _ = normalize_features(
                        X_test_feats, features, experiment_name, datasets_test_sex, fit_scaler=scaler
                    )
                    
                    inner_splits = min(3, len(y_train_sex))
                    inner_cv = GroupKFold(n_splits=inner_splits)
                    
                    search = search_class(model_pipeline, param_grid, cv=inner_cv, 
                                        scoring='neg_mean_absolute_error', error_score='raise', **search_kwargs)
                    search.fit(train_normalized[features].values, y_train_sex, groups=groups_train_sex.values)
                    
                    y_pred = search.best_estimator_.predict(test_normalized[features].values)
                    mae = mean_absolute_error(y_test_sex, y_pred)
                    
                    # Log best params
                    best_params_str = ", ".join([f"{k.split('__')[-1]}: {v}" for k, v in search.best_params_.items()])
                    print(f"({sex} MAE={mae:.3f}, Params: {best_params_str[:30]}..., N={len(y_train_sex)})", end=" ", flush=True)
                    
                else:
                    baseline = MeanPredictor().fit(None, y_train_sex)
                    y_pred = baseline.predict(X_test_sex)
                    mae = mean_absolute_error(y_test_sex, y_pred)
                    print(f"({sex} MAE={mae:.3f}, Mean={baseline.mean_age:.1f}, N={len(y_train_sex)})", end=" ", flush=True)
                
                sex_results[sex]['scores'].append(mae)
                sex_results[sex]['train_size'] = len(y_train_sex) 
                
            except Exception as e:
                print(f"({sex} FAILED: {str(e)[:20]})", end=" ", flush=True)
                continue
        print("", flush=True) # Newline after fold

    final_results = []
    for sex in ['M', 'F']:
        scores = sex_results[sex]['scores']
        if scores:
            final_results.append({
                'pipeline': experiment_name,
                'model': model_type,
                'sex': sex,
                'mean_mae': np.mean(scores),
                'std_mae': np.std(scores),
                'scores': scores,
                'successful_folds': len(scores),
                # Corrected Key Name
                'n_train_samples_per_fold': sex_results[sex]['train_size']
            })
            
    print(f"  [DONE] {model_type} on {experiment_name} -> M: {len(sex_results['M']['scores'])} folds, F: {len(sex_results['F']['scores'])} folds", flush=True)
    return final_results

# --- ENSEMBLE EVALUATION (PARALLELIZED) ---
def evaluate_single_ensemble_model(df, model_type):
    """
    Runs the ensemble logic for a single model type. 
    """
    if model_type == 'baseline': return []
    
    print(f"  [START ENSEMBLE] {model_type}...", flush=True)
    sex_scores = {'M': [], 'F': []}
    
    groups = df['subject_id']
    n_splits = min(7, len(df))
    outer_cv = GroupKFold(n_splits=n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, df['age'], groups)):
        print(f"    Ensemble Fold {fold + 1} ({model_type}):", end=" ", flush=True)
        
        X_train_full, X_test_full = df.iloc[train_idx], df.iloc[test_idx]
        y_train_full, y_test_full = df['age'].iloc[train_idx], df['age'].iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        for sex in ['M', 'F']:
            train_mask = X_train_full['sex'] == sex
            test_mask = X_test_full['sex'] == sex
            
            if not any(test_mask): continue
            
            X_train_sex = X_train_full[train_mask]
            y_train_sex = y_train_full[train_mask]
            groups_train_sex = groups_train[train_mask]
            datasets_train_sex = X_train_sex['dataset']
            
            X_test_sex = X_test_full[test_mask]
            y_test_sex = y_test_full[test_mask]
            datasets_test_sex = X_test_sex['dataset']

            ensemble_preds = np.zeros(len(y_test_sex))
            valid_counts = np.zeros(len(y_test_sex)) 
            
            for exp_name in ENSEMBLE_COMPONENTS:
                features = get_features_for_experiment(df, exp_name)
                if len(features) == 0: continue
                
                # Assume complete cases because of strict loading
                X_comp_train = X_train_sex
                y_comp_train = y_train_sex
                grps_comp_train = groups_train_sex
                dsets_comp_train = datasets_train_sex
                
                X_comp_test = X_test_sex
                dsets_comp_test = datasets_test_sex
                
                try:
                    train_norm, scaler = normalize_features(
                        X_comp_train, features, exp_name, dsets_comp_train
                    )
                    test_norm, _ = normalize_features(
                        X_comp_test, features, exp_name, dsets_comp_test, fit_scaler=scaler
                    )
                    
                    pipeline, param_grid, search_class = create_model_pipeline(model_type)
                    search_kwargs = {}
                    if search_class == RandomizedSearchCV:
                        n_iter = min(10, np.prod([len(v) for v in param_grid.values()]))
                        # Force n_jobs=1
                        search_kwargs = {'n_iter': n_iter, 'random_state': 42, 'n_jobs': 1}
                    
                    inner_splits = min(3, len(y_comp_train))
                    inner_cv = GroupKFold(n_splits=inner_splits)
                    
                    search = search_class(pipeline, param_grid, cv=inner_cv, scoring='neg_mean_absolute_error', error_score='raise', **search_kwargs)
                    search.fit(train_norm[features].values, y_comp_train, groups=grps_comp_train.values)
                    
                    pred = search.best_estimator_.predict(test_norm[features].values)
                    
                    # All rows valid because of strict cleaning
                    ensemble_preds += pred
                    valid_counts += 1
                    
                except Exception as e:
                    pass
            
            final_mask = valid_counts > 0
            if any(final_mask):
                final_preds = ensemble_preds[final_mask] / valid_counts[final_mask]
                final_truth = y_test_sex.values[final_mask]
                mae = mean_absolute_error(final_truth, final_preds)
                sex_scores[sex].append(mae)
                print(f"({sex} MAE={mae:.3f})", end=" ", flush=True)
        print("", flush=True)

    aggregated_results = []
    for sex in ['M', 'F']:
        if sex_scores[sex]:
            aggregated_results.append({
                'pipeline': 'ensemble_4_components',
                'model': model_type,
                'sex': sex,
                'mean_mae': np.mean(sex_scores[sex]),
                'std_mae': np.std(sex_scores[sex]),
                'scores': sex_scores[sex],
                'successful_folds': len(sex_scores[sex]),
                'n_train_samples_per_fold': 'Variable'
            })
            
    print(f"  [DONE ENSEMBLE] {model_type}", flush=True)
    return aggregated_results

# ----------------------------
# Main Execution 
# ----------------------------

def run_experiment_wrapper(exp_name, model_type, df):
    """Wrapper to be called by joblib"""
    try:
        results = nested_cv_evaluation(df, model_type, exp_name)
        # Process results to flatten them before returning
        processed_results = []
        for res in results:
            scores = res.pop('scores', [])
            for i, score in enumerate(scores):
                res[f'mae_fold_{i+1}'] = score
            processed_results.append(res)
        return processed_results
    except Exception as e:
        print(f"CRITICAL ERROR in {exp_name} {model_type}: {e}")
        return []

def run_ensemble_wrapper(model_type, df):
    """Wrapper for ensemble tasks"""
    try:
        results = evaluate_single_ensemble_model(df, model_type)
        processed_results = []
        for res in results:
            scores = res.pop('scores', [])
            for i, score in enumerate(scores):
                res[f'mae_fold_{i+1}'] = score
            processed_results.append(res)
        return processed_results
    except Exception as e:
        print(f"CRITICAL ERROR in Ensemble {model_type}: {e}")
        return []

def main():
    print("=" * 60)
    print("BRAIN AGE - PARALLEL EXECUTION (SLURM OPTIMIZED)")
    print("=" * 60)
    
    # 1. Detect Cores
    n_jobs_available = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))
    if n_jobs_available == -1:
        # If not in SLURM or env var not set, default to standard cpu count - 1
        n_jobs_available = max(1, os.cpu_count() - 1)
        
    print(f"Parallelizing with n_jobs={n_jobs_available}")

    # Use strict loading to match 03_ml_models.py
    df = load_and_preprocess_data('ml_dataset_with_age_sex_BOTH.csv')
    if len(df) == 0: return

    all_results = []
    
    # 2. Build Task List (Experiment Phase)
    tasks = []
    for exp_name in EXPERIMENTS.keys():
        for model_type in MODELS_TO_EVALUATE:
            tasks.append((exp_name, model_type))
            
    print(f"Queuing {len(tasks)} individual experiment tasks...")
    
    # 3. Execute Individual Experiments
    results_lists = Parallel(n_jobs=n_jobs_available)(
        delayed(run_experiment_wrapper)(exp, model, df) for exp, model in tasks
    )
    
    for res_list in results_lists:
        all_results.extend(res_list)

    # 4. Build Task List (Ensemble Phase)
    print("\nQueuing Ensemble tasks...")
    ensemble_tasks = [m for m in MODELS_TO_EVALUATE if m != 'baseline']
    
    ensemble_results_lists = Parallel(n_jobs=n_jobs_available)(
        delayed(run_ensemble_wrapper)(model, df) for model in ensemble_tasks
    )

    for res_list in ensemble_results_lists:
        all_results.extend(res_list)

    # 5. Report
    results_df = pd.DataFrame(all_results)
    
    # Order columns nicely
    if len(results_df) > 0:
        cols = results_df.columns.tolist()
        meta = ['pipeline', 'model', 'sex', 'mean_mae', 'std_mae', 'successful_folds', 'n_train_samples_per_fold']
        fold_cols = sorted([c for c in cols if 'mae_fold' in c])
        other_cols = [c for c in cols if c not in meta and c not in fold_cols]
        final_order = [c for c in meta if c in cols] + fold_cols + other_cols
        results_df = results_df[final_order]

        results_df.to_csv('brain_age_results_cortical_experiments.csv', index=False)
        print(f"\nSaved results to brain_age_results_cortical_experiments.csv")
    else:
        print("No results generated.")
    
    return results_df

if __name__ == "__main__":
    main()
