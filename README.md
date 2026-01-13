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

* **01_ml_features_MNI.py**:
* **01_ml_features_ASEG.py and 01_ml_features_NATIVE.py**: Old scripts I don't really use now. ASEG ingests from the volume outputs (df_wide.csv) of NeuroCI, and NATIVE is siilar to MNI but instead ingests the native-space scans from the Nipoppy datasets.
* **02_ml_age_sex.py**: This script is meant to be run after one of the 01 scripts above.
* **03_ml_models.py**: This script is meant to be run after 02_ml_age_sex.py.
* **04_ml_plot.py**: This script is meant to be run after 03_ml_models.py.
* **dice_hd95.py**:
* **disagreement.py**:
