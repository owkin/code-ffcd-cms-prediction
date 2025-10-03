<div align="center">

<h1>Deep Learning on Histologic Slides Accurately Predicts Consensus Molecular Subtypes and Spatial Heterogeneity in Colon Cancer</h1>

</div>

This repository allows the reproduction of CMS-prediction models trained in the paper `Deep Learning on Histologic Slides Accurately Predicts Consensus Molecular Subtypes and Spatial Heterogeneity in Colon Cancer` by `Le Douget et al.` (DOI: https://doi.org/10.1016/j.modpat.2025.100877).  


# Installation

- Install with `pip` in editable mode

```
git clone https://github.com/owkin/code-ffcd-cms-prediction
cd cms_prediction
pip install -e .
````

- Install with `uv` (also creating a virtual environment)
```
git clone https://github.com/owkin/code-ffcd-cms-prediction
cd cms_prediction
uv sync
```

# Prepare datasets

## Extract features externally

`cms_prediction` expects tile-level whole-slide images features as input.  
Hence, WSI must be featurized prior to training.    

Many feature extractors are publicly available including:
- Phikon https://huggingface.co/owkin/phikon
- H0-Mini https://huggingface.co/bioptimus/H0-mini
- UNI2 https://huggingface.co/MahmoodLab/UNI2-h
- Virchow2 https://huggingface.co/paige-ai/Virchow2


## CMS extraction 

CMS must be extracted from RNAseq using The CMSclassifier package (https://github.com/Sage-Bionetworks/CMSclassifier).  
For each tumor sample, the output should be a 4-dimensional CMS vector of float values, summing to 1.

## Create a Dataset structure

A Dataset is constituted of labels and features. It contains:

- A `labels.csv` file with the following columns : 
    - **patient:** a Patient ID. Used to ensure all WSI of a given patient are within the same split during cross-validation. 
    - **slide:** a WSI ID. To be matched with the feature files.
    - **RF.CMS1, RF.CMS2, RF.CMS3, RF.CMS4:** Continous CMS scores per tumor (must sum to 1)

- A features folder, containing feature files named `<slide_id>.npy`. These files have `T` rows and `N+3` columns with N the output dimension of the feature extractor and T the number of tiles in the given slide. The first 3 columns contain the openslide level at which feature extraction was performed and the 2nd and 3rd columns contains the tile coordinates.



## Toy dataset 

To play with the code and get a grasp at the expected dataset structure, a utillity is available in the repository to 
create a dummy test dataset with random values.

- create the dummy dataset within the repository in `cms_prediction/.data` folder
```
python cms_prediction/data/create_dummy_dataset.py
```

- create the dummy dataset within the folder of your choice
```
python cms_prediction/data/create_dummy_dataset.py --dest_folder </PATH/TO/YOUR/FOLDER>
```

# Train a Model

The training CLI entrypoint is located at `cms_prediction/cli/cms_classification_crossval.py`.  
You can call the `help` module to see available options and defaults:
```
python cms_prediction/cli/cms_classification_crossval.py --help
```

An example bash script is provided at `scripts/train_cms_classifier.sh`.

# Evaluate your model on a separate dataset

The model evaluation CLI entrypoint is located at `cms_prediction/cli/cms_classification_external_val.py`.  
You can call the `help` module to see available options and defaults:
```
python cms_prediction/cli/cms_classification_external_val.py --help
```

An example bash script is provided at `scripts/evaluate_cms_classifier.sh`.
