# **Mule Account Detection:** Bank Account Fraud

## Description

**Objective:** Build a model to detect mule accounts in banking for real-world deployment.

**Approach:**
- Utilized the “Bank Account Fraud Dataset Suite (NeurIPS 2022)”: a highly imbalanced dataset containing 6 million accounts and 30 realistic features.
- Performed techniques such as exploratory data analysis, one-hot encoding and standard scaling.
- Implemented multiple machine learning models including Logistic Regression, Random Forest and XGBoost.
- Validated the model using confusion matrices, precision, recall, F1 scores and AUC-ROC.

**Impact:**
- Achieved 99% accuracy in mule account detection.
- The model successfully predicts the risk percentage for any account.

## Repository Structure

- This repository contains all the code required to replicate the project.
- The code is thoroughly commented to ensure clarity and ease of understanding.


## Dataset Preparation

**Note:** This is the most important step. Correctly preparing the datasets will ensure the scripts run successfully. So, follow all the steps carefully.

**Step 1:**

The public dataset suite is available for download through [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

> The paper describing this dataset suite ''Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation'' has been accepted at **NeurIPS 2022**. 
[NeurIPS Paper](https://arxiv.org/pdf/2211.13358.pdf)

This will give you 6 .csv files- Base, Variant I, Variant II, Variant III, Variant IV, Variant V

You need to add a new column "Account_Number" in all these .csv files with values from 1 to 10,00,000.

Put them in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder without changing their names.

Use these python scripts to do it:-
```python
import glob
import pandas as pd

extension = "csv"  # or "parquet", depending on the downloaded file
data_paths = glob.glob(f"</path/to/datasets/>*.{extension}")

def read_dataset(path, ext=extension):
    if ext == "csv":
        return pd.read_csv(path, index_col=0)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid extension: '{ext}'.")

def get_variant(path):
        return path.split("/")[-1].split(".")[0]

dataframes = {
    get_variant(path): read_dataset(path) for path in data_paths
}
```

**Step 2**

Divide the Base.csv file into two .csv files:

Base_half1.csv: "Account_Number" column from 1 to 5,00,000.

Base_half2.csv: "Account_Number" column from 5,00,001 to 10,00,000.

Save these .csv files in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use these python scripts to do it:-
```python
import glob
import pandas as pd

extension = "csv"  # or "parquet", depending on the downloaded file
data_paths = glob.glob(f"</path/to/datasets/>*.{extension}")

def read_dataset(path, ext=extension):
    if ext == "csv":
        return pd.read_csv(path, index_col=0)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid extension: '{ext}'.")

def get_variant(path):
        return path.split("/")[-1].split(".")[0]

dataframes = {
    get_variant(path): read_dataset(path) for path in data_paths
}
```

**Step 3**

Divide the Base.csv file into two .csv files:

base_99_%.csv: "Account_Number" column from 1 to 9,90,000.

base_1_%.csv: "Account_Number" column from 9,90,001 to 10,00,000.

Save these .csv files in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use these python scripts to do it:-
```python
import glob
import pandas as pd

extension = "csv"  # or "parquet", depending on the downloaded file
data_paths = glob.glob(f"</path/to/datasets/>*.{extension}")

def read_dataset(path, ext=extension):
    if ext == "csv":
        return pd.read_csv(path, index_col=0)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid extension: '{ext}'.")

def get_variant(path):
        return path.split("/")[-1].split(".")[0]

dataframes = {
    get_variant(path): read_dataset(path) for path in data_paths
}
```

**Step 4**

Combine all the variants with the name "All_Variants_Combined.csv"

Use these python scripts to do it:-
```python
import glob
import pandas as pd

extension = "csv"  # or "parquet", depending on the downloaded file
data_paths = glob.glob(f"</path/to/datasets/>*.{extension}")

def read_dataset(path, ext=extension):
    if ext == "csv":
        return pd.read_csv(path, index_col=0)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid extension: '{ext}'.")

def get_variant(path):
        return path.split("/")[-1].split(".")[0]

dataframes = {
    get_variant(path): read_dataset(path) for path in data_paths
}
```

"Account_Number" column needs to be changed with values from 1 to 50,00,000.

Save this .csv file in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use this python script to do it:-
```python
import glob
import pandas as pd

extension = "csv"  # or "parquet", depending on the downloaded file
data_paths = glob.glob(f"</path/to/datasets/>*.{extension}")

def read_dataset(path, ext=extension):
    if ext == "csv":
        return pd.read_csv(path, index_col=0)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid extension: '{ext}'.")

def get_variant(path):
        return path.split("/")[-1].split(".")[0]

dataframes = {
    get_variant(path): read_dataset(path) for path in data_paths
}
```

**Step 5**

**Step 6**

**Step 7**

Create the following blank .txt files:

- actual_fraudulent_accounts.txt
- fraudulent_accounts.txt
- lr_fraudulent_accounts.txt
- xgb_fraudulent_accounts.txt
- rf_fraudulent_accounts.txt

## Environment setup

```bash
conda create -n voicecraft python=3.9.16
conda activate voicecraft

pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
apt-get install ffmpeg # if you don't already have ffmpeg installed
apt-get install espeak-ng # backend for the phonemizer installed below
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
# pip install huggingface_hub
# conda install pocl # above gives an warning for installing pocl, not sure if really need this

# to run ipynb
conda install -n voicecraft ipykernel --no-deps --force-reinstall
```

## Usage


The [first notebook](notebooks/generate_dataset_variants.ipynb) regards the process of sampling from a large dataset to obtain the different variants that constitute the suite of datasets.

The [second notebook](notebooks/empirical_results.ipynb) presents the train of 100 LightGBM models (hyperparameters selected through random search) on the suite of datasets, as well as plots of the results. 

To replicate the environment used in the experiments, install the  `requirements.txt` file via pip in a Python 3.7 environment. 

Additionally, you can find the official documentation of the suite of datasets in the `documents` folder. In this folder you will find the [Paper](documents/BAF_paper.pdf) and [Datasheet](documents/datasheet.pdf)

The paper contains more detailed information on motivation, generation of the base dataset and [variants](notebooks/generate_dataset_variants.ipynb) and an experiment performed in the [empirical results notebook](notebooks/empirical_results.ipynb).
The datasheet contains a summarized description of the dataset. 
