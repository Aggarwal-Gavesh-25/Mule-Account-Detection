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
import pandas as pd
import os

# Define file names
files = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']

# Path to your archive folder where the .csv files are located
archive_path = '/path/to/your/archive/folder/'

# Create account numbers
account_numbers = list(range(1, 1000001))

# Process each file
for file in files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(archive_path, file))
    
    # Add the "Account_Number" column
    df['Account_Number'] = account_numbers
    
    # Overwrite the old file with the modified data
    df.to_csv(os.path.join(archive_path, file), index=False)

print("All files have been updated and replaced in the archive folder.")
```

Instructions:

- Replace /path/to/your/archive/folder/ with the directory path where your .csv files are located in the “archive” folder.


**Step 2**

Divide the Base.csv file into two .csv files:

Base_half1.csv: "Account_Number" column from 1 to 5,00,000.

Base_half2.csv: "Account_Number" column from 5,00,001 to 10,00,000.

Save these .csv files in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use these python scripts to do it:-
```python
import pandas as pd
import os

# Define file names
base_file = 'Base.csv'
half1_file = 'Base_half1.csv'
half2_file = 'Base_half2.csv'

# Path to your archive folder where the .csv file is located
archive_path = '/path/to/your/archive/folder/'

# Read the original Base.csv file
df = pd.read_csv(os.path.join(archive_path, base_file))

# Split the data into two halves based on "Account_Number"
df_half1 = df[df['Account_Number'] <= 500000]
df_half2 = df[df['Account_Number'] > 500000]

# Save the two halves into separate .csv files
df_half1.to_csv(os.path.join(archive_path, half1_file), index=False)
df_half2.to_csv(os.path.join(archive_path, half2_file), index=False)

print("Base.csv has been divided into Base_half1.csv and Base_half2.csv and saved in the archive folder.")
```

Instructions:

- Replace /path/to/your/archive/folder/ with the path to your archive folder.

**Step 3**

Divide the Base.csv file into two .csv files:

base_99_%.csv: "Account_Number" column from 1 to 9,90,000.

base_1_%.csv: "Account_Number" column from 9,90,001 to 10,00,000.

Save these .csv files in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use these python scripts to do it:-
```python
import pandas as pd
import os

# Define file names
base_file = 'Base.csv'
base_99_file = 'base_99_%.csv'
base_1_file = 'base_1_%.csv'

# Path to your archive folder where the .csv file is located
archive_path = '/path/to/your/archive/folder/'

# Read the original Base.csv file
df = pd.read_csv(os.path.join(archive_path, base_file))

# Split the data into two parts based on "Account_Number"
df_base_99 = df[df['Account_Number'] <= 990000]
df_base_1 = df[df['Account_Number'] > 990000]

# Save the two parts into separate .csv files
df_base_99.to_csv(os.path.join(archive_path, base_99_file), index=False)
df_base_1.to_csv(os.path.join(archive_path, base_1_file), index=False)

print("Base.csv has been divided into base_99_%.csv and base_1_%.csv and saved in the archive folder.")
```

Instructions:

- Replace /path/to/your/archive/folder/ with the path to your archive folder.

**Step 4**

Combine all the variants with the name "All_Variants_Combined.csv"

"Account_Number" column needs to be changed with values from 1 to 50,00,000.

Save this .csv file in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.

Use these python scripts to do it:-
```python
import pandas as pd
import os

# Define file names for the variants
variant_files = ['Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
combined_file = 'All_Variants_Combined.csv'

# Path to your archive folder
archive_path = '/path/to/your/archive/folder/'

# List to hold individual dataframes
dfs = []

# Read each variant file and append to the list
for file in variant_files:
    df = pd.read_csv(os.path.join(archive_path, file))
    dfs.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Reset the "Account_Number" column to range from 1 to 5,000,000
combined_df['Account_Number'] = range(1, len(combined_df) + 1)

# Save the combined dataframe to a new .csv file in the archive folder
combined_df.to_csv(os.path.join(archive_path, combined_file), index=False)

print("All variant files have been combined into All_Variants_Combined.csv with updated Account_Number and saved in the archive folder.")
```

Instructions:

- Replace /path/to/your/archive/folder/ with the path to your archive folder.

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
conda create -n muledetect python=3.11.4
conda activate muledetect

pip install matplotlib==3.8.0
pip install pandas==1.5.3
pip install numpy==1.24.4
pip install scikit-learn==1.4.2
pip install joblib==1.2.0
pip install xgboost
pip install tensorflow
pip install pycaret
```

## Usage

Run it step wise to understand better.

**Step 1:**

- The notebook: [2. 0-5_months_train_6-7_test_base.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/2.%200-5_months_train_6-7_test_base.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 2:**

- The notebook: [3. 0-5_months_train_6-7_test_base_less.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/3.%200-5_months_train_6-7_test_base_less.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 3:**

- The notebook: [4. Base_train_Base_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/4.%20Base_train_Base_test.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 4:**

- The notebook: [5. Base_train_1st_half_test_2nd_half.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/5.%20Base_train_1st_half_test_2nd_half.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 5:**

- The notebook: [6. Base_train_99%_test_1%.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/6.%20Base_train_99%25_test_1%25.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 6:**

- The notebook: [7. All_variants_train_base_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/7.%20All_variants_train_base_test.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.

**Step 7:**

- The notebook: [8. Base_train_Variant_1_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/8.%20Base_train_Variant_1_test.ipynb)

- Follow the comments in the notebook.

- Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf) for more information.





The [first notebook](notebooks/generate_dataset_variants.ipynb) regards the process of sampling from a large dataset to obtain the different variants that constitute the suite of datasets.

The [second notebook](notebooks/empirical_results.ipynb) presents the train of 100 LightGBM models (hyperparameters selected through random search) on the suite of datasets, as well as plots of the results. 

To replicate the environment used in the experiments, install the  `requirements.txt` file via pip in a Python 3.7 environment. 

Additionally, you can find the official documentation of the suite of datasets in the `documents` folder. In this folder you will find the [Paper](documents/BAF_paper.pdf) and [Datasheet](documents/datasheet.pdf)

The paper contains more detailed information on motivation, generation of the base dataset and [variants](notebooks/generate_dataset_variants.ipynb) and an experiment performed in the [empirical results notebook](notebooks/empirical_results.ipynb).
The datasheet contains a summarized description of the dataset. 
