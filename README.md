# **Mule Account Detection:** Bank Account Fraud

## Description

> **Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf)**

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

> **Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf)**

- This repository contains all the code required to replicate the project.
- The code is thoroughly commented to ensure clarity and ease of understanding.


## Dataset Preparation

> **Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf)**

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

From the Base.csv file remove nine low-priority features and create a new .csv file with the name base_less_cols.csv.

Low-priority columns: housing_status, phone_home_valid, phone_mobile_valid, bank_months_count, has_other_cards, proposed_credit_limit, source, device_os, device_fraud_count.

Save this .csv files in the archive folder.

Use these python scripts to do it:-
```python
import pandas as pd
import os

# Define the file names
base_file = 'Base.csv'
new_file = 'base_less_cols.csv'

# Low-priority columns to remove
columns_to_remove = [
    'housing_status', 'phone_home_valid', 'phone_mobile_valid',
    'bank_months_count', 'has_other_cards', 'proposed_credit_limit',
    'source', 'device_os', 'device_fraud_count'
]

# Path to your archive folder
archive_path = '/path/to/your/archive/folder/'

# Read the original Base.csv file
df = pd.read_csv(os.path.join(archive_path, base_file))

# Drop the low-priority columns
df_less_cols = df.drop(columns=columns_to_remove)

# Save the new dataframe to a .csv file
df_less_cols.to_csv(os.path.join(archive_path, new_file), index=False)

print("Low-priority columns have been removed from Base.csv and saved as base_less_cols.csv in the archive folder.")
```

Instructions:

- Replace /path/to/your/archive/folder/ with the path to your archive folder.

**Step 6**

Divide the base_less_cols.csv file into two .csv files:

base_less_cols_half1.csv: "Account_Number" column from 1 to 5,00,000.

base_less_cols_half2.csv: "Account_Number" column from 5,00,001 to 10,00,000.

Save these .csv files in the [archive](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/tree/main/Mule_Account_Detection/32%20Column%20Dataset/archive) folder.


Use these python scripts to do it:-
```python
import pandas as pd
import os

# Define file names
input_file = 'base_less_cols.csv'
half1_file = 'base_less_cols_half1.csv'
half2_file = 'base_less_cols_half2.csv'

# Path to your archive folder
archive_path = '/path/to/your/archive/folder/'

# Read the base_less_cols.csv file
df = pd.read_csv(os.path.join(archive_path, input_file))

# Split the data into two halves based on "Account_Number"
df_half1 = df[df['Account_Number'] <= 500000]
df_half2 = df[df['Account_Number'] > 500000]

# Save the two halves into separate .csv files
df_half1.to_csv(os.path.join(archive_path, half1_file), index=False)
df_half2.to_csv(os.path.join(archive_path, half2_file), index=False)

print("base_less_cols.csv has been divided into base_less_cols_half1.csv and base_less_cols_half2.csv and saved in the archive folder.")
```
Instructions:

- Replace /path/to/your/archive/folder/ with the path to your archive folder.

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

> **Refer this [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf)**

Run it step wise to understand better.

**Note:** Training the models will take time.

**Note:** The final model, which achieved an accuracy of 99%, is detailed in step 6. The other steps serve as foundational elements that contributed to the development and understanding of the model. To gain a comprehensive understanding of the process from the ground up, it is essential to review all steps. For a detailed, step-by-step explanation, please **refer to the [document](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/Mule%20Account%20Detection.pdf).** Additionally, reviewing the comments within the notebook will provide further insights.

**Note:**
- The model is trained on a dataset that captures the relationship between features and the classification of accounts as fraudulent or non-fraudulent.
- Upon completing the training, the model applies this learned knowledge to a testing dataset to estimate the risk percentage associated with each account.
- A threshold is then set on this risk score to identify and generate a list of accounts flagged as fraudulent (risk > threshold).

**Step 1:**

> Divided the dataset based on "month".

> Testing data is unseen.

- The notebook: [2. 0-5_months_train_6-7_test_base.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/2.%200-5_months_train_6-7_test_base.ipynb)

- Follow the comments in the notebook.

**Step 2:**

> Divided the dataset based on "month". Removed the low-priority features.

> Testing data is unseen.

- The notebook: [3. 0-5_months_train_6-7_test_base_less.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/3.%200-5_months_train_6-7_test_base_less.ipynb)

- Follow the comments in the notebook.

**Step 3:**

> Testing data is seen.

- The notebook: [4. Base_train_Base_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/4.%20Base_train_Base_test.ipynb)

- Follow the comments in the notebook.

**Step 4:**

> Divided the dataset in the ratio 50:50.

> Testing data is unseen.

- The notebook: [5. Base_train_1st_half_test_2nd_half.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/5.%20Base_train_1st_half_test_2nd_half.ipynb)

- Follow the comments in the notebook.

**Step 5:**

> Divided the dataset in the ratio 99:1.

> Testing data is unseen.

- The notebook: [6. Base_train_99%_test_1%.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/6.%20Base_train_99%25_test_1%25.ipynb)

- Follow the comments in the notebook.

**Step 6:**

> Combined all variants to create a larger and more diverse training dataset.

> Testing data is unseen.

- The notebook: [7. All_variants_train_base_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/7.%20All_variants_train_base_test.ipynb)

- Follow the comments in the notebook.

***To verify the uniqueness of the .csv files— Base, Variant I, Variant II, Variant III, Variant IV, and Variant V, I executed step 7.***

***The low accuracy observed in step 7 indicates that these .csv files are distinct and poorly correlated, thus confirming that the testing data is indeed unseen.***

**Step 7:**

- The notebook: [8. Base_train_Variant_1_test.ipynb](https://github.com/Aggarwal-Gavesh-25/Mule-Account-Detection/blob/main/Mule_Account_Detection/32%20Column%20Dataset/scripts/baselinemodels-roc/8.%20Base_train_Variant_1_test.ipynb)

- Follow the comments in the notebook.

## All the best
