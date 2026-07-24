# MEDPREDICT
MEDICAL INSURANCE COST PREDICTION Group 15

# Medical-Insurance-Cost-Prediction
A machine learning regression project that predicts individual medical health insurance charges based on demographic, lifestyle, and health-related features.

##  Project Overview
This project focuses on predicting individual medical insurance costs using historical beneficiary data sourced from Kaggle. This task falls under **Supervised Learning** because the dataset contains historical records where both features (input variables such as age, BMI, and smoker status) and the continuous **target variable** (charges) are explicitly provided. Furthermore because the target variable is continuous (a numerical dollar amount spanning a real-value range), this problem is modeled as a **regression task** rather than a classification problem.

---

##  The Data Layout
The dataset consists of historical records tracking key health and demographic attributes of individual beneficiaries:

| Feature | Description | Data Type |
| :--- | :--- | :--- |
| age | Age of the primary beneficiary | Integer |
| sex | Gender of the beneficiary (male, female) | Object |
| bmi | Body Mass Index ($\text{kg/m}^2$) | Float |
| children | Number of children covered by health insurance | Integer |
| smoker | Smoking status (yes, no) | Object |
| region | Residential area (northeast, northwest, southeast, southwest) | Object |
| charges | Individual medical costs billed by health insurance (**Target Variable**) | Float |

---

* **Total Rows:** 1,338
* **Total Columns:** 7
* **Missing Values:** None (the dataset is complete across all columns)
* **Duplicate Rows:** 1 (handled during the data cleaning phase)
* **Categorical Encoding:** Categorical variables (sex, smoker, region) are encoded using `LabelEncoder` for model compatibility.
* **Feature Scaling:** Continuous variables are scaled using StandardScaler.
* **Status:** The dataset is fully processed and ready for exploratory data analysis (EDA), model training, and performance evaluation using Linear Regression and Decision Tree Regressor models.

## Data Importation and inspection
Below is the execution workflow covering data import, and  inspection:

```python
1. Import Required Libraries and Modules
import numpy as np
import pandas as pd


2. Importing the Dataset
data = pd.read_csv("/content/insurance.csv")

3. Data Inspection
print("Dataset Shape:", data.shape)  # Returns (1338, 7)

4. Data Preview
data.info()

5. Null Value Checks 
**  Checking for null values in numerical data 
if data.select_dtypes("object").isnull().sum().sum() == 0:
    print("No Null Values Found")
else:
    print("Missing Values:\n", data.isnull().sum())

** Checking for null values in catgorical data**
if data.select_dtypes("int64", "float64").isna().sum().sum() == 0:
  print("No Null Values Found")
else:
  print("Missing Values:\n", data.isna().sum())

6. Checking for duplicates
print("\nDuplicate Rows Count:", data.duplicated().sum())
