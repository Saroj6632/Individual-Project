# Hospital Readmission Prediction

# Project Goals
    This project seeks to create a model that classifies whether a patient is likely to be readmitted to the hospital within 30 days of discharge given the health conditions. This information will help Hospitals to determine individual patients' risk of returning in the hospital in this time period.


# Project description
    Hospitals in USA spent over $41 billion on patients who got readmitted within 30 days of discharge. Being able to determine factors that lead to higher readmission in such patients, and correspondingly being able to predict which patients will get readmitted can help hospitals save millions of dollars while improving quality of care.

# Deliverables
- Github Repository
i. README file
ii. Final Notebook
iii. .py modules with all the necessary functions to reproduce the notebook


# Key Questions 
1. Is there a relation between the race and chances of hospital readmission?
2. Does gender correlates to hospital readmission?
3. Does max_gluserum (glucose level) has relation with patient being readmitted within 30 days?
5. Does patients inpatient visits over the past year increases the chances of hospital rfeadmission?
6. Does patient with longer hospital stays before being discharged has higher chances of readmission?



# Data Dictionary
The data was obtained from Kaggle . The data contains attributes such as patient number, race, gender, age, admission type, time in hospital, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

The dataset has lack of class separation, presenting a challenge for generating predictions and class imbalance of approximately 90% - to - 10%

Out of 101766 Patients record, 99340 records were used for the analysis.



# Steps to Reproduce the Notebook
    - envy.py file with credentials for successful connection with CodeUp DB Server.
    - clone the project repo(including wrangle.py,fun_4.py)
    - import pandas, matplotlib, seaborn, numpy, sklearn libraries 
    - finally you should be able to run Hospital_Readmission_Report


# Pipeline

## Wrangle Data
    - Wrangling process includes both data acquisition and data preparation(data cleaning) process

### Acquire Data
    - wrangle.py module has all necessary functions to read the csv file.

### Data preparation
   wrangle.py module has function that cleans the acquired data
   - converted the invalid datas to np.nan and nulls were handled
   - unwanted columns were dropped
   - dummies for categorical columns were created


## Data Split
    resulting cleaned data was split into train, test and validate sample.
       

# Exploration
Goal: Address the initial questions through visualizations 


# Conclusion
## Summary
    - It was found that if the patient had more Inpatients stays in hospital over a past year the probability of readmission within 30 days was higher.
    - It was found that if the patient had spent more time in hospital ( considering serious illness ) the probability of readmission within 30 days was higher.
    - Race, gender, age  and insulin level, max_gluserum  are  also related to patient being readmitted 

## Recommendations
 - Three different classification models were created and iot was found that none of the models created outperformed the baseline model. Accuracy for the baseline model and 3 different classification model were same around 89%. So, I would recommend on tuning the hyperparameters. And also since there are many columns that can be clustered together, I would recommend approaching the cluster methodoligies.
 - I would recommend choosing a different metrics like recall beacuse readmission will increase the cost to the hospitals and if the algorithm/model predicts that the patients won't be admitted again within 30 days and if the patients gets admitted within 30 days it will cost hospitals so we need to reduce the false neagtive indications.


 ## Next Steps
 Approach a clustering methodologies. Instead of accuracy score other metrics coulde be chosen.

