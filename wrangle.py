import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', None)
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

sns.set_palette(palette='icefire')



# function to acquire the data
def get_diabetic_data():
    '''This function read the diabetic csv data from kaggle'''
    df= pd.read_csv('diabetic_kaggle_data.csv')
    return df




def clean_diabetic(df):
    '''Function to clean the csv file'''
    # replace ? with np.nan
    df.replace('?', np.nan , inplace=True)
    # drop the unwanted columns
    drop_na_list = ['examide' ,'citoglipton','weight','encounter_id','patient_nbr','payer_code','medical_specialty']  
    df.drop(columns=drop_na_list, axis=1, inplace=True)
    #replacing the column with 1 and O
    df.readmitted = [1 if each=='<30' else 0 for each in df.readmitted]
    df.race.replace(np.nan,'Other',inplace=True)
    df.gender.replace('Unknown/Invalid', np.nan , inplace=True)
    df.dropna(subset=['gender'], how='all', inplace = True)
    df= df.drop(columns=['admission_type_id', 'diag_1','diag_2', 'diag_3', 'admission_source_id', 'discharge_disposition_id',
    'change', 'num_lab_procedures', 'num_procedures', 'number_emergency','number_outpatient','number_diagnoses'])
    df['max_glu_serum'].replace(['None', 'Norm', '>300', '>200'],[-99,0,1,1]).astype(int)
    dummy_df = pd.get_dummies(df[['race', 'gender','age','metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide',  'insulin','max_glu_serum',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone','diabetesMed',
       'metformin-pioglitazone']])
    df = pd.concat([df, dummy_df], axis=1)
    df= df.drop(columns=['A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide',
    'tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone','diabetesMed','max_glu_serum'])
    df= df.drop(columns=['repaglinide_No', 'nateglinide_No', 'chlorpropamide_No','glimepiride_No', 'acetohexamide_No','glipizide_No','glyburide_No','tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No','acarbose_No','miglitol_No', 'troglitazone_No', 'tolazamide_No', 'insulin_No','glyburide-metformin_No','glipizide-metformin_No', 'glimepiride-pioglitazone_No','metformin-rosiglitazone_No', 'metformin-pioglitazone_No','diabetesMed_No'])
    return df





def data_split(df, stratify_by='readmitted'):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['readmitted','race', 'gender', 'age','number_inpatient','num_medications'])
    y_train = train['readmitted']
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['readmitted','race', 'gender', 'age','number_inpatient','num_medications'])
    y_validate = validate['readmitted']
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['readmitted','race', 'gender', 'age','number_inpatient','num_medications'])
    y_test = test['readmitted']
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



def generate_auc_roc_curve(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y,  y_pred_proba)
    auc = roc_auc_score(y, y_pred_proba)
    plt.figure(figsize=(8,5), dpi=100)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()
    

def generate_model_report(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted)
    recall = recall_score(y_actual, y_predicted)
    f1 = f1_score(y_actual, y_predicted)
    
    return accuracy, precision, recall, f1



def report_and_matrix(estimator, X_train, X_validate, y_train, y_validate):
    y_hat_train = estimator.predict(X_train)
    y_hat_validate = estimator.predict(X_validate)
    
    # confusion matrix plot
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5), dpi=100) 
    plt.suptitle('Patients Readmitted Within 30 Days')
    
    plot_confusion_matrix(estimator, X_train , y_train, ax=ax1, normalize='all')
    ax1.title.set_text('TRAINING')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual');
    #ax1.xaxis.set_ticklabels(['Yes', 'No']); ax1.yaxis.set_ticklabels(['No', 'Yes']);
    
    plot_confusion_matrix(estimator, X_validate , y_validate, ax=ax2, normalize='all')
    ax2.title.set_text('VALIDATE')
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual');
    #ax2.xaxis.set_ticklabels(['Yes', 'No']); ax2.yaxis.set_ticklabels(['No', 'Yes']);
    
    
    accuracy_train, precision_train, recall_train, f1_train = generate_model_report(y_train, y_hat_train)
    accuracy_validate, precision_validate, recall_validate, f1_validate = generate_model_report(y_validate, y_hat_validate)
    data = {
        'TRAIN':[accuracy_train, precision_train, recall_train, f1_train],
        'VALIDATE':[accuracy_validate, precision_validate, recall_validate, f1_validate]
    }
    df = pd.DataFrame(data, index =['ACCURACY', 'PERCISION', 'RECALL', 'F1'])
    display(df)

