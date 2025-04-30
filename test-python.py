
# from prefect import flow
# import datetime

# @flow(log_prints=True)
# def my_flow():
#     try:
    
#         current_time = datetime.datetime.now()
#         print(current_time.strftime("%H:%M:%S"))
#     except:
#         print("ERROR")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from mlflow.types import ColSpec, Schema
import warnings
import pandas as pd

from imblearn.combine import SMOTETomek

@flow(log_prints=True)
def my_flow():
    warnings.filterwarnings('ignore')
    
    print("STEP 1")
    # Step 1: Create an imbalanced binary classification dataset
    x, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                               weights=[0.9, 0.1], flip_y=0, random_state=42)
    
    np.unique(y, return_counts=True)
    
    
    
    print("STEP 2")
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    
    
    print("STEP 3")
    
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
    np.unique(y_train_res, return_counts=True)
    
    print("STEP 4")
    models = [
        (
            "Logistic Regression", 
            {"C": 1, "solver": 'liblinear'},
            LogisticRegression(), 
            (X_train, y_train),
            (X_test, y_test),
            pd.DataFrame({"feature1": [1.2], "feature2": [2.3], "feature3": [3.4], "feature4": [4.5], "feature5": [5.6], "feature6": [6.7], "feature7": [7.8], "feature8": [8.9], "feature9": [9.0], "feature10": [3.4]}),
          ),
        (
            "Random Forest", 
            {"n_estimators": 30, "max_depth": 3},
            RandomForestClassifier(), 
            (X_train, y_train),
            (X_test, y_test),
            pd.DataFrame({"feature1": [1.2], "feature2": [2.3], "feature3": [3.4], "feature4": [4.5], "feature5": [5.6], "feature6": [6.7], "feature7": [7.8], "feature8": [8.9], "feature9": [9.0], "feature10": [3.4]}),
         ),
        (
            "XGBClassifier",
            {"use_label_encoder": False, "eval_metric": 'logloss'},
            XGBClassifier(), 
            (X_train, y_train),
            (X_test, y_test),
            pd.DataFrame({"feature1": [1.2], "feature2": [2.3], "feature3": [3.4], "feature4": [4.5], "feature5": [5.6], "feature6": [6.7], "feature7": [7.8], "feature8": [8.9], "feature9": [9.0], "feature10": [3.4]}),
             ),
        (
            "XGBClassifier With SMOTE",
            {"use_label_encoder": False, "eval_metric": 'logloss'},
            XGBClassifier(), 
            (X_train_res, y_train_res),
            (X_test, y_test),
            pd.DataFrame({"feature1": [1.2], "feature2": [2.3], "feature3": [3.4], "feature4": [4.5], "feature5": [5.6], "feature6": [6.7], "feature7": [7.8], "feature8": [8.9], "feature9": [9.0], "feature10": [3.4]}),
         )
    ]
    
    
    print("STEP 5")
    reports = []
    
    for model_name, params, model, train_set, test_set,input_example in models:
        X_train = train_set[0]
        y_train = train_set[1]
        X_test = test_set[0]
        y_test = test_set[1]
        
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)
        print(f"model: {model}")

