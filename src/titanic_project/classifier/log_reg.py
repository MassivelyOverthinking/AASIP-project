#==========================================================================================
# IMPORTS & MODULES
#==========================================================================================

import pathlib

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

#==========================================================================================
# LOGISTIC REGRESSION
#==========================================================================================

# Start by loading the Titanic dataset from .CSV file.
titanic_data = pd.read_csv(pathlib.Path("/data/titanic_800.csv"))

# Drop irrelevant data columns ==> These pollute the model and reduce accuracy.
# - PassengerId ==> Unique identifier.
# - Name        ==> Individual name.
# - Ticket      ==> Unique ticket identifier.
# - Cabin       ==> Cabin number.
# - Fare        ==> Individual ticket price.
# - Embarked    ==> Location for embarkation.
titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Embarked"], inplace=True)

# Drop missing data rows ==> Used in the 'Age' column.
titanic_data.dropna(inplace=True)

# Create X and Y test arrays.
X = titanic_data.drop(columns=["Survival"])
y = titanic_data["Survival"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Preprocessing is fitted inside each CV training fold.
Log_model = LogisticRegression(
    max_iter=2000, 
    random_state=42
)

Log_model.fit(X_train, y_train)

y_pred = Log_model.predict(X_test)  
y_prob = Log_model.predict_proba(X_test)[:, 1]


print("\n========== TEST METRICS ==========")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Did not survive", "Survived"],
    zero_division=0,
))

# Confusion matrix: rows = actual, columns = predicted.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did not survive", "Survived"],
)