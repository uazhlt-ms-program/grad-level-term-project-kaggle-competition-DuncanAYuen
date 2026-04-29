"""
LING 539 Term Project : Duncan A. Yuen

If, and only if, either the vectorizer or model is missing:
This must be run second, and the trainer must be run first.
(Hint: The both of them are preprocessed in my submission.)

Inspired by my term project in Prof Sandiway's 388 class two years ago,
wherein I did the same thing, I have elected to make 2 files. The first
trains on a set of data and saves it in a format for import by another.
This allows for the downstream file to be iterated quickly without need
to wait as it retrains each time. It also just makes it easier to run.

For this purpose I have added the joblib library which can quickly save
and load data. Each other import is found in requirements.txt file with
which we have been supplied.

File Outputs:
One CSV named 'submission.csv' with two columns: ID and Prediction
"""
# =========================
# IMPORTS
# =========================
import joblib
import pandas as pd

# =========================
# LOAD TRAINING + TEST DATA
# =========================
# Training Data Imports
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")
# Testing Data Import
testData = pd.read_csv("test.csv")
testText = testData["TEXT"].fillna("").tolist()

# =========================
# PREDICT AND EXPORT LABELS
# =========================
testMatrix = vectorizer.transform(testText) # Create a feature matrix for test data.
predictedLabels = model.predict(testMatrix) # Process predictions based on the feature matrix.
# Export predictions as a CSV with one column for the datapoint ID and one for the assigned label.
pd.DataFrame({"ID": testData["ID"], "LABEL": predictedLabels}).to_csv("submission.csv", index=False)