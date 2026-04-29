"""
LING 539 Term Project : Duncan A. Yuen

If, and only if, either the vectorizer or model is missing:
This must be run first; the classifier must be run second.
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
vectorizer.joblib
model.joblib
"""
# =========================
# IMPORTS
# =========================
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion # LLM Import

# =========================
# LOADING DATA
# =========================
trainData = pd.read_csv("train.csv")
trainText = trainData["TEXT"].fillna("").tolist()
trainLabel = trainData["LABEL"].values

# =========================
# GATHER FEATURES IN MODEL
# =========================
# Initially wrote both vectorizers separately;
# Asked the LLM to merge them, hence featureUnion
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(min_df=3, sublinear_tf=True, ngram_range=(1, 2))),
    ("char", TfidfVectorizer(min_df=3, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5))),
])
trainMatrix = vectorizer.fit_transform(trainText)

# Training a model on the vectorizer features
model = LogisticRegression(solver="liblinear", class_weight="balanced")
model.fit(trainMatrix, trainLabel)

# =========================
# EXPORT PRETRAINING DATA
# =========================
# Initially exported both vectorizers separately;
# Merged the lines when combined via featureUnion
joblib.dump(vectorizer, "vectorizer.joblib")
# Export the model which has been made.
joblib.dump(model, "model.joblib")
