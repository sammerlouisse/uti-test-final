import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib   # ✅ NEW: using joblib instead of pickle

# ---------------------------------
# LOAD DATASET
# ---------------------------------
df = pd.read_csv("urinalysis_data.csv")

# Ensure target is lowercase for consistency
df["Diagnosis"] = df["Diagnosis"].str.lower()

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]  # "positive" / "negative"

# Categorical features (all columns)
categorical_cols = X.columns.tolist()

# ---------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ---------------------------------
# ML MODEL PIPELINE
# ---------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))
])

# ---------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------
model.fit(X_train, y_train)

# ---------------------------------
# TEST ACCURACY
# ---------------------------------
y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# ---------------------------------
# SAVE MODEL WITH JOBLIB
# ---------------------------------
joblib.dump(model, "model.joblib")   # ✅ SAVED WITH JOBLIB

print("\nModel saved as model.joblib")
