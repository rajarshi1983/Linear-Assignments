# pima_train_sklearn.py
# Train a neural-network (MLPClassifier) model for Pima Indians Diabetes

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from joblib import dump

# ---- 1) Load data ------------------------------------------------------------
# Public raw CSV with 768 rows, 8 features + target (last column)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names (per dataset description)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFn", "Age", "Outcome"
]

try:
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
except Exception as e:
    raise SystemExit(
        f"Could not load from URL ({e}). If offline, download the CSV and set\n"
        "DATA_URL = '/path/to/pima-indians-diabetes.csv'"
    )

# Optional: simple zero-to-NaN cleanup for biologically impossible zeros
# (Glucose, BloodPressure, SkinThickness, Insulin, BMI shouldn't be zero)
to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[to_fix] = df[to_fix].replace(0, np.nan)
# Impute with median (simple & robust)
df[to_fix] = df[to_fix].fillna(df[to_fix].median())

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# ---- 2) Train/validation split ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 3) Pipeline + Hyperparameter tuning ------------------------------------
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(max_iter=500, random_state=42))
])

param_grid = {
    "clf__hidden_layer_sizes": [(16, 8), (32, 16), (32,), (64, 32)],
    "clf__alpha": [1e-4, 1e-3, 1e-2],             # L2 regularization
    "clf__learning_rate_init": [1e-3, 5e-4, 1e-4]
}

grid = GridSearchCV(
    pipe, param_grid=param_grid,
    scoring="roc_auc", cv=5, n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
best_model = grid.best_estimator_

# ---- 4) Evaluation -----------------------------------------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.3f}")
print(f"ROC-AUC:  {auc:.3f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:\n", cm)

# ---- 5) ROC curve plot -------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"MLP (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Pima Indians Diabetes (MLPClassifier)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("pima_mlp_roc.png", dpi=160)
plt.close()
print("Saved ROC curve to pima_mlp_roc.png")

# ---- 6) Save the trained model ----------------------------------------------
dump(best_model, "pima_mlp.joblib")
print("Saved trained model to pima_mlp.joblib")

# ---- 7) Helper: single prediction -------------------------------------------
def predict_single(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age, threshold=0.5
):
    """
    Returns (probability, label) using the saved pipeline (scaler + MLP).
    """
    x = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, dpf, age]], dtype=float)
    prob = best_model.predict_proba(x)[0, 1]
    label = int(prob >= threshold)
    return prob, label

# Example usage:
if __name__ == "__main__":
    prob, label = predict_single(
        pregnancies=2, glucose=130, blood_pressure=70, skin_thickness=20,
        insulin=85, bmi=30.5, dpf=0.5, age=35
    )
    print(f"Example prediction → Prob={prob:.3f}, Label={label} (1=diabetic)")