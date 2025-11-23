# -----------------------------------------------------------
# train_loan_models.py
# -----------------------------------------------------------
# Entrena:
# - Regresión logística (sklearn)
# - Regresión logística (Keras) -> exportable a TFJS
# - Red neuronal densa (Keras)
#
# Genera:
# - scaler.joblib
# - le_education.joblib
# - le_self.joblib
# - scaler_info.json (para TFJS)
# - Matrices de confusión (.png)
# - Modelos Keras (SavedModel)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===========================================================
# 1. Cargar CSV
# ===========================================================

df = pd.read_csv("loan_approval_dataset.csv", sep=",", skipinitialspace=True)

# Asumimos que las columnas son EXACTAMENTE estas:
# loan_id, no_of_dependents, education, self_employed, income_annum,
# loan_amount, loan_term, cibil_score, residential_assets_value,
# commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status


# ===========================================================
# 2. Preparación del target
# ===========================================================

df["loan_status_bin"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})


# ===========================================================
# 3. Features + Label Encoding
# ===========================================================

X = df.drop(columns=["loan_id", "loan_status", "loan_status_bin"])
y = df["loan_status_bin"]

# Label Encoding de variables categóricas
le_education = LabelEncoder().fit(X["education"])
le_self = LabelEncoder().fit(X["self_employed"])

X["education_enc"] = le_education.transform(X["education"])
X["self_employed_enc"] = le_self.transform(X["self_employed"])

X = X.drop(columns=["education", "self_employed"])


# Orden fijo de columnas (IMPORTANTE para TFJS)
feature_cols = [
    "no_of_dependents",
    "education_enc",
    "self_employed_enc",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

X = X[feature_cols]


# ===========================================================
# 4. Escalado
# ===========================================================

scaler = StandardScaler().fit(X.astype(float))
X_scaled = scaler.transform(X.astype(float))

# Carpeta de artefactos
os.makedirs("artifacts", exist_ok=True)

# Guardar scaler y encoders
joblib.dump(scaler, "artifacts/scaler.joblib")
joblib.dump(le_education, "artifacts/le_education.joblib")
joblib.dump(le_self, "artifacts/le_self.joblib")

# Exportar scaler_info para usar en TFJS
scaler_info = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_order": feature_cols,
    "education_classes": le_education.classes_.tolist(),
    "self_employed_classes": le_self.classes_.tolist()
}

with open("artifacts/scaler_info.json", "w") as f:
    json.dump(scaler_info, f, indent=2)


# ===========================================================
# 5. Train/Test split
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ===========================================================
# 6. Modelo 1 — Regresión logística (sklearn)
# ===========================================================

lr_model = LogisticRegression(
    solver="lbfgs",
    max_iter=10000,
    class_weight=None,
    n_jobs=None,
    random_state=SEED
)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\n=== Logistic Regression (sklearn) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, zero_division=0))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_lr, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("True")
plt.xlabel("Pred")
plt.savefig("artifacts/cm_logistic_sklearn.png")

joblib.dump(lr_model, "artifacts/logistic_sklearn.joblib")


# ===========================================================
# 7. Modelo 2 — Regresión logística Keras (TFJS friendly)
# ===========================================================

keras.backend.clear_session()
logistic_keras = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation="sigmoid")
])

logistic_keras.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_logistic = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
]

logistic_keras.fit(
    X_train.astype("float32"),
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks_logistic,
    verbose=0
)

# Evaluación
y_prob_lk = logistic_keras.predict(X_test).ravel()
y_pred_lk = (y_prob_lk >= 0.5).astype(int)

print("\n=== Logistic Regression (Keras) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lk))
print(classification_report(y_test, y_pred_lk, zero_division=0))

cm_lk = confusion_matrix(y_test, y_pred_lk)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_lk, annot=True, cmap="Oranges", fmt="d")
plt.title("Confusion Matrix - Keras Logistic")
plt.savefig("artifacts/cm_logistic_keras.png")

# Guardar SavedModel (para TFJS)
# logistic_keras.save("artifacts/logistic_keras_savedmodel")
logistic_keras.save("artifacts/logistic_keras.keras", save_format="keras")
logistic_keras.export("artifacts/logistic_keras_savedmodel")

# ===========================================================
# 8. Modelo 3 — Red Neuronal Densa
# ===========================================================

keras.backend.clear_session()
nn = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation="sigmoid")
])

nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_nn = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=5e-6)
]

nn.fit(
    X_train.astype("float32"),
    y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks_nn,
    verbose=0
)

y_prob_nn = nn.predict(X_test).ravel()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)

print("\n=== Neural Network ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn, zero_division=0))

cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_nn, annot=True, cmap="Greens", fmt="d")
plt.title("Confusion Matrix - Neural Network")
plt.savefig("artifacts/cm_nn.png")

nn.save("artifacts/nn_model.keras", save_format="keras")
nn.export("artifacts/nn_savedmodel")

print("\n\n✔ Todo listo — modelos entrenados y artefactos generados en /artifacts/")
