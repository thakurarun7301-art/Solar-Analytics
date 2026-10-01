import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load data manually
temperature = []
humidity = []
irradiance = []
wind_speed = []
efficiency = []
status = []

with open("data/solar_data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        temperature.append(float(row["temperature"]))
        humidity.append(float(row["humidity"]))
        irradiance.append(float(row["irradiance"]))
        wind_speed.append(float(row["wind_speed"]))
        efficiency.append(float(row["panel_efficiency"]))
        status.append(row["status"])

# Convert to numpy
X = np.array(list(zip(temperature, humidity, irradiance, wind_speed, efficiency)))
y = np.array(status)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2
)

# Scale data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# SVM MODEL
# -------------------------
print("\n--- SVM Model ---")
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average="weighted")

print("SVM Accuracy:", svm_acc)
print("SVM F1 Score:", svm_f1)

# -------------------------
# NAIVE BAYES
# -------------------------
print("\n--- Naive Bayes Model ---")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)

nb_acc = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average="weighted")

print("Naive Bayes Accuracy:", nb_acc)
print("Naive Bayes F1 Score:", nb_f1)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(nb_model, "models/nb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\n✅ Classification models saved!")