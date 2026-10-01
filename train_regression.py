import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load data manually
temperature = []
humidity = []
irradiance = []
wind_speed = []
efficiency = []
power_output = []

with open("data/solar_data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        temperature.append(float(row["temperature"]))
        humidity.append(float(row["humidity"]))
        irradiance.append(float(row["irradiance"]))
        wind_speed.append(float(row["wind_speed"]))
        efficiency.append(float(row["panel_efficiency"]))
        power_output.append(float(row["power_output"]))

# Convert to numpy arrays
X = np.array(list(zip(temperature, humidity, irradiance, wind_speed, efficiency)))
y = np.array(power_output)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------
# UNDERFITTING (only 1 feature)
# -------------------------
print("\n--- Underfitting Example ---")
X_train_under = X_train[:, 0].reshape(-1, 1)  # only temperature
X_test_under = X_test[:, 0].reshape(-1, 1)

model_under = LinearRegression()
model_under.fit(X_train_under, y_train)

y_pred_under = model_under.predict(X_test_under)

print("Underfit RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_under)))
print("Underfit R2:", r2_score(y_test, y_pred_under))

# -------------------------
# GOOD MODEL
# -------------------------
print("\n--- Best Model ---")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/regression_model.pkl")

print("\n✅ Regression model saved!")

# -------------------------
# OVERFITTING DEMO (fake simulation)
# -------------------------
print("\n--- Overfitting Demo (conceptual) ---")
print("High complexity models can memorize noise → bad generalization")