import numpy as np
import csv
import os

np.random.seed(42)
n = 1000

temperature = np.random.uniform(15, 45, n)
humidity = np.random.uniform(20, 95, n)
irradiance = np.random.uniform(200, 1100, n)
wind_speed = np.random.uniform(0, 15, n)
dust_level = np.random.uniform(0, 50, n)

panel_efficiency = 100 - (dust_level * 0.8) - (np.maximum(temperature - 35, 0) * 0.7)
panel_efficiency = np.clip(panel_efficiency, 40, 100)

noise = np.random.normal(0, 20, n)

power_output = (
    irradiance * 0.55
    - humidity * 0.8
    - np.abs(temperature - 28) * 2.5
    + wind_speed * 1.2
    + panel_efficiency * 1.5
    + noise
)

power_output = np.clip(power_output, 50, None)

status = []
for eff in panel_efficiency:
    if eff >= 85:
        status.append("Optimal")
    elif eff >= 60:
        status.append("Maintenance Required")
    else:
        status.append("Critical Failure")

# Create data folder
os.makedirs("data", exist_ok=True)

# Write CSV manually (no pandas)
with open("data/solar_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    writer.writerow([
        "temperature", "humidity", "irradiance",
        "wind_speed", "dust_level", "panel_efficiency",
        "power_output", "status"
    ])
    
    for i in range(n):
        writer.writerow([
            temperature[i],
            humidity[i],
            irradiance[i],
            wind_speed[i],
            dust_level[i],
            panel_efficiency[i],
            power_output[i],
            status[i]
        ])

print("✅ Dataset created successfully!")
print("📁 File saved: data/solar_data.csv")