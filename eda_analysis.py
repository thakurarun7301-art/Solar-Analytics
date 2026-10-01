import csv
import matplotlib.pyplot as plt

temperature = []
humidity = []
irradiance = []
power_output = []

# Read CSV manually
with open("data/solar_data.csv", "r") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        temperature.append(float(row["temperature"]))
        humidity.append(float(row["humidity"]))
        irradiance.append(float(row["irradiance"]))
        power_output.append(float(row["power_output"]))

print("✅ Data loaded successfully")

# Plot 1: Irradiance vs Power
plt.figure()
plt.scatter(irradiance, power_output)
plt.xlabel("Irradiance")
plt.ylabel("Power Output")
plt.title("Irradiance vs Power Output")
plt.savefig("data/irradiance_vs_power.png")
plt.close()

# Plot 2: Temperature vs Power
plt.figure()
plt.scatter(temperature, power_output)
plt.xlabel("Temperature")
plt.ylabel("Power Output")
plt.title("Temperature vs Power Output")
plt.savefig("data/temperature_vs_power.png")
plt.close()

# Plot 3: Humidity vs Power
plt.figure()
plt.scatter(humidity, power_output)
plt.xlabel("Humidity")
plt.ylabel("Power Output")
plt.title("Humidity vs Power Output")
plt.savefig("data/humidity_vs_power.png")
plt.close()

print("📊 Graphs created and saved in data folder!")