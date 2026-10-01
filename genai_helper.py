def generate_ai_insight(
    temperature,
    humidity,
    irradiance,
    wind_speed,
    panel_efficiency,
    predicted_power,
    svm_status,
    nb_status
):
    insights = []

    if predicted_power > 500:
        insights.append("The system is producing high power because irradiance is strong and operating conditions are favorable.")
    elif predicted_power > 300:
        insights.append("The system is producing moderate power under the current environmental conditions.")
    else:
        insights.append("The system is producing low power, likely due to weaker solar conditions or reduced efficiency.")

    if humidity > 75:
        insights.append("High humidity may be reducing solar performance.")
    elif humidity < 35:
        insights.append("Humidity is relatively low, which is generally favorable for panel performance.")

    if temperature > 38:
        insights.append("High temperature may reduce panel efficiency.")
    elif 20 <= temperature <= 32:
        insights.append("Temperature is within a favorable operating range.")

    if irradiance < 400:
        insights.append("Low irradiance is limiting energy generation.")
    elif irradiance > 800:
        insights.append("Strong irradiance is supporting higher solar output.")

    if wind_speed > 10:
        insights.append("Wind speed may help cool the panels slightly, which can support efficiency.")

    if panel_efficiency < 70:
        insights.append("Panel efficiency is low, indicating maintenance may be required.")
    elif panel_efficiency >= 85:
        insights.append("Panel efficiency is strong, suggesting the system is in healthy condition.")

    insights.append(f"SVM predicts the panel health as: {svm_status}.")
    insights.append(f"Naive Bayes predicts the panel health as: {nb_status}.")

    recommendations = []

    if panel_efficiency < 70:
        recommendations.append("Clean the solar panels and inspect for dust buildup or degradation.")
    if humidity > 75:
        recommendations.append("Monitor moisture-related performance loss and inspect for environmental exposure.")
    if predicted_power < 300:
        recommendations.append("Review irradiance conditions and panel efficiency to improve output.")
    if temperature > 38:
        recommendations.append("Monitor temperature-related efficiency drops during peak heat periods.")

    if not recommendations:
        recommendations.append("System conditions appear healthy; continue regular cleaning and monitoring.")
        recommendations.append("Maintain routine inspection schedules to sustain long-term efficiency.")

    output = "### 🤖 AI Explanation\n"
    for item in insights:
        output += f"- {item}\n"

    output += "\n### 🛠️ Recommendations\n"
    for item in recommendations[:3]:
        output += f"- {item}\n"

    return output


def answer_solar_question(question, predicted_power, svm_status, nb_status):
    q = question.lower().strip()

    if "why" in q and "low" in q:
        return (
            f"The output may be low because irradiance is weak, humidity is high, or panel efficiency is reduced. "
            f"The current predicted power is {predicted_power:.2f}, and the panel health indicators are "
            f"SVM = {svm_status}, Naive Bayes = {nb_status}."
        )

    if "why" in q and "high" in q:
        return (
            f"The output is likely high because irradiance is strong and the system is operating efficiently. "
            f"The current predicted power is {predicted_power:.2f}."
        )

    if "maintenance" in q or "clean" in q:
        return (
            "Maintenance should be considered when panel efficiency drops, dust accumulates, or health status shows "
            "maintenance required or critical failure."
        )

    if "factor" in q or "affect" in q:
        return (
            "The most important factors are usually irradiance, panel efficiency, humidity, and temperature. "
            "Irradiance tends to increase power, while very high humidity or temperature can reduce performance."
        )

    if "status" in q or "health" in q:
        return (
            f"The current health predictions are SVM = {svm_status} and Naive Bayes = {nb_status}. "
            "These indicate the likely operating condition of the solar panel system."
        )

    return (
        "Based on the current system conditions, solar performance depends mainly on irradiance, humidity, "
        "temperature, and panel efficiency. Maintaining panel cleanliness and monitoring environmental conditions "
        "can improve performance."
    )