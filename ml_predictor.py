import pandas as pd
import joblib
# ============================================================
# LOAD OPERATIONAL DATA
# ============================================================

ml_data = pd.read_csv(
    "data/final_operational_data.csv"
)

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model_package = joblib.load(
    "models/demand_model.pkl"
)

model = model_package["model"]
preprocessor = model_package["preprocessor"]
feature_columns = model_package["feature_columns"]
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)
# ============================================================
# MODEL METRICS
# ============================================================

mae = 8.99
r2 = 0.2454
# Festival dates used in the original model testing
test_festival_dates = 0
# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "is_festival": "Festival effect",
    "is_holiday": "Holiday effect",
    "is_peak_hour": "Peak-hour effect",
    "is_night_demand": "Night-demand effect",
    "is_weekend": "Weekend effect",
    "historical_demand": "Historical demand",
    "hour": "Time of day",
    "month": "Month/season"
}


# ============================================================
# PREDICT DEMAND
# ============================================================

def predict_demand(
    route_id,
    stop_id,
    hour,
    day_of_week,
    day_type,
    month,
    is_weekend,
    is_holiday,
    is_festival,
    is_peak_hour,
    is_night_demand,
    historical_demand
):

    input_data = pd.DataFrame([{
        "route_id": int(route_id),
        "stop_id": int(stop_id),
        "hour": int(hour),
        "day_of_week": day_of_week,
        "day_type": day_type,
        "month": int(month),
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "is_festival": int(is_festival),
        "is_peak_hour": int(is_peak_hour),
        "is_night_demand": int(is_night_demand),
        "historical_demand": float(historical_demand)
    }])

    input_encoded = preprocessor.transform(
        input_data[feature_columns]
    )

    prediction = float(
        model.predict(input_encoded)[0]
    )

    return max(0, prediction)


# ============================================================
# EXPLANATION
# ============================================================

def explain_demand(
    route_id,
    stop_id,
    hour,
    day_of_week,
    day_type,
    month,
    is_weekend,
    is_holiday,
    is_festival,
    is_peak_hour,
    is_night_demand,
    historical_demand,
    event_name="Regular day"
):

    prediction = predict_demand(
        route_id=route_id,
        stop_id=stop_id,
        hour=hour,
        day_of_week=day_of_week,
        day_type=day_type,
        month=month,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        is_festival=is_festival,
        is_peak_hour=is_peak_hour,
        is_night_demand=is_night_demand,
        historical_demand=historical_demand
    )

    feature_values = {
        "route_id": int(route_id),
        "stop_id": int(stop_id),
        "hour": int(hour),
        "day_of_week": day_of_week,
        "day_type": day_type,
        "month": int(month),
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "is_festival": int(is_festival),
        "is_peak_hour": int(is_peak_hour),
        "is_night_demand": int(is_night_demand),
        "historical_demand": float(historical_demand)
    }

    drivers = []

    driver_features = [
        "is_festival",
        "is_holiday",
        "is_peak_hour",
        "is_night_demand",
        "is_weekend",
        "historical_demand",
        "hour",
        "month"
    ]

    for feature in driver_features:

        modified_values = feature_values.copy()

        if feature in [
            "is_festival",
            "is_holiday",
            "is_peak_hour",
            "is_night_demand",
            "is_weekend"
        ]:
            modified_values[feature] = 0

        elif feature == "historical_demand":
            modified_values[feature] = historical_demand

        elif feature == "hour":
            modified_values[feature] = 12

        elif feature == "month":
            modified_values[feature] = 6

        without_value = predict_demand(
            route_id=modified_values["route_id"],
            stop_id=modified_values["stop_id"],
            hour=modified_values["hour"],
            day_of_week=modified_values["day_of_week"],
            day_type=modified_values["day_type"],
            month=modified_values["month"],
            is_weekend=modified_values["is_weekend"],
            is_holiday=modified_values["is_holiday"],
            is_festival=modified_values["is_festival"],
            is_peak_hour=modified_values["is_peak_hour"],
            is_night_demand=modified_values["is_night_demand"],
            historical_demand=modified_values["historical_demand"]
        )

        delta = prediction - without_value

        drivers.append({
            "feature": feature,
            "with": round(prediction, 2),
            "without": round(without_value, 2),
            "delta": round(delta, 2)
        })

    drivers = sorted(
        drivers,
        key=lambda x: abs(x["delta"]),
        reverse=True
    )
    if is_festival == 1:
        festival_summary = (
            f"Festival day detected: {event_name}. "
            "Festival demand is expected to be higher than a regular day."
    )

    elif is_holiday == 1:
        festival_summary = (
            f"Holiday detected: {event_name}. "
            "Passenger demand may differ from a normal working day."
    )

    else:
        festival_summary = (
            "Regular day detected. "
            "Demand is estimated using time, day type, route, stop and historical demand patterns."
    )   

    return {
        "prediction": prediction,
        "festival_summary": festival_summary,
        "drivers": drivers
    }