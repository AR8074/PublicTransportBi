import pandas as pd
import numpy as np
import mysql.connector

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import os
import joblib


# ============================================================
# 1. LOAD REAL MTC GTFS DATA
# ============================================================

routes = pd.read_csv("data/raw/routes.txt")
trips = pd.read_csv("data/raw/trips.txt")
stops = pd.read_csv("data/raw/stops.txt")
stop_times = pd.read_csv("data/raw/stop_times.txt")

print("Routes:", routes.shape)
print("Trips:", trips.shape)
print("Stops:", stops.shape)
print("Stop times:", stop_times.shape)


# ============================================================
# 2. CREATE BASE GTFS SAMPLE
# ============================================================

np.random.seed(42)

base_data = stop_times.sample(
    1000,
    random_state=42
).copy()

print("\nBase GTFS records:", len(base_data))


# ============================================================
# 3. CONNECT TRIPS WITH ROUTES
# ============================================================

base_data = base_data.merge(
    trips[["trip_id", "route_id"]],
    on="trip_id",
    how="left"
)


# ============================================================
# 4. CONNECT STOPS WITH STOP DETAILS
# ============================================================

base_data = base_data.merge(
    stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
    on="stop_id",
    how="left"
)


# ============================================================
# 5. ADD ROUTE INFORMATION
# ============================================================

route_columns = ["route_id", "route_short_name", "route_long_name"]
available_route_columns = [
    column for column in route_columns if column in routes.columns
]

base_data = base_data.merge(
    routes[available_route_columns],
    on="route_id",
    how="left"
)


# ============================================================
# 6. CREATE FULL-YEAR OPERATIONAL HISTORY
# ============================================================

dates = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
print("\nOperating days:", len(dates))

sample_data = pd.concat(
    [base_data.assign(service_date=date) for date in dates],
    ignore_index=True
)

print("Expanded records:", len(sample_data))


# ============================================================
# 7. CREATE HOUR
# ============================================================

sample_data["hour"] = (
    sample_data["arrival_time"]
    .str.split(":")
    .str[0]
    .astype(int) % 24
)


# ============================================================
# 8. CREATE DAY OF WEEK
# ============================================================

sample_data["day_of_week"] = pd.to_datetime(
    sample_data["service_date"]
).dt.day_name()


# ============================================================
# 9. CREATE MONTH INFORMATION
# ============================================================

sample_data["month"] = pd.to_datetime(sample_data["service_date"]).dt.month
sample_data["month_name"] = pd.to_datetime(sample_data["service_date"]).dt.month_name()


# ============================================================
# 10. CREATE WEEKEND / WEEKDAY FLAG
# ============================================================

sample_data["day_type"] = np.where(
    sample_data["day_of_week"].isin(["Saturday", "Sunday"]),
    "Weekend",
    "Weekday"
)

# Synthetic prototype calendar flags.
# These are assumptions for demonstration, not actual MTC demand statistics.
holiday_dates = pd.to_datetime([
    "2026-01-01", "2026-01-14", "2026-01-26", "2026-02-03",
    "2026-03-19", "2026-03-21", "2026-04-03", "2026-04-14",
    "2026-05-01", "2026-05-27", "2026-06-17", "2026-08-15",
    "2026-08-26", "2026-09-04", "2026-09-14", "2026-10-02",
    "2026-10-20", "2026-10-21", "2026-11-08", "2026-12-25"
])

festival_dates = pd.to_datetime([
    "2026-01-14", "2026-03-19", "2026-04-14", "2026-08-26",
    "2026-09-04", "2026-10-20", "2026-10-21", "2026-11-08"
])

sample_data["is_holiday"] = pd.to_datetime(sample_data["service_date"]).isin(holiday_dates).astype(int)
sample_data["is_festival"] = pd.to_datetime(sample_data["service_date"]).isin(festival_dates).astype(int)


# ============================================================
# 11. DEMAND PATTERN FEATURES
# ============================================================

sample_data["is_peak_hour"] = np.where(
    ((sample_data["hour"] >= 7) & (sample_data["hour"] <= 9)) |
    ((sample_data["hour"] >= 17) & (sample_data["hour"] <= 19)),
    1,
    0
)

sample_data["is_night_demand"] = np.where(
    ((sample_data["day_of_week"] == "Friday") & (sample_data["hour"] >= 19)) |
    ((sample_data["day_of_week"] == "Sunday") & (sample_data["hour"] >= 18)),
    1,
    0
)


# ============================================================
# 12. GENERATE REALISTIC DELAY
# ============================================================

np.random.seed(42)

sample_data["delay_minutes"] = np.clip(
    np.random.normal(loc=4, scale=5, size=len(sample_data)).round(),
    0,
    25
).astype(int)


# ============================================================
# 13. PASSENGER DEMAND GENERATION
# ============================================================

base_passengers = np.random.randint(15, 35, size=len(sample_data))

peak_multiplier = np.where(
    ((sample_data["hour"] >= 7) & (sample_data["hour"] <= 9)) |
    ((sample_data["hour"] >= 17) & (sample_data["hour"] <= 19)),
    1.5,
    1.0
)

weekend_multiplier = np.where(
    sample_data["day_of_week"] == "Saturday",
    0.90,
    np.where(sample_data["day_of_week"] == "Sunday", 0.85, 1.0)
)

night_multiplier = np.where(
    ((sample_data["day_of_week"] == "Friday") & (sample_data["hour"] >= 19)) |
    ((sample_data["day_of_week"] == "Sunday") & (sample_data["hour"] >= 18)),
    1.4,
    1.0
)

month_multiplier = np.where(
    sample_data["month"].isin([1, 4, 5, 6, 10, 11, 12]),
    1.10,
    1.0
)

random_variation = np.random.uniform(0.85, 1.15, size=len(sample_data))

sample_data["passengers"] = (
    base_passengers
    * peak_multiplier
    * weekend_multiplier
    * night_multiplier
    * month_multiplier
    * np.where(sample_data["is_holiday"] == 1, 1.25, 1.0)
    * np.where(sample_data["is_festival"] == 1, 1.40, 1.0)
    * random_variation
).round().astype(int)

sample_data["passengers"] = sample_data["passengers"].clip(5, 60)


# ============================================================
# 14. MAKE HIGH-DEMAND PERIODS SLIGHTLY MORE DELAYED
# ============================================================

additional_delay = np.where(
    sample_data["passengers"] >= 45,
    np.random.randint(0, 4, size=len(sample_data)),
    0
)

sample_data["delay_minutes"] = (
    sample_data["delay_minutes"] + additional_delay
).clip(0, 25).astype(int)


# ============================================================
# 15. GENERATE ACTUAL ARRIVAL TIME
# ============================================================

def add_delay(gtfs_time, delay):
    hours, minutes, seconds = map(int, gtfs_time.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds + delay * 60
    new_hours = total_seconds // 3600
    remaining = total_seconds % 3600
    new_minutes = remaining // 60
    new_seconds = remaining % 60
    return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d}"

sample_data["actual_arrival_time"] = sample_data.apply(
    lambda row: add_delay(row["arrival_time"], row["delay_minutes"]),
    axis=1
)


# ============================================================
# 16. TICKET TYPE
# ============================================================

ticket_types = ["Single", "Daily Pass", "Student Concession"]
sample_data["ticket_type"] = np.random.choice(
    ticket_types,
    size=len(sample_data),
    p=[0.65, 0.10, 0.25]
)


# ============================================================
# 17. FARE
# ============================================================

fare_map = {
    "Single": 20,
    "Daily Pass": 50,
    "Student Concession": 10
}

sample_data["fare"] = sample_data["ticket_type"].map(fare_map)


# ============================================================
# 18. REVENUE
# ============================================================

sample_data["revenue"] = sample_data["passengers"] * sample_data["fare"]


# ============================================================
# 19. BUS INFORMATION
# ============================================================

bus_ids = [f"MTC-{i:03d}" for i in range(1, 51)]
sample_data["bus_id"] = np.random.choice(bus_ids, size=len(sample_data))
sample_data["capacity"] = np.random.choice([40, 50, 55, 60], size=len(sample_data))
sample_data["bus_status"] = np.random.choice(
    ["Available", "Maintenance"],
    size=len(sample_data),
    p=[0.90, 0.10]
)


# ============================================================
# 20. PASSENGER LOAD FACTOR
# ============================================================

sample_data["load_factor"] = (
    sample_data["passengers"] / sample_data["capacity"] * 100
).round(2)


# ============================================================
# 21. ON-TIME STATUS
# ============================================================

sample_data["on_time"] = np.where(
    sample_data["delay_minutes"] <= 5,
    "On Time",
    "Delayed"
)


# ============================================================
# 22. REVENUE PER PASSENGER
# ============================================================

sample_data["revenue_per_passenger"] = (
    sample_data["revenue"] / sample_data["passengers"]
).round(2)


# ============================================================
# 23. DISPLAY DATA SUMMARY
# ============================================================

otp = (sample_data["on_time"] == "On Time").mean() * 100

print("\n========================================")
print("        DATASET SUMMARY")
print("========================================")
print("Total records:", len(sample_data))
print("Start date:", sample_data["service_date"].min())
print("End date:", sample_data["service_date"].max())
print("Number of days:", sample_data["service_date"].nunique())
print("Number of routes:", sample_data["route_id"].nunique())
print("Number of stops:", sample_data["stop_id"].nunique())
print("Number of buses:", sample_data["bus_id"].nunique())
print("Total passengers:", sample_data["passengers"].sum())
print("Total revenue:", round(sample_data["revenue"].sum(), 2))
print("Average delay:", round(sample_data["delay_minutes"].mean(), 2), "minutes")
print("Average load factor:", round(sample_data["load_factor"].mean(), 2), "%")
print("Overall OTP:", round(otp, 2), "%")


# ============================================================
# 24. SHOW SAMPLE DATA
# ============================================================

print("\n========================================")
print("        SAMPLE FINAL DATA")
print("========================================")

columns_to_show = [
    "service_date",
    "day_of_week",
    "hour",
    "trip_id",
    "route_id",
    "stop_id",
    "passengers",
    "delay_minutes",
    "ticket_type",
    "revenue",
    "bus_id",
    "load_factor",
    "on_time"
]

print(sample_data[columns_to_show].head(10))


# ============================================================
# 25. MONTHLY SUMMARY
# ============================================================

monthly_summary = sample_data.groupby("month_name").agg(
    total_passengers=("passengers", "sum"),
    total_revenue=("revenue", "sum"),
    average_delay=("delay_minutes", "mean"),
    average_load_factor=("load_factor", "mean")
).round(2)

print("\n========================================")
print("        MONTHLY SUMMARY")
print("========================================")
print(monthly_summary)


# ============================================================
# 26. SAVE FINAL OPERATIONAL DATA
# ============================================================

final_columns = [
    "service_date",
    "month",
    "month_name",
    "hour",
    "day_of_week",
    "day_type",
    "trip_id",
    "route_id"
]

if "route_short_name" in sample_data.columns:
    final_columns.append("route_short_name")

if "route_long_name" in sample_data.columns:
    final_columns.append("route_long_name")

final_columns += [
    "stop_id",
    "stop_name",
    "stop_lat",
    "stop_lon",
    "arrival_time",
    "actual_arrival_time",
    "delay_minutes",
    "passengers",
    "ticket_type",
    "fare",
    "revenue",
    "revenue_per_passenger",
    "bus_id",
    "capacity",
    "bus_status",
    "load_factor",
    "on_time"
]

sample_data[final_columns].to_csv("data/final_operational_data.csv", index=False)

print("\nFinal operational data saved successfully.")
print("File: data/final_operational_data.csv")


# ============================================================
# 27. PREPARE ML DATA
# ============================================================

# Use the freshly generated operational dataset for ML so the synthetic
# holiday/festival demand effects above are included in training.
ml_data = sample_data.copy()

print("\nML training data source: generated operational dataset")


# ============================================================
# 28. ENSURE ML FEATURE COLUMNS EXIST
# ============================================================

if "service_date" in ml_data.columns:
    ml_data["service_date"] = pd.to_datetime(ml_data["service_date"])

if "day_of_week" not in ml_data.columns:
    ml_data["day_of_week"] = pd.to_datetime(ml_data["service_date"]).dt.day_name()

if "day_type" not in ml_data.columns:
    ml_data["day_type"] = np.where(
        ml_data["day_of_week"].isin(["Saturday", "Sunday"]),
        "Weekend",
        "Weekday"
    )

if "is_weekend" not in ml_data.columns:
    ml_data["is_weekend"] = np.where(
        ml_data["day_of_week"].isin(["Saturday", "Sunday"]),
        1,
        0
    )


ml_data["is_peak_hour"] = np.where(
    ((ml_data["hour"] >= 7) & (ml_data["hour"] <= 9)) |
    ((ml_data["hour"] >= 17) & (ml_data["hour"] <= 19)),
    1,
    0
)

ml_data["is_night_demand"] = np.where(
    ((ml_data["day_of_week"] == "Friday") & (ml_data["hour"] >= 19)) |
    ((ml_data["day_of_week"] == "Sunday") & (ml_data["hour"] >= 18)),
    1,
    0
)

print("\n========================================")
print("       ML FEATURE DATASET")
print("========================================")
print("Shape:", ml_data.shape)
print("\nColumns:")
print(ml_data.columns.tolist())
print("\nFirst 5 rows:")
print(ml_data.head())


# ============================================================
# 29. CREATE HISTORICAL DEMAND FEATURE
# ============================================================

ml_data = ml_data.sort_values(["route_id", "stop_id", "hour", "service_date"]).reset_index(drop=True)

if "passengers" in ml_data.columns:
    ml_data["historical_demand"] = (
        ml_data.groupby(["route_id", "stop_id", "hour"])["passengers"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )
    ml_data["historical_demand"] = ml_data["historical_demand"].fillna(ml_data["passengers"].mean())
else:
    ml_data["historical_demand"] = 0

print("\n========================================")
print("       HISTORICAL DEMAND")
print("========================================")
print(
    ml_data[["service_date", "route_id", "stop_id", "hour", "passengers", "historical_demand"]]
    .head(10)
)


# ============================================================
# 30. PREPARE ML FEATURES
# ============================================================

feature_columns = [
    "route_id",
    "stop_id",
    "hour",
    "day_of_week",
    "day_type",
    "month",
    "is_weekend",
    "is_holiday",
    "is_festival",
    "is_peak_hour",
    "is_night_demand",
    "historical_demand"
]

target_column = "passengers"

for col in ["route_id", "stop_id"]:
    if col in ml_data.columns:
        ml_data[col] = pd.to_numeric(ml_data[col], errors="coerce")

X = ml_data[feature_columns]
y = ml_data[target_column]

print("\n========================================")
print("       ML INPUT AND TARGET")
print("========================================")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nInput features:")
print(X.head())
print("\nTarget values:")
print(y.head())


# ============================================================
# 31. TRAIN / TEST SPLIT
# ============================================================

ml_data = ml_data.sort_values("service_date").reset_index(drop=True)
split_index = int(len(ml_data) * 0.80)

train_data = ml_data.iloc[:split_index]
test_data = ml_data.iloc[split_index:]

X_train_raw = train_data[feature_columns]
X_test_raw = test_data[feature_columns]
y_train = train_data[target_column]
y_test = test_data[target_column]

categorical_features = ["day_of_week", "day_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("\n========================================")
print("       TIME-BASED TRAIN / TEST SPLIT")
print("========================================")
print("Training records:", len(train_data))
print("Testing records :", len(test_data))
print("Training period :", train_data["service_date"].min(), "to", train_data["service_date"].max())
print("Testing period  :", test_data["service_date"].min(), "to", test_data["service_date"].max())
print("Training X:", X_train.shape)
print("Testing X :", X_test.shape)
print("Training y:", y_train.shape)
print("Testing y:", y_test.shape)


# ============================================================
# 32. MODEL TRAINING
# ============================================================

model = RandomForestRegressor(
    n_estimators=30,
    max_depth=16,
    max_samples=0.7,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\n========================================")
print("       MODEL TRAINING")
print("========================================")
print("Random Forest model trained successfully!")


# ============================================================
# 33. EVALUATION
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========================================")
print("       SAMPLE PREDICTIONS")
print("========================================")
for actual, predicted in zip(y_test.head(10), y_pred[:10]):
    print(f"Actual: {actual} passengers | Predicted: {predicted:.2f} passengers")

print("\n========================================")
print("       MODEL EVALUATION")
print("========================================")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 34. FEATURE IMPORTANCE
# ============================================================

feature_names = preprocessor.get_feature_names_out()
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

print("\n========================================")
print("       TOP 15 FEATURE IMPORTANCE")
print("========================================")
print(feature_importance.head(15).to_string(index=False))


# ============================================================
# 35. SAVE TRAINED MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "preprocessor": preprocessor,
    "feature_columns": feature_columns
}

joblib.dump(
    model_package,
    "models/demand_model.pkl"
)

print("\n========================================")
print("       MODEL SAVED")
print("========================================")
print("File: models/demand_model.pkl")


# ============================================================
# 36. DEMAND PREDICTION FUNCTION
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

    # Apply the exact same preprocessing used during model training.
    input_encoded = preprocessor.transform(
        input_data[feature_columns]
    )

    prediction = float(
        model.predict(input_encoded)[0]
    )

    # The synthetic holiday/festival uplift is already present in the
    # training data, so do NOT multiply the prediction again here.
    return max(0, prediction)


# ============================================================
# 37. DEMAND EXPLANATION
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

ml_mean_demand = float(
    ml_data["passengers"].mean()
)


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
            modified_values[feature] = ml_mean_demand
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
            "Demand is estimated using time, day type, route, stop "
            "and historical demand patterns."
        )

    return {
        "prediction": prediction,
        "festival_summary": festival_summary,
        "drivers": drivers
    }


# ============================================================
# 38. MODEL PERFORMANCE INFORMATION
# ============================================================

test_festival_dates = int(
    test_data.loc[
        test_data["is_festival"] == 1,
        "service_date"
    ].nunique()
)


# ============================================================
# 39. TEST DEMAND PREDICTION
# ============================================================

predicted_passengers = predict_demand(
    route_id=20578,
    stop_id=6780,
    hour=8,
    day_of_week="Thursday",
    day_type="Weekday",
    month=1,
    is_weekend=0,
    is_holiday=0,
    is_festival=0,
    is_peak_hour=1,
    is_night_demand=0,
    historical_demand=30
)

print("\\n========================================")
print("       DEMAND PREDICTION")
print("========================================")
print(f"Predicted passengers: {predicted_passengers:.0f}")


# ============================================================
# 40. CAPACITY PLANNING
# ============================================================

bus_capacity = 50
planned_buses = 1
planned_capacity = planned_buses * bus_capacity
capacity_shortage = max(0, predicted_passengers - planned_capacity)
additional_buses = math.ceil(capacity_shortage / bus_capacity)

if capacity_shortage > 0:
    recommendation = (
        f"Add {additional_buses} additional bus(es) to meet the predicted demand."
    )
else:
    recommendation = (
        "Existing bus capacity is sufficient. No additional bus required."
    )

print("\\n========================================")
print("       CAPACITY RECOMMENDATION")
print("========================================")
print(f"Predicted passengers : {predicted_passengers:.0f}")
print(f"Bus capacity         : {bus_capacity}")
print(f"Planned buses        : {planned_buses}")
print(f"Planned capacity     : {planned_capacity}")
print(f"Capacity shortage    : {capacity_shortage:.0f}")
print(f"Additional buses     : {additional_buses}")
print(f"Recommendation       : {recommendation}")
