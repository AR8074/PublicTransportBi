import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, date

import transport_calendar

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Chennai MTC Transport Intelligence",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Chennai MTC Transport Intelligence System")
st.write(
    "Enter your journey details to get route, timing, fare, "
    "delay, demand and bus availability."
)

# ============================================================
# LOAD GTFS DATA
# ============================================================

@st.cache_data
def load_gtfs():

    routes = pd.read_csv(
        "data/raw/routes.txt"
    )

    trips = pd.read_csv(
        "data/raw/trips.txt"
    )

    stops = pd.read_csv(
        "data/raw/stops.txt"
    )

    stop_times = pd.read_csv(
        "data/raw/stop_times.txt",
        usecols=[
            "trip_id",
            "arrival_time",
            "departure_time",
            "stop_id",
            "stop_sequence"
        ]
    )

    return routes, trips, stops, stop_times


routes, trips, stops, stop_times = load_gtfs()
# ============================================================
# LOAD ML FUNCTIONS
# ============================================================

import ml_predictor as ml


# ============================================================
# PREPARE STOP LIST
# ============================================================

stops = stops.dropna(
    subset=["stop_name"]
).copy()

stops["stop_id"] = stops["stop_id"].astype(int)

stops = stops.sort_values(
    "stop_name"
).drop_duplicates(
    subset=["stop_id"]
)


stop_dictionary = dict(
    zip(
        stops["stop_id"],
        stops["stop_name"]
    )
)


# ============================================================
# USER INPUTS
# ============================================================

st.header("🔎 Journey Search")

col1, col2 = st.columns(2)

with col1:

    selected_date = st.date_input(
        "📅 Date",
        value=date(2026, 10, 7),
        min_value=date(2026, 1, 1),
        max_value=date(2026, 12, 31)
    )


with col2:

    selected_time = st.time_input(
        "🕐 From Time",
        value=datetime.strptime(
            "08:00",
            "%H:%M"
        ).time()
    )


col3, col4 = st.columns(2)

# ============================================================
# FROM / TO LOCATION SELECTION
# ============================================================

stop_ids = stops["stop_id"].tolist()


# ============================================================
# FIND POSSIBLE DESTINATIONS FROM A FROM STOP
# ============================================================

@st.cache_data
def get_available_to_stops(from_stop):

    if from_stop is None:
        return []

    from_data = stop_times[
        stop_times["stop_id"] == from_stop
    ][
        ["trip_id", "stop_sequence"]
    ].copy()

    if from_data.empty:
        return []

    to_data = stop_times[
        stop_times["trip_id"].isin(
            from_data["trip_id"]
        )
    ][
        ["trip_id", "stop_id", "stop_sequence"]
    ].copy()

    to_data = to_data.merge(
        from_data,
        on="trip_id",
        suffixes=("_to", "_from")
    )

    # Destination must come AFTER From
    to_data = to_data[
        to_data["stop_sequence_to"]
        > to_data["stop_sequence_from"]
    ]

    return sorted(
        to_data["stop_id"]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# FIND POSSIBLE ORIGINS FOR A TO STOP
# ============================================================

@st.cache_data
def get_available_from_stops(to_stop):

    if to_stop is None:
        return []

    to_data = stop_times[
        stop_times["stop_id"] == to_stop
    ][
        ["trip_id", "stop_sequence"]
    ].copy()

    if to_data.empty:
        return []

    from_data = stop_times[
        stop_times["trip_id"].isin(
            to_data["trip_id"]
        )
    ][
        ["trip_id", "stop_id", "stop_sequence"]
    ].copy()

    from_data = from_data.merge(
        to_data,
        on="trip_id",
        suffixes=("_from", "_to")
    )

    # From must come BEFORE To
    from_data = from_data[
        from_data["stop_sequence_from"]
        < from_data["stop_sequence_to"]
    ]

    return sorted(
        from_data["stop_id"]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "from_stop" not in st.session_state:
    st.session_state.from_stop = None

if "to_stop" not in st.session_state:
    st.session_state.to_stop = None


# ============================================================
# WHEN FROM IS CHANGED
# ============================================================

def from_changed():

    selected_from = st.session_state.from_stop
    selected_to = st.session_state.to_stop

    if selected_from is None:
        return

    possible_to = get_available_to_stops(
        selected_from
    )

    # If current To is not possible,
    # clear it.
    if (
        selected_to is not None
        and selected_to not in possible_to
    ):
        st.session_state.to_stop = None


# ============================================================
# WHEN TO IS CHANGED
# ============================================================

def to_changed():

    selected_to = st.session_state.to_stop
    selected_from = st.session_state.from_stop

    if selected_to is None:
        return

    possible_from = get_available_from_stops(
        selected_to
    )

    # If current From is not possible,
    # clear it.
    if (
        selected_from is not None
        and selected_from not in possible_from
    ):
        st.session_state.from_stop = None


# ============================================================
# DETERMINE DROPDOWN OPTIONS
# ============================================================

selected_from = st.session_state.from_stop
selected_to = st.session_state.to_stop


# -------------------------
# FROM OPTIONS
# -------------------------

if selected_to is not None:

    from_options = get_available_from_stops(
        selected_to
    )

else:

    from_options = stop_ids


# -------------------------
# TO OPTIONS
# -------------------------

if selected_from is not None:

    to_options = get_available_to_stops(
        selected_from
    )

else:

    to_options = stop_ids


# ============================================================
# DISPLAY DROPDOWNS
# ============================================================

col3, col4 = st.columns(2)


with col3:

    from_stop = st.selectbox(
        "📍 From Address",
        from_options,
        index=None,
        placeholder="Select From Address",
        format_func=lambda x: stop_dictionary[x],
        key="from_stop",
        on_change=from_changed
    )


with col4:

    to_stop = st.selectbox(
        "📍 To Address",
        to_options,
        index=None,
        placeholder="Select To Address",
        format_func=lambda x: stop_dictionary[x],
        key="to_stop",
        on_change=to_changed
    )


# ============================================================
# NO POSSIBLE ROUTE MESSAGE
# ============================================================

if (
    st.session_state.from_stop is not None
    and len(to_options) == 0
):

    st.warning(
        "⚠️ No direct destination is available "
        "from the selected From address."
    )


if (
    st.session_state.to_stop is not None
    and len(from_options) == 0
):

    st.warning(
        "⚠️ No direct origin is available "
        "for the selected To address."
    )

search = st.button(
    "🔍 Search Journey",
    type="primary"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def time_to_seconds(value):

    if pd.isna(value):
        return None

    parts = str(value).split(":")

    if len(parts) != 3:
        return None

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def seconds_to_time(seconds):

    seconds = int(seconds)

    hours = (seconds // 3600) % 24
    minutes = (seconds % 3600) // 60

    return f"{hours:02d}:{minutes:02d}"


def calculate_distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * earth_radius
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# FIND DIRECT ROUTE
# ============================================================

def find_route(
    from_stop,
    to_stop,
    requested_seconds
):

    from_data = stop_times[
        stop_times["stop_id"] == from_stop
    ].copy()

    to_data = stop_times[
        stop_times["stop_id"] == to_stop
    ].copy()

    if from_data.empty or to_data.empty:
        return None

    journeys = from_data.merge(
        to_data,
        on="trip_id",
        suffixes=(
            "_from",
            "_to"
        )
    )

    journeys = journeys[
        journeys["stop_sequence_from"]
        <
        journeys["stop_sequence_to"]
    ].copy()

    if journeys.empty:
        return None

    journeys["departure_seconds"] = (
        journeys["departure_time_from"]
        .apply(time_to_seconds)
    )

    journeys = journeys.dropna(
        subset=["departure_seconds"]
    )

    if journeys.empty:
        return None

    # Select the trip closest to requested time
    journeys["time_difference"] = (
        journeys["departure_seconds"]
        - requested_seconds
    ).abs()

    selected = journeys.sort_values(
        "time_difference"
    ).iloc[0]

    trip_id = selected["trip_id"]

    trip_info = trips[
        trips["trip_id"] == trip_id
    ]

    if trip_info.empty:
        return None

    route_id = trip_info.iloc[0]["route_id"]

    route_info = routes[
        routes["route_id"] == route_id
    ]

    if route_info.empty:
        return None

    route_info = route_info.iloc[0]

    return {
        "trip_id": trip_id,
        "route_id": route_id,
        "route_short_name":
            route_info.get(
                "route_short_name",
                "N/A"
            ),
        "route_long_name":
            route_info.get(
                "route_long_name",
                "N/A"
            ),
        "departure_time":
            selected["departure_time_from"],
        "arrival_time":
            selected["arrival_time_to"],
        "from_sequence":
            selected["stop_sequence_from"],
        "to_sequence":
            selected["stop_sequence_to"]
    }


# ============================================================
# FARE ESTIMATION
# ============================================================
def estimate_fare(from_stop, to_stop):

    from_info = stops[
        stops["stop_id"] == from_stop
    ].iloc[0]

    to_info = stops[
        stops["stop_id"] == to_stop
    ].iloc[0]

    # Approximate geographic distance
    distance = calculate_distance_km(
        from_info["stop_lat"],
        from_info["stop_lon"],
        to_info["stop_lat"],
        to_info["stop_lon"]
    )

    # MTC official ordinary adult fare
    ordinary_fares = {
        1: 5,
        2: 6,
        3: 7,
        4: 8,
        5: 9,
        6: 10,
        7: 11,
        8: 12,
        9: 13,
        10: 14,
        11: 15,
        12: 15,
        13: 16,
        14: 16,
        15: 17,
        16: 17,
        17: 18,
        18: 18,
        19: 19,
        20: 19,
        21: 20,
        22: 20,
        23: 21,
        24: 21,
        25: 22,
        26: 22,
        27: 23,
        28: 23,
        29: 24,
        30: 24
    }

    # Approximate stages
    stages = max(
        1,
        math.ceil(distance / 2)
    )

    stages = min(stages, 30)

    fare = ordinary_fares[stages]

    return distance, stages, fare

# ============================================================
# HOLIDAY / FESTIVAL
# ============================================================

CALENDAR = transport_calendar.TransportCalendar(
    transport_calendar.load_calendar()
)


# ============================================================
# PREBUILT LOOKUP TABLES
# ============================================================
# The previous implementation copied the whole 77 MB feature table
# on every lookup. These indexes are built once and answer the same
# questions without touching a per-request copy.

@st.cache_resource
def build_history_index():

    frame = ml.ml_data.copy()

    frame["route_key"] = frame["route_id"].astype(str)
    frame["stop_key"] = frame["stop_id"].astype(str)

    series = {}

    for key, group in frame.groupby(
        ["route_key", "stop_key", "hour"]
    ):
        ordered = group.sort_values("service_date")
        series[key] = {
            "dates": ordered["service_date"].to_numpy(),
            "values": ordered["passengers"].to_numpy(dtype=float),
        }

    return {
        "series": series,
        "fallback": float(frame["passengers"].mean()),
    }


ML_HISTORY = build_history_index()


@st.cache_resource
def build_operational_index():

    frame = ml.ml_data.copy()

    frame["route_key"] = frame["route_id"].astype(str)
    frame["stop_key"] = frame["stop_id"].astype(str)

    exact = (
        frame.groupby(["service_date", "route_key", "stop_key", "hour"])[
            "delay_minutes"
        ]
        .mean()
    )

    by_route_hour = (
        frame.groupby(["route_key", "hour"])["delay_minutes"].mean()
    )

    return {
        "exact": exact,
        "by_route_hour": by_route_hour,
        "overall": float(frame["delay_minutes"].mean()),
    }


OPERATIONAL_INDEX = build_operational_index()


@st.cache_resource
def get_modal_capacity():

    capacity = pd.to_numeric(
        ml.ml_data["capacity"],
        errors="coerce",
    ).dropna()

    if capacity.empty:
        return 50

    return int(capacity.mode().iloc[0])


ML_MODAL_CAPACITY = get_modal_capacity()


def get_calendar_information(
    selected_date
):

    date_value = pd.Timestamp(
        selected_date
    ).normalize()

    is_holiday = CALENDAR.is_holiday(
        date_value
    )

    is_festival = CALENDAR.is_festival(
        date_value
    )

    return is_holiday, is_festival


def get_event_name(
    selected_date
):

    return CALENDAR.event_name(
        pd.Timestamp(selected_date).normalize()
    )


# ============================================================
# HISTORICAL DEMAND
# ============================================================

@st.cache_data
def get_historical_demand(route_id, stop_id, hour, selected_date):
    target = pd.Timestamp(selected_date).normalize()

    history = ML_HISTORY["series"].get(
        (str(route_id), str(stop_id), int(hour))
    )

    if history is None:
        return ML_HISTORY["fallback"]

    earlier = history["dates"] < target
    if not earlier.any():
        return ML_HISTORY["fallback"]

    return float(history["values"][earlier][-7:].mean())


# ============================================================
# MAIN SEARCH
# ============================================================

if search:

    if to_stop is None:

        st.error(
            "No direct destination is available "
            "from the selected From stop."
        )

        st.stop()


    if from_stop == to_stop:

        st.error(
            "From and To stops must be different."
        )

        st.stop()


    requested_seconds = (
        selected_time.hour * 3600
        + selected_time.minute * 60
    )


    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    selected_day = pd.Timestamp(
        selected_date
    ).day_name()

    is_weekend = int(
        selected_day
        in ["Saturday", "Sunday"]
    )

    day_type = (
        "Weekend"
        if is_weekend
        else "Weekday"
    )

    month = selected_date.month

    hour = selected_time.hour


    is_peak_hour = int(
        (
            7 <= hour <= 9
        )
        or
        (
            17 <= hour <= 19
        )
    )

    is_night_demand = int(
        (
            selected_day == "Friday"
            and hour >= 19
        )
        or
        (
            selected_day == "Sunday"
            and hour >= 18
        )
    )


    # --------------------------------------------------------
    # HOLIDAY / FESTIVAL
    # --------------------------------------------------------

    is_holiday, is_festival = (
        get_calendar_information(
            selected_date
        )
    )


    # --------------------------------------------------------
    # FIND ROUTE
    # --------------------------------------------------------

    journey = find_route(
        from_stop,
        to_stop,
        requested_seconds
    )


    if journey is None:

        st.error(
            "No direct MTC route was found "
            "between these two stops."
        )

        st.stop()


    route_id = journey["route_id"]


    # --------------------------------------------------------
    # FARE
    # --------------------------------------------------------

    distance, stages, fare = (
        estimate_fare(
            from_stop,
            to_stop
        )
    )


    # --------------------------------------------------------
    # DELAY INFORMATION
    # --------------------------------------------------------

    delay_key = (
        pd.Timestamp(selected_date).normalize().to_datetime64(),
        str(route_id),
        str(from_stop),
        int(hour),
    )

    exact_key = (
        pd.Timestamp(selected_date).normalize(),
        str(route_id),
        str(from_stop),
        int(hour),
    )

    if exact_key in OPERATIONAL_INDEX["exact"].index:
        delay = float(
            OPERATIONAL_INDEX["exact"].loc[exact_key]
        )

    elif (str(route_id), int(hour)) in OPERATIONAL_INDEX[
        "by_route_hour"
    ].index:
        delay = float(
            OPERATIONAL_INDEX["by_route_hour"].loc[
                (str(route_id), int(hour))
            ]
        )

    else:
        delay = float(OPERATIONAL_INDEX["overall"])


    on_time = (
        "On Time"
        if delay <= 5
        else "Delayed"
    )


    # --------------------------------------------------------
    # HISTORICAL DEMAND
    # --------------------------------------------------------

    historical_demand = (
        get_historical_demand(
            route_id,
            from_stop,
            hour,
            selected_date
        )
    )


    # --------------------------------------------------------
    # ML DEMAND PREDICTION
    # --------------------------------------------------------

    event_name = get_event_name(selected_date)

    explanation = ml.explain_demand(
        route_id=int(route_id),
        stop_id=int(from_stop),
        hour=hour,
        day_of_week=selected_day,
        day_type=day_type,
        month=month,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        is_festival=is_festival,
        is_peak_hour=is_peak_hour,
        is_night_demand=is_night_demand,
        historical_demand=historical_demand,
        event_name=event_name,
    )
    predicted_passengers = max(
        0,
        round(explanation["prediction"])
    )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    # Taken from the fleet rather than hardcoded, so a festival peak
    # that genuinely needs a second bus is visible.
    bus_capacity = ML_MODAL_CAPACITY

    required_buses = max(
        1,
        math.ceil(
            predicted_passengers
            / bus_capacity
        )
    )

    total_capacity = (
        required_buses
        * bus_capacity
    )

    available_capacity = max(
        0,
        total_capacity
        - predicted_passengers
    )

    additional_buses = max(
        0,
        required_buses - 1
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    st.divider()

    st.header("🚌 Journey Result")


    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    st.subheader("🛣️ Route Information")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Route",
        str(
            journey["route_short_name"]
        )
    )

    r2.metric(
        "From",
        stop_dictionary[from_stop]
    )

    r3.metric(
        "To",
        stop_dictionary[to_stop]
    )

    st.write(
        f"**Route:** {journey['route_long_name']}"
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    st.subheader("⏱️ Schedule")

    departure_seconds = time_to_seconds(
        journey["departure_time"]
    )

    arrival_seconds = time_to_seconds(
        journey["arrival_time"]
    )

    journey_minutes = (
        arrival_seconds
        - departure_seconds
    ) / 60

    # Expected times based on predicted delay
    expected_departure_seconds = (
        departure_seconds
        + delay * 60
    )

    expected_arrival_seconds = (
        arrival_seconds
        + delay * 60
    )

    def seconds_to_time(total_seconds):

        total_seconds = int(total_seconds)

        hours = (total_seconds // 3600) % 24
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    expected_departure = seconds_to_time(
        expected_departure_seconds
    )

    expected_arrival = seconds_to_time(
        expected_arrival_seconds
    )

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Scheduled Departure",
        str(journey["departure_time"])
    )

    t2.metric(
        "Expected Departure",
        expected_departure
    )

    t3.metric(
        "Scheduled Arrival",
        str(journey["arrival_time"])
    )

    t4.metric(
        "Expected Arrival",
        expected_arrival
    )

    st.info(
        f"🕐 Scheduled Journey Time: "
        f"{journey_minutes:.0f} min  |  "
        f"Expected Delay: {delay:.1f} min"
    )

    # --------------------------------------------------------
    # DATE INFORMATION
    # --------------------------------------------------------
    st.subheader("📅 Date Information")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Day", selected_day)
    d2.metric("Day Type", day_type)
    d3.metric("Holiday", "Yes" if is_holiday else "No")
    d4.metric("Festival", "Yes" if is_festival else "No")

    # --------------------------------------------------------
    # FARE
    # --------------------------------------------------------

    st.subheader("💰 Fare")

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "Estimated Distance",
        f"{distance:.1f} km"
    )

    f2.metric(
        "Fare Stages",
        stages
    )

    f3.metric(
        "Ordinary Fare",
        f"₹{fare}"
    )

    st.caption(
        "Fare is an approximate ordinary-service fare "
        "based on MTC's stage-wise fare structure. "
        "Actual fare can differ by service type."
    )

    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    st.subheader("🚦 Service Status")

    s1, s2 = st.columns(2)

    s1.metric(
        "Expected Delay",
        f"{delay:.1f} min"
    )

    s2.metric(
        "Status",
        on_time
    )

    # --------------------------------------------------------
    # DEMAND
    # --------------------------------------------------------

    st.subheader("📊 Demand Prediction")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Predicted Passengers",
        predicted_passengers
    )

    p2.metric(
        "Bus Capacity",
        bus_capacity
    )

    p3.metric(
        "Required Buses",
        required_buses
    )

    p4.metric(
        "Available Capacity",
        available_capacity
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    if additional_buses > 0:

        st.warning(
            f"⚠️ High demand expected. "
            f"Consider {additional_buses} additional bus(es)."
        )

    else:

        st.success(
            "✅ Existing bus capacity is sufficient."
        )


    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    st.subheader("🧠 Why This Number")

    if event_name != "Regular day":

        st.info(
            f"📅 **{event_name}**"
        )

    st.write(explanation["festival_summary"])

    if explanation["drivers"]:

        driver_frame = pd.DataFrame(
            explanation["drivers"]
        )

        driver_frame["feature"] = driver_frame["feature"].map(
            lambda name: ml.FEATURE_LABELS.get(name, name)
        )

        st.dataframe(
            driver_frame[
                ["feature", "with", "without", "delta"]
            ].round(2),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Each row re-runs the model with one driver switched "
            "off. The delta is how many passengers that driver "
            "is responsible for."
        )

    festival_bar = ml.feature_importance

    if festival_bar is not None:

        with st.expander("Model-wide feature importance"):

            st.dataframe(
                festival_bar.head(12).round(4),
                hide_index=True,
                use_container_width=True,
            )
    st.caption(
    f"Model accuracy: MAE {ml.mae:.2f} · "
    f"R² {ml.r2:.3f}"
)