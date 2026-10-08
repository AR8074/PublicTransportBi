import os

import pandas as pd


DATABASE_CONFIG = {
    "host": os.getenv("MTC_DB_HOST", "127.0.0.1"),
    "user": os.getenv("MTC_DB_USER", "root"),
    "password": os.getenv("MTC_DB_PASSWORD", ""),
    "database": os.getenv("MTC_DB_NAME", "PublicTransportBI"),
}


FEATURES_TABLE = "ml_transport_features"
CALENDAR_TABLE = "holiday_calendar"


FALLBACK_CALENDAR = [
    ("2026-01-01", "New Year's Day", None, 1, 0),
    ("2026-01-14", None, "Pongal / Sankranti Festival Period", 0, 1),
    ("2026-01-15", "Pongal", "Pongal / Sankranti Festival Period", 1, 1),
    ("2026-01-16", "Thiruvalluvar Day", "Pongal / Sankranti Festival Period", 1, 1),
    ("2026-01-17", "Uzhavar Thirunal", "Pongal / Sankranti Festival Period", 1, 1),
    ("2026-01-18", None, "Pongal / Sankranti Festival Period", 0, 1),
    ("2026-01-26", "Republic Day", None, 1, 0),
    ("2026-02-01", "Thai Poosam", "Thai Poosam", 1, 1),
    ("2026-03-19", "Telugu New Year's Day", None, 1, 0),
    ("2026-03-21", "Ramzan (Idu'l Fitr)", "Ramzan", 1, 1),
    ("2026-03-31", "Mahaveer Jayanthi", "Mahaveer Jayanthi", 1, 1),
    ("2026-04-03", "Good Friday", "Good Friday", 1, 1),
    ("2026-04-14", "Tamil New Year / Dr. B.R. Ambedkar Birthday", "Tamil New Year", 1, 1),
    ("2026-05-01", "May Day", None, 1, 0),
    ("2026-05-28", "Bakrid (Idul Azha)", "Bakrid", 1, 1),
    ("2026-06-26", "Muharram", "Muharram", 1, 1),
    ("2026-08-15", "Independence Day", None, 1, 0),
    ("2026-08-26", "Milad-un-Nabi", "Milad-un-Nabi", 1, 1),
    ("2026-09-04", "Krishna Jayanthi", "Krishna Jayanthi", 1, 1),
    ("2026-09-14", "Vinayakar Chathurthi", "Vinayakar Chathurthi", 1, 1),
    ("2026-10-02", "Gandhi Jayanthi", None, 1, 0),
    ("2026-10-17", None, "Dasara / Dussehra Festival Period", 0, 1),
    ("2026-10-18", None, "Dasara / Dussehra Festival Period", 0, 1),
    ("2026-10-19", "Ayutha Pooja", "Dasara / Dussehra Festival Period", 1, 1),
    ("2026-10-20", "Vijaya Dasami", "Dasara / Dussehra Festival Period", 1, 1),
    ("2026-10-21", None, "Dasara / Dussehra Festival Period", 0, 1),
    ("2026-11-08", "Deepavali", "Deepavali", 1, 1),
    ("2026-12-25", "Christmas", "Christmas", 1, 1),
]


DEFAULT_FESTIVAL_MULTIPLIER = 1.40
DEFAULT_HOLIDAY_MULTIPLIER = 1.15

FESTIVAL_EVENT_MULTIPLIERS = {
    "Pongal / Sankranti Festival Period": 1.50,
    "Dasara / Dussehra Festival Period": 1.45,
    "Deepavali": 1.45,
    "Tamil New Year": 1.40,
    "Christmas": 1.35,
}

HOLIDAY_EVENT_MULTIPLIERS = {
    "New Year's Day": 1.25,
    "Independence Day": 1.20,
    "Gandhi Jayanthi": 1.15,
}


CALENDAR_COLUMNS = [
    "holiday_date",
    "holiday_name",
    "festival_name",
    "is_holiday",
    "is_festival",
    "event_name",
    "festival_multiplier",
    "holiday_multiplier",
    "demand_multiplier",
]


def _event_multiplier(row):
    if row["is_festival"] == 1:
        festival_name = row["festival_name"]
        if pd.isna(festival_name):
            return DEFAULT_FESTIVAL_MULTIPLIER
        return FESTIVAL_EVENT_MULTIPLIERS.get(
            festival_name,
            DEFAULT_FESTIVAL_MULTIPLIER,
        )

    if row["is_holiday"] == 1:
        holiday_name = row["holiday_name"]
        if pd.isna(holiday_name):
            return DEFAULT_HOLIDAY_MULTIPLIER
        return HOLIDAY_EVENT_MULTIPLIERS.get(
            holiday_name,
            DEFAULT_HOLIDAY_MULTIPLIER,
        )

    return 1.0


def _label_event(row):
    if row["is_festival"] == 1:
        festival_name = row["festival_name"]
        if not pd.isna(festival_name):
            return festival_name
        return "Festival"

    if row["is_holiday"] == 1:
        holiday_name = row["holiday_name"]
        if not pd.isna(holiday_name):
            return holiday_name
        return "Public holiday"

    return "Regular day"


def _normalise(raw_calendar):
    calendar = raw_calendar.copy()

    calendar["holiday_date"] = pd.to_datetime(
        calendar["holiday_date"]
    ).dt.normalize().astype("datetime64[s]")

    calendar["is_holiday"] = calendar["is_holiday"].astype(int)
    calendar["is_festival"] = calendar["is_festival"].astype(int)

    calendar["event_name"] = calendar.apply(
        _label_event,
        axis=1,
    )

    calendar["demand_multiplier"] = calendar.apply(
        _event_multiplier,
        axis=1,
    )

    calendar["festival_multiplier"] = calendar["demand_multiplier"].where(
        calendar["is_festival"] == 1,
        1.0,
    )

    calendar["holiday_multiplier"] = calendar["demand_multiplier"].where(
        calendar["is_holiday"] == 1,
        1.0,
    )

    calendar = calendar.sort_values("holiday_date").reset_index(
        drop=True
    )

    return calendar[CALENDAR_COLUMNS]


def fallback_calendar():
    calendar = pd.DataFrame(
        FALLBACK_CALENDAR,
        columns=[
            "holiday_date",
            "holiday_name",
            "festival_name",
            "is_holiday",
            "is_festival",
        ],
    )

    return _normalise(calendar)


def load_calendar(connection=None):
    owns_connection = connection is None

    if owns_connection:
        try:
            import mysql.connector

            connection = mysql.connector.connect(**DATABASE_CONFIG)
        except Exception:
            return fallback_calendar()

    try:
        calendar = pd.read_sql(
            f"SELECT holiday_date, holiday_name, festival_name, "
            f"is_holiday, is_festival FROM {CALENDAR_TABLE}",
            connection,
        )
    except Exception:
        return fallback_calendar()
    finally:
        if owns_connection and connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    if calendar.empty:
        return fallback_calendar()

    return _normalise(calendar)


class TransportCalendar:
    def __init__(self, calendar):

        self.calendar = calendar

        self._by_date = {
            row["holiday_date"]: row
            for _, row in calendar.iterrows()
        }

        self._multiplier_by_date = {
            date: float(row["demand_multiplier"])
            for date, row in self._by_date.items()
        }

        self.dates = set(self._multiplier_by_date)

    def is_festival(self, date):
        row = self._by_date.get(pd.Timestamp(date).normalize())
        if row is None:
            return 0
        return int(row["is_festival"])

    def is_holiday(self, date):
        row = self._by_date.get(pd.Timestamp(date).normalize())
        if row is None:
            return 0
        return int(row["is_holiday"])

    def event_name(self, date):
        row = self._by_date.get(pd.Timestamp(date).normalize())
        if row is None:
            return "Regular day"
        return row["event_name"]

    def multiplier(self, date):
        date = pd.Timestamp(date).normalize()
        if date not in self._multiplier_by_date:
            return 1.0
        return self._multiplier_by_date[date]

    def covers(self, date):
        return pd.Timestamp(date).normalize() in self._multiplier_by_date

    def summary(self):
        return {
            "events": len(self.calendar),
            "holidays": int(self.calendar["is_holiday"].sum()),
            "festivals": int(self.calendar["is_festival"].sum()),
            "first": self.calendar["holiday_date"].min(),
            "last": self.calendar["holiday_date"].max(),
        }


def apply_calendar_flags(data, calendar, date_column="service_date"):
    if isinstance(calendar, TransportCalendar):
        calendar = calendar.calendar

    data[date_column] = pd.to_datetime(data[date_column])

    normalised = data[date_column].dt.normalize().astype("datetime64[s]")

    flags = calendar.set_index("holiday_date")

    data["is_festival"] = (
        normalised.map(flags["is_festival"]).fillna(0).astype(int)
    )

    data["is_holiday"] = (
        normalised.map(flags["is_holiday"]).fillna(0).astype(int)
    )

    data["calendar_multiplier"] = (
        normalised.map(flags["demand_multiplier"]).fillna(1.0)
    )

    data["event_name"] = normalised.map(
        flags["event_name"]
    ).fillna("Regular day")

    return data