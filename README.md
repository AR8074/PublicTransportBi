# Chennai MTC Transport Intelligence System

An interactive **Public Transport Data Warehousing and Business Intelligence system** for Chennai Metropolitan Transport Corporation (MTC).

The system combines real Chennai MTC GTFS data with simulated operational data, SQL-based analytics, Machine Learning, and Streamlit to provide transport insights, passenger demand prediction, and capacity planning recommendations.

## 🚍 Live Application

🔗 **Streamlit App:**  
https://publictransportbi-zcjib26vwh4qzen3gpn6pg.streamlit.app/

---

## 📌 Project Overview

Public transportation systems generate large amounts of data related to routes, stops, schedules, passengers, delays, and vehicle capacity.

This project develops a transport intelligence system that helps analyze this information and provide useful decision-support insights.

The user provides:

- Date
- Time
- From Stop
- To Stop

The application then provides relevant transport information such as:

- Route information
- Scheduled departure and arrival
- Expected arrival time
- Delay estimation
- Fare information
- Passenger demand prediction
- Bus capacity analysis
- Holiday and festival information
- Capacity recommendations

---

## 🎯 Objectives

- Analyze real Chennai MTC transportation network data.
- Integrate routes, stops, trips, and schedules.
- Prepare operational data for analytics.
- Perform SQL-based transport analytics.
- Identify important transport KPIs.
- Predict passenger demand using Machine Learning.
- Analyze bus capacity requirements.
- Provide an interactive decision-support application.
- Demonstrate an end-to-end Data Science and Business Intelligence workflow.

---

## 🏗️ System Architecture

```text
Real Chennai MTC GTFS Data
          ↓
      Python / Pandas
          ↓
Data Cleaning + Operational Data
          ↓
        MySQL
          ↓
      SQL Analytics
          ↓
Holiday / Festival Calendar
          ↓
 Machine Learning Model
          ↓
   Demand Prediction
          ↓
      Streamlit App
          ↓
Capacity Planning & Recommendations
