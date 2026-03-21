import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime
import os


st.set_page_config(page_title="Southern Barnegat Bay Predictions", page_icon="⛅", layout="centered")

PRIMARY_BG = "#486e8d"
PANEL_BG = "#4d6070"
ACCENT = "#FFC94A"
TEXT_LIGHT = "#F5F5F5"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {PRIMARY_BG};
        color: {TEXT_LIGHT};
    }}
    .stSidebar {{
        background-color: {PANEL_BG} !important;
    }}
    <style>
    [data-testid="stDecoration"] {{
        background-color: {PANEL_BG};
        background-image: none;
    }}
    .stButton>button {{
        background-color: {ACCENT};
        color: #333333;
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

#Load models and scaler
dir = os.path.dirname(__file__)
model_dir = os.path.join(dir, "models")

scaler_x = load(os.path.join(model_dir, "scaler_x.joblib"))
reg_speed = load(os.path.join(model_dir, "wind_speed_linear.joblib"))
reg_gust = load(os.path.join(model_dir, "wind_gust_linear.joblib"))

direction_map = {'N': 0,'NNE': 1,'NE': 2,'ENE': 3,'E': 4,'ESE': 5,'SE': 6,'SSE': 7,'S': 8,'SSW': 9,'SW': 10,'WSW': 11,'W': 12,'WNW': 13,'NW': 14,'NNW': 15}
allowed_dirs_deg = np.array([0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5])
compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]


#Get variables
st.sidebar.header("Input data")

st.sidebar.markdown("**Stafford Weather Station**")
data_main_air_temp = st.sidebar.number_input(
    "Mainland air temperature (C or F)", value=25.0, step=0.1
)
data_humidity_per = st.sidebar.number_input(
    "Humidity (%)", value=100.0, min_value=0.0, max_value=100.0, step=0.1
)
data_wind_direction = st.sidebar.selectbox(
    "Wind direction (Cardinal)", list(direction_map.keys())
)
data_wind_speed = st.sidebar.number_input(
    "Wind speed (mph)", value=5.0, min_value=0.0, step=0.1
)
data_gusting = st.sidebar.number_input(
    "Gusting Wind Speeds (max wind speed, mph)", value=10.0, min_value=0.0, step=0.1
)
data_pressure = st.sidebar.number_input(
    "Atmospheric pressure (IN)", value=29.95, step=0.01
)
data_rainfall = st.sidebar.number_input(
    "Precipitation (inches)", value=0.0, min_value=0.0, step=0.01
)

st.sidebar.markdown("**Nearby water / NJDEP MB_01 Buoy**")
data_bay_temp = st.sidebar.number_input(
    "Bay temperature (C or F)", value=22.0, step=0.1
)
data_salinity = st.sidebar.number_input(
    "Salinity (ppt)", value=30.0, step=0.1
)

st.sidebar.markdown("**SCYC Weather Station**")
data_lbi_temp = st.sidebar.number_input(
    "LBI air temperature (C or F)", value=24.0, step=0.1
)

st.sidebar.markdown("**NDBC Station 44091**")
data_ocean_temp = st.sidebar.number_input(
    "Ocean temperature (C or F)", value=20.0, step=0.1
)

st.sidebar.markdown("**Temperature units**")
air_celsius = st.sidebar.radio(
    "Are air temperatures in Celsius?", ["Yes", "No"], index=0
)
water_celsius = st.sidebar.radio(
    "Are water temperatures in Celsius?", ["Yes", "No"], index=0
)

if st.button("Run prediction", width="stretch"):
    # Copy your conversion logic
    if air_celsius == "No":
        data_main_air_temp = round((data_main_air_temp - 32) * 5.0 / 9.0, 1)
        data_lbi_temp = round((data_lbi_temp - 32) * 5.0 / 9.0, 1)
    if water_celsius == "No":
        data_ocean_temp = round((data_ocean_temp - 32) * 5.0 / 9.0, 1)
        data_bay_temp = round((data_bay_temp - 32) * 5.0 / 9.0, 1)

    #map direction
    data_wind_direction_idx = direction_map[data_wind_direction]
    direction_deg = allowed_dirs_deg[data_wind_direction_idx]
    direction_label = compass[data_wind_direction_idx]
    print("Variables retrieved")

    #round all of the columns
    data_main_air_temp = round(data_main_air_temp, 1)
    data_humidity_per = round(data_humidity_per, 1)
    data_wind_speed = round(data_wind_speed, 1)
    data_gusting = round(data_gusting, 1)
    data_pressure = round(data_pressure, 2)
    data_rainfall = round(data_rainfall, 2)
    data_bay_temp = round(data_bay_temp, 2)
    data_salinity = round(data_salinity, 2)
    data_lbi_temp = round(data_lbi_temp, 1)
    data_ocean_temp = round(data_ocean_temp, 1)
    data_upwelling_flag = 0

    #determines if its a onshore breeze and adds a new column for it
    onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
    if data_wind_direction in onshore_degrees:
        data_onshore_flag = 1
    else:
        data_onshore_flag = 0 

    #save all input data into one Numpy array
    dataset = np.array([
            data_main_air_temp,
            data_humidity_per,
            data_wind_direction_idx,
            #data_wind_speed,
            data_gusting,
            data_pressure,
            data_rainfall,
            data_bay_temp,
            data_salinity,
            data_lbi_temp,
            data_ocean_temp,
            data_onshore_flag,
            data_upwelling_flag,
        ]).reshape(1, -1)

    scaledx = scaler_x.transform(dataset)

    #Predict wind speed
    speed_pred_lr = reg_speed.predict(scaledx)
    speed_pred_lr = float(speed_pred_lr.squeeze())
    speed_pred_lr = np.maximum(speed_pred_lr, 0.0)
    speed_pred_lr = np.round(speed_pred_lr, 1)

    #saves all input data into one Numpy array
    dataset = np.array([
        data_main_air_temp,
        data_humidity_per,
        data_wind_direction_idx,
        data_wind_speed,
        #data_gusting,
        data_pressure,
        data_rainfall,
        data_bay_temp,
        data_salinity,
        data_lbi_temp,
        data_ocean_temp,
        data_onshore_flag,
        data_upwelling_flag,
    ]).reshape(1, -1)

    scaledx = scaler_x.transform(dataset)

    #Predict wind gusting speed
    gust_pred_lr = reg_gust.predict(scaledx)
    gust_pred_lr = float(gust_pred_lr.squeeze())
    gust_pred_lr = np.maximum(gust_pred_lr, 0.0)
    gust_pred_lr = np.round(gust_pred_lr, 1)

    direction_pred_idx = data_wind_direction_idx
    #Convert to degrees
    direction_pred_deg = allowed_dirs_deg[direction_pred_idx]
    #Convert to cardinal directions
    direction_pred_label = data_wind_direction
    direction_pred_label = compass[direction_pred_idx]

    #Predict onshore flag
    if data_wind_direction in onshore_degrees:
        onshore_pred_flag = "Yes"
    else:
        onshore_pred_flag = "No"

    st.subheader("Prediction")
    st.write(f"**Predicted wind speed:** {speed_pred_lr} mph")
    st.write(f"**Predicted wind gust:** {gust_pred_lr} mph")
    st.write(f"**Predicted direction:** {direction_pred_deg}° ({direction_pred_label})")
    st.write(f"**Is this a Onshore Breeze?:** {onshore_pred_flag}")

    if (speed_pred_lr >= 10) or (gust_pred_lr >= 10):
        st.warning("Be careful on the water. High wind speeds predicted.")