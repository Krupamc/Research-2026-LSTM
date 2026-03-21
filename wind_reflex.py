import reflex as rx
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime
import os
#from rxconfig import config

# Load models
scaler_x = load("models/scaler_x.joblib")
reg_speed = load("models/wind_speed_linear.joblib")
reg_gust = load("models/wind_gust_linear.joblib")

# Direction maps
direction_map = {
    "N": 0, "NNE": 1, "NE": 2, "ENE": 3,
    "E": 4, "ESE": 5, "SE": 6, "SSE": 7,
    "S": 8, "SSW": 9, "SW": 10, "WSW": 11,
    "W": 12, "WNW": 13, "NW": 14, "NNW": 15,
}
allowed_dirs_deg = np.array(
    [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
     180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
)
compass = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


class State(rx.State):
    # Inputs
    main_air_temp: float = 20.0
    humidity: float = 50.0
    wind_dir: str = "N"
    wind_speed: float = 5.0
    gusting: float = 10.0
    pressure: float = 30.00
    rainfall: float = 0.00
    bay_temp: float = 20.0
    salinity: float = 30.0
    lbi_temp: float = 20.0
    ocean_temp: float = 20.0
    air_in_c: bool = True
    water_in_c: bool = True

    # Outputs
    pred_speed: float = 0.0
    pred_gust: float = 0.0
    pred_dir_deg: float = 0.0
    pred_dir_label: str = "N"
    pred_onshore_flag: int = 0
    warning: str = ""

    #prediction logic

    def predict(self):
        data_main_air_temp = self.main_air_temp
        data_humidity_per = self.humidity

        d = self.wind_dir.strip().upper()
        if d not in direction_map:
            self.warning = "Invalid wind direction."
            return

        idx = direction_map[d]
        direction_deg = allowed_dirs_deg[idx]
        direction_label = compass[idx]

        data_wind_speed = self.wind_speed
        data_gusting = self.gusting
        data_pressure = self.pressure
        data_rainfall = self.rainfall
        data_bay_temp = self.bay_temp
        data_salinity = self.salinity
        data_lbi_temp = self.lbi_temp
        data_ocean_temp = self.ocean_temp

        # Convert units
        if not self.air_in_c:
            data_main_air_temp = round((data_main_air_temp - 32) * 5.0 / 9.0, 1)
            data_lbi_temp = round((data_lbi_temp - 32) * 5.0 / 9.0, 1)
        if not self.water_in_c:
            data_ocean_temp = round((data_ocean_temp - 32) * 5.0 / 9.0, 1)
            data_bay_temp = round((data_bay_temp - 32) * 5.0 / 9.0, 1)

        # Onshore / upwelling flags
        onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
        data_onshore_flag = 1 if idx in onshore_degrees else 0
        data_upwelling_flag = 0

        # SPEED model dataset
        dataset_speed = np.array([
            data_main_air_temp,
            data_humidity_per,
            idx,
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

        scaledx_speed = scaler_x.transform(dataset_speed)
        speed_pred_lr = reg_speed.predict(scaledx_speed)
        speed_pred_lr = np.maximum(speed_pred_lr, 0.0)
        speed_pred_lr = np.round(speed_pred_lr, 1)

        # GUST model dataset
        dataset_gust = np.array([
            data_main_air_temp,
            data_humidity_per,
            idx,
            data_wind_speed,
            data_pressure,
            data_rainfall,
            data_bay_temp,
            data_salinity,
            data_lbi_temp,
            data_ocean_temp,
            data_onshore_flag,
            data_upwelling_flag,
        ]).reshape(1, -1)

        scaledx_gust = scaler_x.transform(dataset_gust)
        gust_pred_lr = reg_gust.predict(scaledx_gust)
        gust_pred_lr = np.maximum(gust_pred_lr, 0.0)
        gust_pred_lr = np.round(gust_pred_lr, 1)

        # Onshore from direction
        onshore_pred_flag = 1 if idx in onshore_degrees else 0

        # Save outputs to state
        self.pred_speed = float(speed_pred_lr[0])
        self.pred_gust = float(gust_pred_lr[0])
        self.pred_dir_deg = float(direction_deg)
        self.pred_dir_label = direction_label
        self.pred_onshore_flag = int(onshore_pred_flag)

        if self.pred_speed >= 10 or self.pred_gust >= 10:
            self.warning = "Be careful on the water. High wind speeds predicted."
        else:
            self.warning = ""

        # Save record to CSV
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "Mainland Air Temp": data_main_air_temp,
            "Humidity (%)": data_humidity_per,
            "Direction Degree": direction_deg,
            "Direction label": direction_label,
            "Gusting": data_gusting,
            "Atmospheric Pressure (IN)": data_pressure,
            "Precipitation Rate": data_rainfall,
            "Bay Temp": data_bay_temp,
            "Salinity": data_salinity,
            "LBI Air Temp": data_lbi_temp,
            "Ocean Temp": data_ocean_temp,
            "Onshore flag": data_onshore_flag,
            "Pred Direction Degree": direction_deg,
            "Pred Direction label": direction_label,
            "Pred Wind Speed": self.pred_speed,
            "Pred Wind Gust": self.pred_gust,
            "Pred Onshore flag": self.pred_onshore_flag,
        }

        os.makedirs("results", exist_ok=True)
        pred_path = "results/prediction_results.csv"
        if os.path.exists(pred_path):
            pred = pd.read_csv(pred_path)
            pred = pd.concat([pred, pd.DataFrame([record])], ignore_index=True)
        else:
            pred = pd.DataFrame([record])
        pred.to_csv(pred_path, index=False)


def index():
    return rx.vstack(
        rx.text(f"Entered Number: {State.number_val}"),
        rx.input(
            type_="number",  # Sets input to accept only numbers
            on_change=State.set_number_val,
            placeholder="Enter a number",
        ),
    )

app = rx.App()
app.add_page(index, title="Barnegat Bay Wind Model")