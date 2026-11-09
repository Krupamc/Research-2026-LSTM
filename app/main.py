from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests, os
from pathlib import Path
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime

# Turn on 
# fastapi dev --host 0.0.0.0  --port 8002 app/main.py
# Start server: uvicorn main:app --reload

app = FastAPI(title="Onshore Wind Model Predictions!")

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, "home.html", {"request": request})

@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots_txt():
    return """\
User-agent: *

Allow: /

"""

@app.get("/weather", response_class=HTMLResponse)
def weather(request: Request):
    return templates.TemplateResponse(request, "weather.html", {"request": request})

@app.get("/south-barnegat", response_class=HTMLResponse)
def south_barnegat(request: Request):
    return templates.TemplateResponse(request, "south-barnegat.html", {"request": request})

@app.post("/south-barnegat")
def south_barnegat_input(request: Request, mainland_temp: float = Form(), humidity: float = Form(), wind_dir: str = Form(), wind_speed: float = Form(), wind_gust: float = Form(), pressure: float = Form(), precipitation: float = Form(), salinity: float = Form(), lbi_temp: float = Form(), bay_temp: float = Form(), ocean_temp: float = Form(), air_in_celsius: bool = Form(), water_in_celsius: bool = Form()):
    # Load models/scaler
    dir = os.path.dirname(__file__)
    model_dir = os.path.join(dir, "models")

    scaler_x = load(os.path.join(model_dir, "scaler_x.joblib"))
    lr_speed = load(os.path.join(model_dir, "wind_speed_linear.joblib"))
    lr_gust = load(os.path.join(model_dir, "wind_gust_linear.joblib"))

    direction_map = {
        "N": 0, "NNE": 1, "NE": 2, "ENE": 3, 
        "E": 4, "ESE": 5, "SE": 6, "SSE": 7, 
        "S": 8, "SSW": 9, "SW": 10, "WSW": 11, 
        "W": 12, "WNW": 13, "NW": 14, "NNW": 15
    }

    allowed_dirs_deg = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
    compass = [
        "N","NNE","NE","ENE",
        "E","ESE","SE","SSE",
        "S","SSW","SW","WSW",
        "W","WNW","NW","NNW"
    ]

    if air_in_celsius is False:
        mainland_temp = round((mainland_temp - 32) * 5.0 / 9.0, 2)
        lbi_temp = round((lbi_temp - 32) * 5.0/9.0, 2)
    if water_in_celsius is False:
        ocean_temp = round((ocean_temp - 32) * 5.0/9.0, 2)
        bay_temp = round((bay_temp - 32) * 5.0/9.0, 2)

    # Map direction convert.
    wind_dir_bin = direction_map[wind_dir]
    direction_deg = allowed_dirs_deg[wind_dir_bin]
    direction_label = compass[wind_dir_bin]


    print("Variables Recieved")

    #rounding
    mainland_temp = round(mainland_temp, 2)
    humidity = round(humidity, 2)
    wind_speed = round(wind_speed, 2)
    wind_gust = round(wind_gust, 2)
    pressure = round(pressure, 2)
    precipitation = round(precipitation, 2)
    bay_temp = round(bay_temp, 2)
    salinity = round(salinity, 2)
    lbi_temp = round(lbi_temp, 2)
    ocean_temp = round(ocean_temp, 2)
    upwelling_flag = 0

    onshore_degrees = [8, 7, 6, 4, 3, 2, 1]
    onshore_flag = 0
    if wind_dir_bin in onshore_degrees:
        onshore_flag = 1

    # Input data into one array
    data = np.array([
        mainland_temp,
        humidity,
        wind_dir_bin,
        wind_gust,
        pressure,
        precipitation,
        bay_temp,
        salinity,
        lbi_temp,
        ocean_temp,
        onshore_flag,
        upwelling_flag,
    ]).reshape(1, -1)

    scaledx = scaler_x.transform(data)

    # Predict wind
    scaled_wind = lr_speed.predict(scaledx)
    wind_pred = float(scaled_wind.squeeze())
    wind_pred = np.maximum(wind_pred, 0.0)
    wind_pred = np.round(wind_pred, 2)

    # Input into another array
    data = np.array([
        mainland_temp,
        humidity,
        wind_dir_bin,
        wind_speed,
        pressure,
        precipitation,
        bay_temp,
        salinity,
        lbi_temp,
        ocean_temp,
        onshore_flag,
        upwelling_flag,
    ]).reshape(1, -1)

    scaledx = scaler_x.transform(data)

    # predict wind gust
    scaled_gust = lr_gust.predict(scaledx)
    gust_pred = float(scaled_gust.squeeze())
    gust_pred = np.maximum(gust_pred, 0.0)
    gust_pred = np.round(gust_pred, 2)

    if onshore_flag == 1:
        onshore_flag = "Yes!"
    else: onshore_flag = "No"

    data = {
        "wind_pred": wind_pred,
        "gust_pred": gust_pred,
        "direction": {"degree": direction_deg, "label": direction_label},
        "onshore": onshore_flag
    }
    print(data)
    return templates.TemplateResponse(request, "south-barnegat.html", {"request": request, "data": data})
