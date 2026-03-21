from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time, sys
import os
import pandas as pd
import numpy as np

#create variables for loading screen
def bouncing_bar():
    width = 20
    pos = 0
    direction = 1

    for _ in range(20):  # number of animation steps
        bar = [" "] * width
        bar[pos] = "#"
        sys.stdout.write("\r[" + "".join(bar) + "]")
        sys.stdout.flush()
        time.sleep(0.01)

        pos += direction
        if pos == 0 or pos == width - 1:
            direction *= -1
    print()

print()
print()
print("South Barnegat Bay Onshore Wind Model Prediction with the\nUse of Long Short-Term Memory Neural Networks - Terminal\nProgram for Real-Time Predictions\n")
print("Written by ----- \nWritten in Python 3.11.14\n")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("Damp South and Western onshore winds initiate the summertime event known as upwelling.\nUpwelling in small, localized areas like Barnegat Bay has a significant impact on bay temperature, creating a large land-sea temperature difference.\nThis difference can lead to harsh and fast onshore breezes\nthat can “swamp” small watercraft. This study uses an LSTM, a deep learning neural network,\nto predict these large gusts, along with Naive and Linear\nRegression algorithms to verify its effectiveness. To\nutilize these models, thirteen variables were collected on\nan hourly basis for June-August. Once the models were\ncreated and trained, the mean absolute error was calculated for each of themodels as a comparison.\nShockingly, it seemed that Linear\nRegressions and Naive models performed marginally better\nthan the LSTM, which had collapsed to\npredicting a value close to the mean in almost all tests.\nAccurate wind speeds and direction were still predicted, as the hypothesis says, just with different models.\n")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("-----Hello! Welcome to combined_program.py!-----\n")
print("-----I would like to thank for taking time to use my\npredition model for Barnegat Bay!-----\n")
print("-----If you encounter any problems, please let me know!-----\n")
print("-----(This model is only trained for summer (June-August)\nand will not perform if given data outside this period)-----")
print()

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("-----This program will ask for the weather conditions,\nand predict the next hour for you and save the\nresults in a results csv and txt-----")
print("\n")
print()
print("-----Several times you may be asked 'Yes' or 'No' questions and reply with (y/n) in the terminal-----")
print()
print()
print()
while True:
    reply = input("------Are you ready to start?\n(You have no choice in this one :D)-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please enter 'y' to continue-----")
bouncing_bar()
print()
print()
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

#Read the file
raw_data = pd.read_csv("Csv/observed_data/RAW_data.csv")

#Get all variables
print("-----This program will ask you for the row number to get a prediction for the next row to showcase the program-----")
while True:
    print()
    row = input("-----What is the row for which you are entering data?-----").strip().lower()
    row = int(row)
    row-=2
    print()
    print(raw_data.iloc[row])
    reply = input("Is this the row you would like? (y/n)")
    print()
    if reply in ("y"):
        break

#set all variables
data_main_air_temp = raw_data.iloc[row]['Mainland Air Temp']
data_humidity_per = raw_data.iloc[row]['Humidity (%)']
data_wind_direction = raw_data.iloc[row]['Direction (A)']
data_wind_speed = raw_data.iloc[row]['Wind Speed (A)']
data_gusting = raw_data.iloc[row]['Gusting']
data_pressure = raw_data.iloc[row]['Atmospheric Pressure (IN)']
data_rainfall = raw_data.iloc[row]['Precipitation Rate']
data_bay_temp = raw_data.iloc[row]['Bay Temp']
data_salinity = raw_data.iloc[row]['Salinity']
data_lbi_temp = raw_data.iloc[row]['LBI Air Temp']
data_ocean_temp = raw_data.iloc[row]['Ocean Temp']

#replace the direction column with the corresponding degree values
direction_map = {'N': 0,'NNE': 1,'NE': 2,'ENE': 3,'E': 4,'ESE': 5,'SE': 6,'SSE': 7,
            'S': 8,'SSW': 9,'SW': 10,'WSW': 11,'W': 12,'WNW': 13,'NW': 14,'NNW': 15
        }
data_wind_direction = direction_map[data_wind_direction]

#Convert to degrees
direction_deg = data_wind_direction
allowed_dirs_deg = np.array([
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5
])

direction_deg = allowed_dirs_deg[direction_deg]

#Convert to cardinal directions
direction_label = data_wind_direction
compass = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

direction_label = compass[direction_label]

#make sure values are in celsius
while True:
    reply = input("-----Are your Air Temperatures in Celsius (y/n)?").strip().lower()
    if reply in ("y", "n"):
        break
    print("-----Please enter 'y' or 'n'-----")
    print()
    
if reply=='n':
    #convert to Celsius
    data_main_air_temp = round((data_main_air_temp-32) * 5.0/9.0, 1)
    data_lbi_temp = round((data_lbi_temp  -32) * 5.0/9.0, 1)
    print("-----Converted air temperatures from Fahrenheit to Celsius...-----")
    bouncing_bar()
    print()
    
else:
    print("-----Thanks! Less work for me-----")

reply = input("-----Are your Water Temperatures in Celsius (y/n)?").strip().lower()
if reply=='n':
    #convert to Celsius
    data_ocean_temp = round((data_ocean_temp-32) * 5.0/9.0, 1)
    data_bay_temp = round((data_bay_temp-32) * 5.0/9.0, 1)
    print("-----Converted water temperatures from Fahrenheit to Celsius...-----")
    bouncing_bar()
    print()

else:
    print("-----Thanks! Less work for me-----")
    print()

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
print("-----Loading...-----")
bouncing_bar()
print()

#determines if its a onshore breeze and adds a new column for it
onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
if data_wind_direction in onshore_degrees:
    data_onshore_flag = 1
else:
    data_onshore_flag = 0  

print("-----Determined if it's an Onshore Breeze...------")
bouncing_bar()
print()

if data_onshore_flag == 1:
    print("-----It's an Onshore Breeze!-----")
else:    
    print("-----It's not an Onshore Breeze!-----")
bouncing_bar()
print()

#No upwelling can be predicted
print("-----No upwelling can be predicted for only one hour of data-----")
data_upwelling_flag = 0
bouncing_bar()
print()
#saves all input data into one Numpy array
dataset = np.array([
    data_main_air_temp,
    data_humidity_per,
    data_wind_direction,
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
])
print("-----Loading 1/4...-----")
dataset = dataset.reshape(1, -1)
bouncing_bar()

#Scaler
scaler_x = load("models/scaler_x.joblib")
scaledx = scaler_x.transform(dataset)
print("-----Loading 2/4...-----")
bouncing_bar()

#Open the model
reg_speed = load("models/wind_speed_linear.joblib")
print("-----Loading 3/4...-----")
bouncing_bar()

#Predict
speed_pred_lr = reg_speed.predict(scaledx)
speed_pred_lr = float(speed_pred_lr.squeeze())
speed_pred_lr = np.maximum(speed_pred_lr, 0.0)
speed_pred_lr = np.round(speed_pred_lr, 1)
print("-----Loading 4/4...-----")
bouncing_bar()
print("-----Predicted Wind Speed!-----")
bouncing_bar()

#saves all input data into one Numpy array
dataset = np.array([
    data_main_air_temp,
    data_humidity_per,
    data_wind_direction,
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
])
print("-----Loading 1/4...-----")
dataset = dataset.reshape(1, -1)
bouncing_bar()

#Scaler
scaledx = scaler_x.transform(dataset)
print("-----Loading 2/4...-----")
bouncing_bar()

#Open the model
reg_gust = load("models/wind_gust_linear.joblib")
print("-----Loading 3/4...-----")
bouncing_bar()

#Predict
gust_pred_lr = reg_gust.predict(scaledx)
gust_pred_lr = float(gust_pred_lr.squeeze())
gust_pred_lr = np.maximum(gust_pred_lr, 0.0)
gust_pred_lr = np.round(gust_pred_lr, 1)
print("-----Loading 4/4...-----")
bouncing_bar()
print("-----Predicted Wind Gust Speed!-----")
bouncing_bar()

direction_pred = data_wind_direction
print("-----Loading 1/3...-----")
bouncing_bar()

#Convert to degrees
allowed_dirs_deg = np.array([
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5
])
direction_pred_deg = allowed_dirs_deg[direction_pred]
print("-----Loading 2/3...-----")
bouncing_bar()

#Convert to cardinal directions
direction_pred_label = data_wind_direction
compass = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]
direction_pred_label = compass[direction_pred_label]
print("-----Loading 3/3...-----")
bouncing_bar()
print("-----Predicted Wind Direction!-----")
bouncing_bar()

#Onshore from direction
print("-----Loading 1/2...-----")
bouncing_bar()

onshore_bins = [8, 6, 7, 4, 3, 2, 1]

if data_wind_direction in onshore_bins:
    onshore_pred_flag = 1
else:
    onshore_pred_flag = 0

print("-----Loading 2/2...-----")
bouncing_bar()
print("-----Predicted if it's an Onshore Breeze!-----")
print()
time.sleep(0.05)
print()
time.sleep(0.05)
print()
time.sleep(0.05)
print()
time.sleep(0.05)
print()
time.sleep(0.05)
print()
time.sleep(0.05)
print()
time.sleep(0.05)

#Print the MAE report to the terminal
print(f"-----Wind Speed: {speed_pred_lr}-----")
print(f"-----Wind Gust: {gust_pred_lr}-----")
print(f"-----Wind Direction Degrees: {direction_pred_deg}-----")
print(f"-----Wind Direction Label: {direction_pred_label}-----")
print(f"-----Onshore Breeze from Direction: {onshore_pred_flag}-----")
if (speed_pred_lr >= 10) or (gust_pred_lr >= 10):
    print("\nBe careful on the water. High wind speeds predicted.\n")
print()
print("---------")
print("---------")
print("---------")
print()
print("-----Huh. That's it. If you would like to do some more\npredicting, you know where to go-----")
print()
print("-----You can find your 'all_predictions.csv' csv with all\nthe predictions in the 'predictions' folder as well as all\nthe MAE values saved to a 'mae_report.txt' file in the\n'predictions' folder-----")   
print()
print()
print()