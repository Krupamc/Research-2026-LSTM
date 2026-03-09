
# Research-2026-LSTM

# South Barnegat Bay Onshore Wind & Upwelling Predictor

A time‑series deep learning project that uses an LSTM (Long Short‑Term Memory) neural network, Linear Regression model, and Naive persistance model to predict sudden onshore wind events and upwelling conditions in southern Barnegat Bay, New Jersey. The model learns from hourly data to forecast the **next hour’s** wind speed, wind gusting speed, wind direction, onshore status, and upwelling status to help local boaters avoid hazardous conditions.
---

## Table of Contents

- [Background](#background)
- [Project Goals](#project-goals)
- [Current Model Setup](#current-model-setup)
- [Data Sources](#data-sources)
- [Data Cleaning](#data-cleaning)
- [Model Design](#model-design)
- [Model Evaluation](#model-evaluation)
- [Setup and Installation (To be writen)](#setup-and-installation)
- [How to Run (To be writen)](#how-to-run)
  - [1. Preprocess data](#1-preprocess-data)
  - [2. Create training windows](#2-create-training-windows)
  - [3. Train the models](#3-train-the-models)
  - [4. Evaluate with MAE](#4-evaluate-with-mae)
  - [5. Make a one‑hour‑ahead prediction](#5-make-a-one-hour-ahead-prediction)
- [Planned Deployment (To be writen)](#planned-deployment)
- [Limitations (To be writen)](#limitations)
- [Future Work (To be writen)](#future-work)
- [Acknowledgments (To be writen)](#acknowledgments)

---

## Background

South Barnegat Bay frequently experiences **sudden, powerful onshore wind gusts** during summer, often connected to local upwelling events that bring cold water from deeper parts of the bay or nearby ocean to the surface. These gusts can rapidly create steep, choppy waves that are dangerous for kayaks, canoes, and personal watercraft.

Traditional weather models and public forecasts operate at regional scales and often **miss small‑scale, bay‑specific changes**, leaving local boaters with little warning.

This project explores whether a **local LSTM model** trained on detailed hourly observations can provide more accurate, location‑specific, one‑hour‑ahead warnings.

(Take a look at the attached research paper for more infomation)
---

## Project Goals

1. **Predict five key quantities one hour ahead** at a mainland station near Stafford:
   - Wind speed
   - Wind gust speed
   - Wind direction
   - Onshore flag (1 = onshore, 0 = offshore)
   - Upwelling flag (1 = likely upwelling‑driven conditions, 0 = not)

2. Use LSTM (With a 24 hour sliding window), Linear Regression, and Naive models to make each prediction.

3. **Evaluate accuracy of the models** using Mean Absolute Error (MAE), targeting **< 10% error** for wind speed/direction, comparable to short‑range weather forecasts.

4. Eventually **deploy the trained model** via:
   - [ ] A simple website
   - [x] A local desktop/terminal script.
   - [x] A executable program.

---

## Model Design

The project currently uses **five** separate models. (Every older model version can be found under source_code/unused_models) Including one for each target:

1. Wind speed
2. Wind Gusting speed
3. Wind direction (in bins)
4. Onshore flag (binary)
5. Upwelling flag (binary)

Each model:

- Takes as input the previous hour of multivariate data (all feature columns).
- Predicts the **next hour’s** value of its target.

### Features

- Mainland air temperature
- Long Beach Island air temperature
- Bay water temperature
- Ocean water temperature
- Wind speed and gusts
- Wind direction (bins)
- Humidity
- Atmospheric pressure
- Precipitation
- `Onshore` flag (current hour)
- `upwelling_flag` 

All numeric features are standardized with `StandardScaler` before training.

---

## Current Model Setup

Currently this project is using:

- `Linear Regression`: Wind Speed
- `Linear Regression`: Wind Gusting Speed
- `Naive`: Wind Direction
- `Derived from Direction Prediction`: Onshore flag (binary)
- No models were able to predict for Upwelling events due to the low chances of occurance, leading to the models all collasping.

## Data Sources

The model uses hourly observations from June 1 – September 24, 2025 (June 1 is excluded in the making of the upwelling flags) from multiple stations:

- **Mainland weather station** (Stafford Weather Underground Weather station - https://www.wunderground.com/dashboard/pws/KNJMANAH7/table/2025-08-10/2025-08-10/daily).
- **Long Beach Island (LBI) station** (e.g., Surf City Yacht Club Weather Underground Weather station - https://www.wunderground.com/dashboard/pws/KNJSURFC12/table/2025-08-8/2025-08-8/daily).
- **Bay buoy** (NJDEP MB_01 Buoy - https://njdep.rutgers.edu/continuous/graphing/NJBuoy767/).
- **Offshore buoy** (NDBC Station 44091 - https://www.ndbc.noaa.gov/station_page.php?station=44091).

Typical variables include:

- Air temperature (mainland, island)
- Wind speed and gusts
- Wind direction (16‑point compass)
- Humidity
- Precipitation
- Atmospheric pressure
- Bay water temperature and salinity
- Ocean water temperature

All data are combined into a single time‑indexed CSV with one row per hour.

---

## Data Cleaning

### Wind direction converted to Bins

Wind directions such as N, NNE, NE, … are converted to **'bins'** using a 16‑point (0-15) compass mapping:

- N = 0°, NNE = 1°, NE = 2°, …, NNW = 15°.

### Convert Farenheit to Celsius
Converted all of the Land Air tempertures to Celsius using `C° = (F°-32) x 5/9`

### Round all the columns
All columns were rounded to `1` decimal place except:
- Bay Temperature `(2 places)`
- Bay Salinity `(2 places)`
- Precipitation `(2 places)`
- Atmospheric Pressure `(2 places)`

### Onshore flag

An `Onshore` column (0/1) is created based on **direction sectors that blow from the ocean/bay toward Long Beach Island**. For this project, directions roughly from **S through NE** are considered onshore, for example:

- S (8), SSE (7), SE (6), ESE (5), E (4), ENE (3), NE (2), NNE (1).

Rows with directions in this set are labeled `Onshore = 1`; all others are `0`.

### Upwelling flag

Upwelling in shallow coastal systems is often associated with:

- Persistent along‑shore/onshore winds, and
- Sudden drops in surface water temperature,
- Often with ocean water colder than recent ocean temperatures.

To find out if the current hour is upwelling:
1. Check if the wind direction is a **damp** upwelling wind `(SE, SSE, S, W)` 
2. The rolling mean of the last `48` hours of ocean temperature is subtracted from the rolling mean of the last `6` hours of ocean temperature.
3. If this number is less or equal to than a defined threshold `(-3.0)`, and is sustained for `6` hours it is considered upwelling.
4. The result is a labelled `upwelling_flag` column that the models can try to predict. **(All numbers mentioned above can be configured and changed in the training program)**

---

## Model Evaluation


## Setup and Installation

### This tutorial will be for *Windows* machines. A Macos Tutorial will soon follow.

### Python Installation

1. Go download python from: https://www.python.org/downloads/  The images below will show the process from the standalone installer of python 3.14.3. (3.11.14 or above will work)
<img width="891" height="316" alt="image" src="https://github.com/user-attachments/assets/f2208c30-e02b-427b-9c75-2f879e6b32b7" />

2. Run the downloaded file and click `Add python.eve to PATH` and `Install now`.
<img width="743" height="497" alt="image" src="https://github.com/user-attachments/assets/9a9a9964-bb7b-49ea-b152-8fca585f039e" />
<img width="711" height="448" alt="image" src="https://github.com/user-attachments/assets/457d5a4e-725e-4726-801f-15dbd378a6e8" />

4. Once setup is successful you can close the installer.
5. In the Window's start bar search for `cmd` or the command prompt (also known as a terminal) and open the application.
<img width="940" height="564" alt="image" src="https://github.com/user-attachments/assets/53a04308-01f7-43e7-994e-7892c6c67c64" />
<img width="1148" height="645" alt="image" src="https://github.com/user-attachments/assets/a5a4d218-dc5a-42ce-8be0-e674f5268de8" />

6. Now we need to update pip, python's package manager to be able to download the required dependancies. So type into the terminal `python.exe -m pip install --upgrade pip` and hit enter to run the command
<img width="1147" height="656" alt="image" src="https://github.com/user-attachments/assets/1dd77f8e-6a9c-4ed1-9bf3-ee3c4ca68945" />
<img width="1149" height="648" alt="image" src="https://github.com/user-attachments/assets/c419ee1f-3a04-44de-b9a5-fc6194be273f" />

### Virtual Enviroment Creation and Dependencies Installation - (A virtual enviroment is a seperate python installation than the base system)

1. Download the project files from this github and put them in a place of your choosing all in a single folder. (For this guide I will place them on my desktop in a folder called `lstm`)
<img width="953" height="421" alt="image" src="https://github.com/user-attachments/assets/4df3a0c8-7a10-4c0f-9a8f-9e12e96ba3ca" />
3. Download as a zip file and then unzip it.
<img width="448" height="320" alt="image" src="https://github.com/user-attachments/assets/8e0180b7-0186-4de6-8ae9-ac642585f685" />

<img width="192" height="186" alt="image" src="https://github.com/user-attachments/assets/d4c0b64d-f73d-429e-a443-e3b14bb8ec1c" />

4. In the command prompt, we need to get to the **full** file location of the project (C:\Users\<your_user>\Desktop\lstm) using the `cd` command. So find your full file path and take note of it.
4. Type in `cd` and then the file path. Then hit enter, the command prompt will display the location on the very left before the left angle bracket (>). (If you every get lost you can run the `dir` command to list the files in your directory)
<img width="1164" height="672" alt="image" src="https://github.com/user-attachments/assets/ef92d47b-1c7b-4b88-9d19-c4ae4eaca892" />

5. To create the actual virtual enviroment run `python.exe -m venv <env_name>`. You can choose the name Eg. lstmenv. (You can do a .<env_name> to make the files invisible to make working in the project easier. This is what I will do)
<img width="1138" height="649" alt="image" src="https://github.com/user-attachments/assets/9d42aa73-06f5-4e1b-84f3-a5adabb6c49b" />

6. Now we must activate the virtual enviroment. Run: <env_name>\Scripts\activate.bat
<img width="1132" height="641" alt="image" src="https://github.com/user-attachments/assets/2456cb29-82b2-44dd-be84-7f13f226bc5c" />

7. Finally we can install the dependencies. Run: `pip install pandas tensorflow numpy scikit-learn nbformat` to install from the list in requirements.txt. It will install Pandas, Tensorflow, Numpy, and Scikit-Learn
<img width="1134" height="645" alt="image" src="https://github.com/user-attachments/assets/e41a7226-5e1b-41e3-aeb1-c10da55bf7f0" />




