
# Research-2026-LSTM

# South Barnegat Bay Onshore Wind & Upwelling Predictor

A time‑series deep learning project that uses an LSTM (Long Short‑Term Memory) neural network to predict sudden onshore wind events and upwelling conditions in southern Barnegat Bay, New Jersey. The model learns from hourly data to forecast the **next hour’s** wind speed, wind direction, onshore status, and upwelling status to help local boaters avoid hazardous conditions.
---

## Table of Contents

- [Background](#background)
- [Project Goals](#project-goals)
- [Data Sources](#data-sources)
- [Labels: Onshore and Upwelling](#labels-onshore-and-upwelling)
- [Model Design](#model-design)
- [Setup and Installation (To be writen)](#setup-and-installation)
- [How to Run (To be writen)](#how-to-run)
  - [1. Preprocess data](#1-preprocess-data)
  - [2. Create training windows](#2-create-training-windows)
  - [3. Train the models](#3-train-the-models)
  - [4. Evaluate with MAE](#4-evaluate-with-mae)
  - [5. Make a one‑hour‑ahead prediction](#5-make-a-one-hour-ahead-prediction)
- [Model Evaluation (To be writen)](#model-evaluation)
- [Planned Deployment (To be writen)](#planned-deployment)
- [Limitations (To be writen)](#limitations)
- [Future Work (To be writen)](#future-work)
- [Acknowledgments (To be writen)](#acknowledgments)

---

## Background

South Barnegat Bay frequently experiences **sudden, powerful onshore wind gusts** during summer, often connected to local upwelling events that bring cold water from deeper parts of the bay or nearby ocean to the surface. These gusts can rapidly create steep, choppy waves that are dangerous for kayaks, canoes, and personal watercraft.

Traditional weather models and public forecasts operate at regional scales (tens of kilometers, hours to days) and often **miss small‑scale, bay‑specific changes**, leaving local boaters with little warning.

This project explores whether a **local LSTM model** trained on detailed hourly observations can provide more accurate, location‑specific, one‑hour‑ahead warnings.

(Take a look at the attached research paper for more infomation)
---

## Project Goals

1. **Predict four key quantities one hour ahead** at a mainland station near Stafford:
   - Wind speed
   - Wind direction
   - Onshore flag (1 = onshore, 0 = offshore)
   - Upwelling flag (1 = likely upwelling‑driven conditions, 0 = not)

2. **Use only the previous 24 hours of data** (a sliding window) to make each prediction.[web:18]

3. **Evaluate model accuracy** using Mean Absolute Error (MAE), targeting **< 10% error** for wind speed/direction, comparable to short‑range weather forecasts.

4. Eventually **deploy the trained model** on a home server (Proxmox) and expose predictions via:
   - A simple website (Flask app),
   - A local desktop/terminal script.

---

## Data Sources

The model uses hourly observations from June 1 – September 24, 2025 (June 1 is excluded in the making of the upwelling flags) from multiple stations:

- **Mainland weather station** (Stafford area).
- **Long Beach Island (LBI) station** (e.g., Surf City Yacht Club).
- **Bay buoy** (NJDEP inside Barnegat Bay).
- **Offshore buoy** (NDBC).

Typical variables include:

- Air temperature (mainland, island)
- Wind speed and gusts
- Wind direction (16‑point compass)
- Humidity
- Precipitation
- Atmospheric pressure
- Bay water temperature
- Ocean water temperature

All data are combined into a single time‑indexed CSV with one row per hour.

---

## Labels: Onshore and Upwelling

### Wind direction → degrees

Wind directions such as N, NNE, NE, … are converted to degrees using a 16‑point compass mapping:

- N = 0°, NNE = 22.5°, NE = 45°, …, NNW = 337.5°.

A helper class handles this conversion and stores the degree values in the wind direction column.

### Onshore flag

An `Onshore` column (0/1) is created based on **direction sectors that blow from the ocean/bay toward the mainland**. For this project, directions roughly from **S through NE** are considered onshore, for example:

- S (180°), SSE (157.5°), SE (135°), ESE (112.5°), E (90°), ENE (67.5°), NE (45°), NNE (22.5°).

Rows with directions in this set are labeled `Onshore = 1`; all others are `0`.

### Upwelling flag

Upwelling in shallow coastal systems is often associated with:

- Persistent along‑shore/onshore winds, and
- Sudden drops in surface water temperature,
- Often with ocean water colder than recent bay temperatures.

This project’s simple definition for an `upwelling_flag` (0/1) is:

1. Onshore winds (Onshore = 1)
2. Wind speeds greater than a set threshold 
3. The absolute value of the two lowest ocean temps of the day averaged together with the curent bay temp subtracted being greator than a set threshold.

These conditions are implemented with tunable thresholds in code (e.g., ≥1–2°F difference).

The result is a labelled `upwelling_flag` column that the LSTM can try to predict.

---

## Model Design

The project currently uses **four** separate LSTM models, one for each target:

1. Wind speed
2. Wind direction (in degrees)
3. Onshore flag (classification)
4. Upwelling flag (classification)

Each model:

- Takes as input the previous 24 hours of multivariate data (all feature columns).
- Predicts the **next hour’s** value of its target.

### Features

Example feature set used for each 24‑hour input window:

- Mainland air temperature
- Island air temperature
- Bay water temperature
- Ocean water temperature
- Wind speed and gusts
- Wind direction (degrees)
- Humidity
- Atmospheric pressure
- Precipitation
- `Onshore` flag (current hour)
- `upwelling_flag` (optional as an input, depending on experiment)

All numeric features are standardized with `StandardScaler` before training.

### Time‑series windowing

For each sample:

- **Input**: rows [t‑23, …, t] with all feature columns  
- **Target**: row [t+1] with the single target column (e.g., wind speed at t+1)

These windows are built with a sliding‐window function that converts the cleaned DataFrame into NumPy arrays of shape:

- `X.shape = (num_samples, 24, num_features)`
- `y.shape = (num_samples,)` (or `(num_samples, 1)`)

### Train / validation / test split

Because this is time series, the split is **chronological**, not random:

- First 90% of windows → Training set
- Final 10% → Test set (never seen during training)

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
<img width="192" height="186" alt="image" src="https://github.com/user-attachments/assets/d4c0b64d-f73d-429e-a443-e3b14bb8ec1c" />

2. In the command prompt, we need to get to the **full** file location of the project (C:\Users\<your_user>\Desktop\lstm) using the `cd` command. So find your full file path and take note of it.
3. Type in `cd` and then the file path. Then hit enter, the command prompt will display the location on the very left before the left angle bracket (>). (If you every get lost you can run the `dir` command to list the files in your directory)
<img width="1164" height="672" alt="image" src="https://github.com/user-attachments/assets/ef92d47b-1c7b-4b88-9d19-c4ae4eaca892" />

3. To create the actual virtual enviroment run `python.exe -m venv <env_name>`. You can choose the name Eg. lstmenv. (You can do a .<env_name> to make the files invisible to make working in the project easier. This is what I will do)
<img width="1138" height="649" alt="image" src="https://github.com/user-attachments/assets/9d42aa73-06f5-4e1b-84f3-a5adabb6c49b" />

4. Now we must activate the virtual enviroment. Run: <env_name>\Scripts\activate.bat
<img width="1132" height="641" alt="image" src="https://github.com/user-attachments/assets/2456cb29-82b2-44dd-be84-7f13f226bc5c" />

5. Finally we can install the dependencies. Run: `pip install -r requirements.txt` to install from the list in requirements.txt. It will install Pandas, Tensorflow, Numpy, and Scikit-Learn
<img width="1134" height="645" alt="image" src="https://github.com/user-attachments/assets/e41a7226-5e1b-41e3-aeb1-c10da55bf7f0" />




