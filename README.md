
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


