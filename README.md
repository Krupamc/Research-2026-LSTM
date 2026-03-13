
# Research-2026-LSTM

# South Barnegat Bay Onshore Wind Model Prediction with the Use of Long Short-Term Memory Neural Networks

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
- [Training Program](#training-program)
- [Terminal Program](#terminal-program)
- [Graphs Program](#graphs-program)
- [Raspberry Pi Program](#raspberry-pi-program)
- [Source Code](#source-code)
- [Setup and Installation](#setup-and-installation)
  - [Executable Program Installation](#executable-program-installation)
  - [Python Installation](#python-installation)
- [How to Run](#how-to-run)
  - [Executable Program](#executable-program)
  - [Python Program](#python-program)
- [Limitations](#limitations)
- [Planned Development](#planned-development)
- [Acknowledgments](#acknowledgments)

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
- **Ocean buoy** (NDBC Station 44091 - https://www.ndbc.noaa.gov/station_page.php?station=44091).

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

The models were all evaulated using the `Mean Absolute Error` or MAE. MAE values are a measure of the average difference in units that forecasts may vary from the observed values.
- The Onshore wind model using values derived from the wind prediction is called the **Derived MAE**.

### MAE values of every model tested, with the chosen models highlighted.
<img width="734" height="187" alt="image" src="https://github.com/user-attachments/assets/e340c875-0073-4993-8ad4-a02eabd9e278" />

---

## Training Program

In this folder there are several files:
- A requirements.txt in order for you to download the dependencies to run these files for yourself. (more on this in the [Installation guide](#setup-and-installation))
- A `models` folder containing the saved scaler `.joblib` files and the Linear Regression `.joblib` model files. (Naive requires no model as it is just a formula)
- A notebook and regular python file version of the training program.
  - (Quick Summary)
  - Cleans the data. (as mentioned above in [Data Cleaning](#data-cleaning))
  - Changed date and hour columns into single **pandas** `Datetime` column.
  - Saved all input features into one **numpy** array.
  - Scale input array and output data.
  - Setup training length.
  - Predict values and calculate MAE.
  - Save to csv.

---

## Terminal Program

In this folder there are several files:
`executable` folder:
  - A .exe version of the program that requires no further steps to run. (more on this in the [Installation guide](#setup-and-installation))
  - - A `models` folder containing the saved scaler `.joblib` files, Linear Regression `.joblib` model files, and the saved scaler from the training program. (Naive requires no model as it is just a formula)
- `python` folder:
  - A requirements.txt in order for you to download the dependencies to run these files for yourself. (again more on this in the [Installation guide](#setup-and-installation))
  - A `models` folder containing the saved scaler `.joblib` files, Linear Regression `.joblib` model files, and the saved scaler from the training program. (Naive requires no model as it is just a formula)
  - A notebook and regular python file version of the training program.
    - (Quick Summary)
    - Cleans the data. (as mentioned above in [Data Cleaning](#data-cleaning))
    - Asks for data input from the users
    - Saved all input features into one **numpy** array.
    - Import saved models and scaler.
    - Scale the input data.
    - Predict values and report the results.
    - Save to csv and txt.
    - If the wind speed/wind gust speed prediction is over ten miles per hour, the user will be warned!
    
---

## Graphs Program

The data cleaned before being graphed using `Matplotlib`. All variables are graphed on their own and with their predicted values for comparison. You can see all of the graphs in the `graphs.ipynb` file.

---

## Raspberry Pi Program

The same Train and Terminal program was used for application on a 3.5 `Raspberry Pi` screen. Functionality is all the same except for the print lines being reshaped to fit on the smaller screen.

---

## Source Code

In this folder there are several files:
- A requirements.txt in order for you to download the dependencies to run these files for yourself. (more on this in the [Installation guide](#setup-and-installation))
- A `models` folder containing the saved scaler `.joblib` files and the Linear Regression `.joblib` model files. (Naive requires no model as it is just a formula)
- A `unused_models` folder containing other model prototypes. (Like the LSTM models and a few Linear Regression variations)
- All of the notebooks contain code for each final model. (A lot of this code is redundant which is why it was all summed up in the [Terminal Program](#training-program))

---

## Setup and Installation

### This tutorial is for *Windows* machines. A *macOS* Tutorial will soon follow.

### Executable Program Installation
1. Download the project files from this github and put them in a place of your choosing all in a single folder. (For this guide I will place them on my desktop in a folder called `lstm`)
<img width="953" height="421" alt="image" src="https://github.com/user-attachments/assets/4df3a0c8-7a10-4c0f-9a8f-9e12e96ba3ca" />


3. Download as a zip file and then unzip it.


<img width="425" height="287" alt="image" src="https://github.com/user-attachments/assets/939a9f63-9bd3-45bc-8e16-c3dda527dccc" />


4. To unzip the file, go to your `Downloads` folder in `Windows`, click on the folder and then extract or use a program like `winrar`. Once you have the extracted folder, move it to where you want. (I moved it to the `Desktop`)


<img width="673" height="141" alt="image" src="https://github.com/user-attachments/assets/bcbf3fdd-9738-4bcb-80ba-dc0899db806f" />


<img width="192" height="186" alt="image" src="https://github.com/user-attachments/assets/d4c0b64d-f73d-429e-a443-e3b14bb8ec1c" />

5. Tada! You have successfully installed the program and are ready to start using it.




### Python Installation

1. Go download python from: https://www.python.org/downloads/  The images below will show the process from the standalone installer of python 3.14.3. (3.11.14 or above will work)
<img width="891" height="316" alt="image" src="https://github.com/user-attachments/assets/f2208c30-e02b-427b-9c75-2f879e6b32b7" />

2. Run the downloaded file and click `Add python.exe to PATH` and `Install now`.
<img width="743" height="497" alt="image" src="https://github.com/user-attachments/assets/9a9a9964-bb7b-49ea-b152-8fca585f039e" />
<img width="711" height="448" alt="image" src="https://github.com/user-attachments/assets/457d5a4e-725e-4726-801f-15dbd378a6e8" />

4. Once setup is successful you can close the installer.
5. In the Windows start bar search for `cmd` or the command prompt (also known as a terminal) and open the application.
<img width="940" height="564" alt="image" src="https://github.com/user-attachments/assets/53a04308-01f7-43e7-994e-7892c6c67c64" />
<img width="1148" height="645" alt="image" src="https://github.com/user-attachments/assets/a5a4d218-dc5a-42ce-8be0-e674f5268de8" />

6. Now we need to update pip, python's package manager to be able to download the required dependencies. So type into the terminal `python.exe -m pip install --upgrade pip` and hit enter to run the command
<img width="1147" height="656" alt="image" src="https://github.com/user-attachments/assets/1dd77f8e-6a9c-4ed1-9bf3-ee3c4ca68945" />
<img width="1149" height="648" alt="image" src="https://github.com/user-attachments/assets/c419ee1f-3a04-44de-b9a5-fc6194be273f" />

### Virtual Environment Creation and Dependencies Installation - (A virtual environment is a seperate python installation than the base system)

1. Download the project files from this github and put them in a place of your choosing all in a single folder. (For this guide I will place them on my desktop in a folder called `lstm`)
<img width="953" height="421" alt="image" src="https://github.com/user-attachments/assets/4df3a0c8-7a10-4c0f-9a8f-9e12e96ba3ca" />


3. Download as a zip file and then unzip it.


<img width="425" height="287" alt="image" src="https://github.com/user-attachments/assets/5bed1838-eae2-40a7-b32c-37aae01d22d8" />


4. To unzip the file, go to your `Downloads` folder in `Windows`, click on the folder and than extract or use a program like `winrar`. Once you have the extracted folder, move it to where you want. (I moved it to the `Desktop`)
<img width="673" height="141" alt="image" src="https://github.com/user-attachments/assets/bcbf3fdd-9738-4bcb-80ba-dc0899db806f" />
<img width="192" height="186" alt="image" src="https://github.com/user-attachments/assets/d4c0b64d-f73d-429e-a443-e3b14bb8ec1c" />

5. In the command prompt, we need to get to the **full** file location of the project (C:\Users\<your_user>\Desktop\lstm) using the `cd` command. So find your full file path and take note of it.
6. Type in `cd` and then the file path. Then hit enter, the command prompt will display the location on the very left before the left angle bracket (>). (If you every get lost you can run the `dir` command to list the files in your directory)
<img width="1164" height="672" alt="image" src="https://github.com/user-attachments/assets/ef92d47b-1c7b-4b88-9d19-c4ae4eaca892" />

7. To create the actual virtual environment run `python.exe -m venv <env_name>`. You can choose the name Eg. lstmenv. (You can do a .<env_name> to make the files invisible to make working in the project easier. This is what I will do)
<img width="1138" height="649" alt="image" src="https://github.com/user-attachments/assets/9d42aa73-06f5-4e1b-84f3-a5adabb6c49b" />

8. Now we must activate the virtual environment. Run: `<env_name>\Scripts\activate.bat` (Make sure you use `\`'s and not `/`)
<img width="1132" height="641" alt="image" src="https://github.com/user-attachments/assets/2456cb29-82b2-44dd-be84-7f13f226bc5c" />

9. Finally we can install the dependencies. (If you are trying to run the `Terminal Program`, run `cd terminal_program/python` or cd into whatever folder of the program you are trying to run, and then run following command) Run: `pip install -r requirements.txt` to install from the list in requirements.txt. 
<img width="1129" height="326" alt="image" src="https://github.com/user-attachments/assets/70b85b04-f918-4bae-bef8-031cd0e174ab" />
<img width="1134" height="642" alt="image" src="https://github.com/user-attachments/assets/0b8960d0-4834-4739-a3b9-25db23ee78ae" />
<img width="1129" height="635" alt="image" src="https://github.com/user-attachments/assets/68690444-7876-44f3-8a02-c792fd38aefa" />



10. Tada! You have successfully installed the program and are ready to start using it. (You can close the command prompt)

---

## How to run

We can start predicting the wind any moment now! If the wind speed/wind gust speed prediction is over ten miles per hour, the user will be warned! The program will ask you for the current, mainland air temperature, humidity, wind direction, wind speed, wind gusting speed (or max wind), atmospheric pressure, precipitation rate, bay temperature, bay salinity, LBI air temperature, and the ocean temperature. You can get these values from these weather stations:

- **Mainland weather station** (Stafford Weather Underground Weather station - https://www.wunderground.com/dashboard/pws/KNJMANAH7/table/2025-08-10/2025-08-10/daily).
- **Long Beach Island (LBI) station** (e.g., Surf City Yacht Club Weather Underground Weather station - https://www.wunderground.com/dashboard/pws/KNJSURFC12/table/2025-08-8/2025-08-8/daily).
- **Bay buoy** (NJDEP MB_01 Buoy - https://njdep.rutgers.edu/continuous/graphing/NJBuoy767/).
- **Ocean buoy** (NDBC Station 44091 - https://www.ndbc.noaa.gov/station_page.php?station=44091).

- Due to the offline activity of the NJDEP buoy, you can collect bay temperature and bay salinity in the bay water near your location. (Or any other buoy inside of Barnegat Bay by LBI)
- Collect LBI air temperature from the Surf City Yacht Club weather station.
- Get the LBI ocean temperature from the NDBC buoy.
- Gather all other values from the Stafford Weather station.
- You can keep all of your temperatures in Celsius or Fahrenheit but keep it consistent for each type (Air vs Water)

Once you give the program the weather conditions and it predicts what is going to happen, it will save these results within the `results` folder of the `terminal_program` folder (Depends on if you use executable or python version!!! `...\lstm\terminal_program\executable\results` **or** `...\lstm\terminal_program\python\results`) as a csv and a txt. You can open the txt right away or open the csv in `Microsoft Excel`/`Google Sheets`. The time, current weather conditions, predicted conditions, and a few extra information is printed.

### Executable Program

1. Now that the program is all unzipped, go to where you put your `lstm` (Or whatever you renamed it to) folder and open the `terminal_program` folder.
<img width="192" height="186" alt="image" src="https://github.com/user-attachments/assets/d4c0b64d-f73d-429e-a443-e3b14bb8ec1c" />

2. Open the `executable` folder
<img width="1144" height="652" alt="image" src="https://github.com/user-attachments/assets/59bbdd49-c496-44c1-8fbf-c400b421bde7" />
<img width="1138" height="643" alt="image" src="https://github.com/user-attachments/assets/290d1722-35c2-49cf-be59-c5f88fc0751c" />
4. Double click the `Terminal Program` app to run it. (The one with the weather icon!)
5. Depending on your computer, you may encounter a pop up about a unknown app. (This program is safe. But if you would like to be make sure that the file was not modified you can run the file through a `SHA256` checksum (You can use: https://emn178.github.io/online-tools/sha256_checksum.html). If the result is not `c9aebca26523454a67853b6cc1e40ffc4ce1c1fc9f966f42a6f66629c722e3a3`, you downloaded a modified file, please delete it and then empty the `Recycling Bin`.
<img width="1153" height="668" alt="image" src="https://github.com/user-attachments/assets/0e6b0208-40d8-42f6-9eb7-723cdda4b0b0" />
6. Now hit the `More info` button.
<img width="1152" height="667" alt="image" src="https://github.com/user-attachments/assets/0085ad4d-974e-43c5-9c46-b3567a50d911" />
7. Finally, you can hit `Run anyway`.
<img width="1157" height="680" alt="image" src="https://github.com/user-attachments/assets/90666e81-8612-412b-8777-25ba21004543" />
8. The program is now running and ready for your inputs!
<img width="2301" height="1171" alt="image" src="https://github.com/user-attachments/assets/93504a22-3ba0-457a-bdc9-441d0ea58d2f" />

### Python Program

5. In the Windows start bar search for `cmd` or the command prompt (also known as a terminal) and open the application.
<img width="940" height="564" alt="image" src="https://github.com/user-attachments/assets/53a04308-01f7-43e7-994e-7892c6c67c64" />
<img width="1148" height="645" alt="image" src="https://github.com/user-attachments/assets/a5a4d218-dc5a-42ce-8be0-e674f5268de8" />
6. Type in `cd` and then the file path of the `lstm` (Or whatever you renamed it to) folder.\ls
<img width="1129" height="640" alt="image" src="https://github.com/user-attachments/assets/be4597d4-5626-42e5-a759-614b171d03cf" />
7. Activate your python enviroment by running: `<env_name>\Scripts\activate.bat` (Make sure you use `\`'s and not `/`)
<img width="1126" height="268" alt="image" src="https://github.com/user-attachments/assets/1c9f8117-62cc-4b36-8a88-23185e2f62f5" />
8. Run: `cd terminal_program/python`
<img width="1130" height="269" alt="image" src="https://github.com/user-attachments/assets/7336b8a2-2c97-48f3-924b-d6257a03be8f" />
9. Finally run `python terminal.py` to open the program.
<img width="1312" height="502" alt="image" src="https://github.com/user-attachments/assets/aaf0dcbb-40ca-4940-bee9-824bb7220083" />
10. The program is now running and ready for your inputs!

---

## Limitations

This project did set out what it planned to do: Create a program to predict the next hour's wind speed and even more! However, there are some limitations to the approach taken. 
- Due to the volatility of the changes in weather in the beginning and end of summer, the models are most accurate in the middle of summer. If the NJDEP buoy stayed online, models could also be created for the other seasons as well to have a year-wide working system.
- Naive persistance models were very effective when it came to predicting wind, however this type of model cannot handle great changes as it just assumes the last value is the current value.

---

## Planned Development

In the future, I would like to continue this project by:
- Reduce MAE with the use of different or perhaps **more** models.
- Gather historical data to train the models on.
- Gather data year-round in order to have different **sets** of models for every season (Or more if needed)
- Put this program onto a website for anyone to enjoy.
- Please email me with any other ideas or questions!

---

## Acknowledgments

I would like to thank so many for the oppertunity to work on this project. I thank my reserach advisors for encouraging me. I thank my computer science teacher for providing feedback and ideas. I thank one of my older teachers for all I know about machine learning development. I thank my contact with a meteorologist for all his insight and explanations. I thank my family for putting up with my hours of work. I thank my friends for not thinking I went insane. And most of all I thank you. You for reading. You for using this program.

----
