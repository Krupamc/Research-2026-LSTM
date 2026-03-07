from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time, sys
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
print("South Barnegat Bay Onshore Wind Model Prediction with the\nUse of Long Short-Term Memory Neural Networks - Training and MAE\n")
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

print("This file contains all the code for training the Linear\nRegression models and the Naive models and finding their MAE values. If you would like to take a look at the LSTM code\ngo to 'Source Code/Unused LSTM'. Also within that folder is some code for a Linear Regression predicition for onshore\nflags.")
print()
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

print("-----This program will take a csv of your choosing,\n(make sure it is formatted as specified in the readme)-----")
print("-----return your orginal data (in the same location), a\neditied version for the model, and a results csv-----")

print("-----Several times you may be asked 'Yes' or 'No' questions and reply with (y/n) in the terminal-----")
print()

while True:
    reply = input("------Are you ready to start?\n(You have no choice in this one :D)-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please enter 'y' to continue-----")
print()
print()
print()

print("-----Now we will start 'cleaning' the data------")
bouncing_bar()
print()
print()
print()
#Read the file and watch for errors
print("-----Please make sure your file is in the 'observed_data'\nfolder and has been renamed 'RAW_data.csv'-----")
bouncing_bar()
filename = 'Csv/observed_data/RAW_data.csv'

try:
    untouched_csv = pd.read_csv(filename)
except FileNotFoundError:
    print(f"-----CSV not found: {filename}. Try renaming the csv to 'RAW_data.csv'-----")
except pd.errors.EmptyDataError:
    print(f"-----CSV is empty or corrupted: {filename}-----")
except Exception as e:
    print(f"-----Problem reading {filename}: {e}-----")

print("-----Opening your csv...-----")
bouncing_bar()
print("-----Reading all it's goodies....------")
bouncing_bar()
pd.set_option('future.no_silent_downcasting', True)  #no pandas warning
#replace the direction column with the corresponding degree values
direction_map = {'N': 0,'NNE': 1,'NE': 2,'ENE': 3,'E': 4,'ESE': 5,'SE': 6,'SSE': 7,
            'S': 8,'SSW': 9,'SW': 10,'WSW': 11,'W': 12,'WNW': 13,'NW': 14,'NNW': 15
        }

untouched_csv['Direction (A)'] = untouched_csv['Direction (A)'].replace(direction_map).infer_objects(copy = False)
untouched_csv['Direction (A)'] = untouched_csv['Direction (A)'].astype('int')
print("-----Remapped Directions to Bins...-----")
bouncing_bar()
#make sure values are in celsius
while True:
    reply = input("-----Are your Air Tempuratures in Celsius (y/n)?").strip().lower()
    if reply in ("y", "n"):
        break
    print("-----Please enter 'y' or 'n'-----")
    
if reply=='n':
    #convert to Celsius
    untouched_csv['Mainland Air Temp'] = round((untouched_csv['Mainland Air Temp']-32) * 5.0/9.0, 1)
    untouched_csv['LBI Air Temp'] = round((untouched_csv['LBI Air Temp']-32) * 5.0/9.0, 1)
    print("-----Converted air temperatures from\nFahrenheit to Celsius...-----")
    bouncing_bar()
    
else:
    print("-----Thanks! Less work for me-----")
    print()

reply = input("-----Are your Water Temperatures in Celsius (y/n)?").strip().lower()
if reply=='n':
    #convert to Celsius
    untouched_csv['Ocean Temp'] = round((untouched_csv['Ocean Temp']-32) * 5.0/9.0, 1)
    untouched_csv['Bay Temp'] = round((untouched_csv['Bay Temp']-32) * 5.0/9.0, 1)
    print("-----Converted water temperatures from Fahrenheit to Celsius...-----")
    bouncing_bar()
else:
    print("-----Thanks! Less work for me-----")
    print()

#round all of the columns
untouched_csv['Humidity (%)'] = round(untouched_csv['Humidity (%)'], 1)
untouched_csv['Wind Speed (A)'] = round(untouched_csv['Wind Speed (A)'], 1)
untouched_csv['Gusting'] = round(untouched_csv['Gusting'], 1)
untouched_csv['Atmospheric Pressure (IN)'] = round(untouched_csv['Atmospheric Pressure (IN)'], 2)
untouched_csv['Precipitation Rate'] = round(untouched_csv['Precipitation Rate'], 2)
untouched_csv['Bay Temp'] = round(untouched_csv['Bay Temp'], 2)
untouched_csv['Salinity'] = round(untouched_csv['Salinity'], 2)
untouched_csv['LBI Air Temp'] = round(untouched_csv['LBI Air Temp'], 1)
untouched_csv['Ocean Temp'] = round(untouched_csv['Ocean Temp'], 1)
print("-----Rounded Data...-----")
bouncing_bar()
#determines if its a onshore breeze and adds a new column for it
onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
untouched_csv['Onshore'] = untouched_csv['Direction (A)'].isin(onshore_degrees)
untouched_csv['Onshore'] = untouched_csv['Onshore'].astype(int)
print("-----Determined Onshore Breezes...------")
bouncing_bar()
total = untouched_csv["Onshore"].sum()
print("-----Found ", total, " onshore wind events...-----")
bouncing_bar()
#thresholds
ocean_thresh = 1.0
wind_thresh = 7.0
#Find rolling means for last few days and for the last few hours
backround_ocean = untouched_csv['Ocean Temp'].rolling(48, min_periods=12).mean()
recent_ocean = untouched_csv['Ocean Temp'].rolling(6, min_periods=3).mean()
delta = recent_ocean - backround_ocean

#threshols
drop_threshold = -1.5
min_sustain_hours = 7
#Check if its a upwelling direction
upwell_wind = [6, 7, 8, 12]
upwell_wind = untouched_csv["Direction (A)"].isin(upwell_wind)


candidate = (delta <= drop_threshold)

# sustained: run-length filtering
candidate_int = candidate.astype(int)
run_lengths = candidate_int.groupby(
    (candidate_int != candidate_int.shift()).cumsum()
).transform("size")

sustained = (candidate & (run_lengths >= min_sustain_hours))

untouched_csv["upwelling_sustained_drop_flag"] = sustained.astype(int)

untouched_csv["upwelling_flag"] = (untouched_csv["upwelling_sustained_drop_flag"] & upwell_wind).astype(int)

print("-----Found upwelling events...-----")
bouncing_bar()
total = untouched_csv.index[untouched_csv["upwelling_flag"] == 1]
print("-----Checking my answers...-----")
bouncing_bar()
print("-----Found ", len(total), " upwelling events...-----")
bouncing_bar()
#List the indices of the upwell events
#print(list(total))
untouched_csv.head()
#drop the colomns
untouched_csv.drop(['upwelling_sustained_drop_flag'], axis=1, inplace=True)
untouched_csv = untouched_csv.dropna(how="all").reset_index(drop=True)
print("-----Fixing typos....-----")
bouncing_bar()
untouched_csv.head()
#Save this data frame to a Finished csv
#Removes first 24 hrs of data to remove NA's
untouched_csv = untouched_csv[24:]
untouched_csv.to_csv('Csv/CLEAN.csv', index=False)
print("-----Saved cleaned data to 'Csv/CLEAN.csv'-----")
print()
print()
print()
print("-----Hazah! You've made it thus far. Your data has been\ncleaned for the models-----")
bouncing_bar()
print()
print()
#Opens a new dataframe with the Clean csv
cleancsv = pd.read_csv('Csv/CLEAN.csv')
print("-----Opened our cleaned up csv...-----")
bouncing_bar()
print("-----Gobbling up all the tasty data....-----")
bouncing_bar()
#Convert data into Date time and create date filter
cleancsv['Date'] = pd.to_datetime(cleancsv['Date'])
cleancsv['Date'] = cleancsv['Date'] + pd.to_timedelta(cleancsv["Hr"], unit="h")
cleancsv.drop('Hr', axis=1, inplace=True)
print("-----Converted Date and Hour to DateTime...------")
bouncing_bar()
#Ask if they want a date filter
while True:
    reply = input("------Would you like to filter your data by date? (y/n)-----").strip().lower()
    if reply in ("y", "n"):
        break
    print("-----Please enter 'y' or 'n' to continue-----")

if reply == "y":
    # only ask for dates AFTER the user chose 'y'
    start_str = input("-----Enter START date (YYYY-MM-DD): ").strip()
    end_str   = input("-----Enter END date   (YYYY-MM-DD): ").strip()
    bouncing_bar()
    try:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end   = datetime.strptime(end_str, "%Y-%m-%d")

        if end < start:
            raise ValueError("End date must be after start date")

        mask = (cleancsv["Date"] >= start) & (cleancsv["Date"] <= end)
        cleancsv = cleancsv.loc[mask].reset_index(drop=True)
        print(f"-----Filtered data to dates between {start_str} and {end_str}. Rows: {len(cleancsv)}-----")

    except ValueError as e:
        print(f"-----Date error: {e}-----")
    except Exception as e:
        print(f"------Unexpected error while filtering dates: {e}-----")
else:
    print("-----Continuing without filtering data...-----")
    bouncing_bar()
#Create month colomn and restrict to only summer months
summer_mask = cleancsv["Date"].dt.month.isin([6, 7, 8, 9])
cleancsv = cleancsv[summer_mask].reset_index(drop=True)
print("----Making sure your data is summer only...-----")
bouncing_bar()
print()
print()
print()
print("-----Now the fun part! Model time! With your results you\nalso get a four course meal of models with a side\nof Numpy(pie)-----")
bouncing_bar()
#Prepare colomns into variables
data_main_air_temp = cleancsv['Mainland Air Temp']
data_humidity_per = cleancsv['Humidity (%)']
data_wind_direction = cleancsv['Direction (A)']
data_wind_speed = cleancsv['Wind Speed (A)']
data_gusting = cleancsv['Gusting']
data_pressure = cleancsv['Atmospheric Pressure (IN)']
data_rainfall = cleancsv['Precipitation Rate']
data_bay_temp = cleancsv['Bay Temp']
data_salinity = cleancsv['Salinity']
data_lbi_temp = cleancsv['LBI Air Temp']
data_ocean_temp = cleancsv['Ocean Temp']
data_onshore_flag = cleancsv['Onshore']
data_upwelling_flag = cleancsv['upwelling_flag']
print("-----Converting Columns to variables------")
bouncing_bar()
print()
print()
print()
print("-----Model One: Wind Speed-----")
bouncing_bar()
#saves all input data into one Numpy array
dataset = np.column_stack([
    data_main_air_temp.values,
    data_humidity_per.values,
    data_wind_direction.values,
    #data_wind_speed.values,
    data_gusting.values,
    data_pressure.values,
    data_rainfall.values,
    data_bay_temp.values,
    data_salinity.values,
    data_lbi_temp.values,
    data_ocean_temp.values,
    data_onshore_flag.values,
    data_upwelling_flag.values,
])
print("-----Cooking up a nice Numpy array...-----")
bouncing_bar()
#Save output data into variables and reshape it to be a 1d array
output_data = data_wind_speed.values
output_data = np.array(output_data).reshape(-1, 1)
print("-----Reshaping our output to a 1D array...-----")
bouncing_bar()
#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.90)) #Use 90% of training data
print("-----Setting the length of the training data...-----")
bouncing_bar()
#Scaler
scaler_x= StandardScaler()
scaler_y= StandardScaler()


scaledx = scaler_x.fit_transform(dataset)
scaledy = scaler_y.fit_transform(output_data)
print("-----Scaling with love...-----")
bouncing_bar()
#Setting to all
X_all = scaledx          # (N, 12)
y_all = output_data
print("-----Scaling with everything in mind...-----")
bouncing_bar()
#Train's and test
X_train = X_all[:training_data_len]
y_train = y_all[:training_data_len]
X_test  = X_all[training_data_len:]
y_test  = y_all[training_data_len:]
print("-----Chooo-chooooo! Setting up our X/y trains-----")
print("-----Don't forget about our X/y tests-----")
bouncing_bar()
#Open the model
reg_speed = load("models/wind_speed_linear.joblib")
print("-----Opening up the model...-----")
bouncing_bar()
#Predict
speed_pred_lr = reg_speed.predict(X_test)
speed_pred_lr = np.maximum(speed_pred_lr, 0.0)
speed_pred_lr = np.round(speed_pred_lr, 1)
print("-----Doing the math...-----")
bouncing_bar()
print("-----It's a lot of math okay?-----")
bouncing_bar()
#MAE
speed_mae_lr = mean_absolute_error(y_test, speed_pred_lr)
speed_mae_lr = round(speed_mae_lr, 2)
print("-----Checking my work...-----")
bouncing_bar()
print("-----Linear MAE of:", speed_mae_lr, "-----")
bouncing_bar()
# attach to dataframe for export
speed_linear = cleancsv.iloc[training_data_len:training_data_len + len(speed_pred_lr)].copy()
speed_linear["wind_speed_pred_linear"] = speed_pred_lr
speed_linear.to_csv("Csv/predictions/predicted_wind_speed.csv", index=False)
print("----Typing up a essay of data for you...-----")
bouncing_bar()
print("-----YES! All saved. Moving on....-----")
bouncing_bar()
print()
print()
print("-----Model Two: Wind Gust Speed-----")
bouncing_bar()
#saves all input data into one Numpy array
dataset = np.column_stack([
    data_main_air_temp.values,
    data_humidity_per.values,
    data_wind_direction.values,
    data_wind_speed.values,
    #data_gusting.values,
    data_pressure.values,
    data_rainfall.values,
    data_bay_temp.values,
    data_salinity.values,
    data_lbi_temp.values,
    data_ocean_temp.values,
    data_onshore_flag.values,
    data_upwelling_flag.values,
])
print("-----Cooking up a nice Numpy array...-----")
bouncing_bar()
#Save output data into variables and reshape it to be a 2d array
output_data = data_gusting.values
output_data = np.array(output_data).reshape(-1, 1)
print("-----Reshaping our output to a 1D array...-----")
bouncing_bar()
#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.90)) #Use 90% of training data
print("-----Setting the length of the training data...-----")
bouncing_bar()
#Scaler
scaler_x= StandardScaler()
scaler_y= StandardScaler()


scaledx = scaler_x.fit_transform(dataset)
scaledy = scaler_y.fit_transform(output_data)
print("-----Scaling with love...-----")
bouncing_bar()
#Setting to all
X_all = scaledx
y_all = output_data
print("-----Scaling with everything in mind...-----")
bouncing_bar()
#Train's and test
X_train = X_all[:training_data_len]
y_train = y_all[:training_data_len]
X_test  = X_all[training_data_len:]
y_test  = y_all[training_data_len:]
print("-----Chooo-chooooo! Setting up our X/y trains-----")
bouncing_bar()
print("-----Don't forget about our X/y tests-----")
bouncing_bar()
#Open the model
reg_gust = load("models/wind_gust_linear.joblib")
print("-----Opening up the model...-----")
bouncing_bar()
#Predict
gust_pred_lr = reg_gust.predict(X_test)
gust_pred_lr = np.maximum(gust_pred_lr, 0.0)
gust_pred_lr = np.round(gust_pred_lr, 1)
print("-----Doing the math...-----")
bouncing_bar()
print("-----It's a lot of math okay?-----")
bouncing_bar()
#MAE
gust_mae_lr = mean_absolute_error(y_test, gust_pred_lr)
gust_mae_lr = round(gust_mae_lr, 2)
print("-----Checking my work...-----")
bouncing_bar()
print("-----Linear MAE of:", gust_mae_lr, "-----")
bouncing_bar()
# attach to dataframe for export
linear_df = cleancsv.iloc[training_data_len:training_data_len + len(gust_pred_lr)].copy()
linear_df["wind_gust_pred_linear"] = gust_pred_lr
linear_df.to_csv("Csv/predictions/predicted_wind_gust.csv", index=False)
print("-----YES! All saved. Moving on....-----")
bouncing_bar()
print()
print()
print("-----Model Three: Wind Direction-----")
bouncing_bar()
#Create Naive model
wd = cleancsv['Direction (A)'].values.astype(int)
print("-----Creating a Naive model...-----")
bouncing_bar()
#Train the model
direction_pred = wd[training_data_len-1:-1]
direction_true = wd[training_data_len:]
print("-----Teaching the model a few tricks...-----")
bouncing_bar()
#Stop negatives and round to X.X
direction_pred_pred = np.maximum(direction_pred, 0.0)
print("-----Stopping negatives and rounding...-----")
bouncing_bar()
#Find MAE
direction_mae_naive = mean_absolute_error(direction_true, direction_pred_pred)
direction_mae_naive = round(direction_mae_naive, 2)
print("-----Checking my work...-----")
bouncing_bar()
print("-----Naive MAE of:", direction_mae_naive, "-----")
bouncing_bar()
print("-----YES! All saved. Moving on....-----")
bouncing_bar()
print()
print()
print("-----Model Four: Onshore Breezes-----")
bouncing_bar()
#Save onshore to variable and check if the predicted direction is onshore
onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
onshore_pred_flag = np.isin(direction_pred, onshore_degrees).astype(int)
print("-----Determining onshore breezes...-----")
bouncing_bar()
#Find MAE of onshore
onshore_check = cleancsv['Onshore'].iloc[training_data_len:].values.astype(int)
mae_onshore = mean_absolute_error(onshore_check, onshore_pred_flag)
mae_onshore = round(mae_onshore, 2)
print("-----Onshore flag MAE (naive-direction):", mae_onshore, "-----")
bouncing_bar()
#Save to csv
naive_df = cleancsv.iloc[training_data_len:].copy()
naive_df["wind_direction_pred_naive"] = direction_pred
naive_df["onshore_pred_direction"] = onshore_pred_flag
naive_df.to_csv("Csv/predictions/predicted_direction.csv", index=False)
print("-----YES! All saved. Moving on....-----")
bouncing_bar()
print()
print()
print("-----Model Five: Upwelling-----")
print("-----This model uses less data for training and more for\ntesting due to lack of later upwelling events-----") 
bouncing_bar()
#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.60)) #Use 90% of training data
print("-----Setting the length of the training data...-----")
bouncing_bar()
#Naive model creation
up = cleancsv['upwelling_flag'].values
print("-----Creating a Naive model...-----")   
bouncing_bar()
#Fitting the model
upwell_pred = up[training_data_len-1:-1]
upwell_true = up[training_data_len:]
#Calculate MAE
upwell_mae_naive = mean_absolute_error(upwell_true, upwell_pred)
upwell_mae_naive = round(upwell_mae_naive, 2)
print("-----Checking my work...-----")
bouncing_bar()
print("-----Naive MAE of:", upwell_mae_naive, "-----")
bouncing_bar()
#Save to csv
upwell_df = cleancsv.iloc[training_data_len:].copy()
upwell_df["upwelling_pred_naive"] = upwell_pred
upwell_df["upwelling_pred_naive"] = upwell_df["upwelling_pred_naive"].astype(int)

upwell_df.to_csv("Csv/predictions/predicted_upwelling.csv", index=False)
print("-----YES! All saved. Moving on....-----")
bouncing_bar()
print()
print()
print("-----Whew! That was a lot! But we are almost there!\nJust a few more steps-----")
bouncing_bar()
#Load clean csv to use to merge data
clean = pd.read_csv("Csv/CLEAN.csv")
clean["Date"] = pd.to_datetime(clean["Date"]) + pd.to_timedelta(clean["Hr"], unit="h")
clean = clean.drop(columns=["Hr"])
print("-----Opening your csv...-----")
bouncing_bar()
print("-----Time is nothing...But thats why we have Datetime...-----")
bouncing_bar()
#Data frames for every csv
pred_direction = pd.read_csv('Csv/predictions/predicted_direction.csv', sep=',')
pred_upwelling = pd.read_csv('Csv/predictions/predicted_upwelling.csv', sep=',')
pred_wind_gust = pd.read_csv('Csv/predictions/predicted_wind_gust.csv', sep=',')
pred_wind_speed = pd.read_csv('Csv/predictions/predicted_wind_speed.csv', sep=',')

for df in [pred_direction, pred_upwelling, pred_wind_gust, pred_wind_speed]:
    df["Date"] = pd.to_datetime(df["Date"])
    print("-----Opening even more csvs...-----")
    bouncing_bar()
#Merge all the csv's into one using the dates
merged = clean.merge(
    pred_upwelling[["Date", "upwelling_pred_naive"]],
    on="Date", how="outer"
)

merged = merged.merge(
    pred_direction[["Date", "onshore_pred_direction"]],  
    on="Date", how="left"
)

merged = merged.merge(
    pred_direction[["Date", "wind_direction_pred_naive"]],  
    on="Date", how="left"
)

merged = merged.merge(
    pred_wind_speed[["Date", "wind_speed_pred_linear"]],
    on="Date", how="outer"
)

merged = merged.merge(
    pred_wind_gust[["Date", "wind_gust_pred_linear"]],
    on="Date", how="outer"
)


print("-----Merging all the csvs together...-----")
bouncing_bar()
print("-----Merging all the csvs together...-----")
bouncing_bar()
print("-----Merging all the csvs together...-----")
bouncing_bar()
print("-----Merging all the csvs together...-----")
bouncing_bar()
print("-----Alright that's enough of that-----")
bouncing_bar()
#Sort by date
merged = merged.sort_values("Date").reset_index(drop=True)
#Make warning colomn if wind speeds are over 10 mph
merged["Wind_pred_warning"] = ((merged["wind_speed_pred_linear"] > 10) | (merged["wind_gust_pred_linear"] > 10))
#Save to csv
merged.to_csv("Csv/predictions/all_predictions.csv", index=False)
print("-----Saving the final csv with all the predictions...-----")
bouncing_bar()
with open("Csv/predictions/mae_report.txt", "w") as f:
    f.write(f"Wind Speed Linear MAE: {round(speed_mae_lr, 2)}\n")
    f.write(f"Wind Gust Linear MAE: {round(gust_mae_lr, 2)}\n")
    f.write(f"Upwelling Naive MAE: {round(upwell_mae_naive, 2)}\n")
    f.write(f"Wind Direction Naive MAE: {round(direction_mae_naive, 2)}\n")
    f.write(f"Onshore Breeze from Direction MAE: {round(mae_onshore, 2)}\n")
print()

#Print the MAE report to the terminal as well
print(f"-----Wind Speed Linear MAE: {round(speed_mae_lr, 2)}-----")
print(f"-----Wind Gust Linear MAE: {round(gust_mae_lr, 2)}-----")
print(f"-----Upwelling Naive MAE: {round(upwell_mae_naive, 2)}-----")
print(f"-----Wind Direction Naive MAE: {round(direction_mae_naive, 2)}-----")
print(f"-----Onshore Breeze from Direction MAE: {round(mae_onshore, 2)}-----")
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