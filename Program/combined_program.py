from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time, sys
import pandas as pd
import numpy as np

print()
print("South Barnegat Bay Onshore Wind Model Prediction with the Use of Long Short-Term Memory Neural Networks - Training and Predictions\n")
print("Written by ----- \nWritten in Python 3.11.14\n")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please press enter to continue-----")
print()


print("Damp South and Western onshore winds initiate the summertime event known as upwelling.\nUpwelling in small, localized areas like Barnegat Bay has a significant impact on bay temperature, creating a large land-sea temperature difference.\nThis difference can lead to harsh and fast onshore breezes that can “swamp” small watercraft. This study uses an LSTM, a deep learning neural network,\nto predict these large gusts, along with Naive and Linear Regression algorithms to verify its effectiveness. To utilize these models, thirteen variables\nwere collected on an hourly basis for June-August. Once the models were created and trained, the mean absolute error was calculated for each of the\nmodels as a comparison. Shockingly, it seemed that Linear Regressions and Naive models performed marginally better than the LSTM, which had collapsed to\npredicting a value close to the mean in almost all tests. Accurate wind speeds and direction were still predicted, as the hypothesis says, just with\ndifferent models. With the rarity of upwelling events and their spontaneity, it’s a wonder that the models could predict them in any way.\n")
while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please press enter to continue-----")
print()
print("This file contains all the code for training the Linear Regression models and the Naive models and finding their MAE values. If you would like to take\na look at the LSTM code go to 'Source Code/Unused LSTM'. Also within that folder is some code for a Linear Regression predicition for onshore flags.\nThis file is a cleaned up combined version of all the working source code files running in order.")
print()
print("-----Several times you may be asked 'Yes' or 'No' questions and reply with (y/n) in the terminal-----")
print()
while True:
    reply = input("------Are you ready to start? (You have no choice in this one :D)-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please enter 'y' to continue-----")
print("-----Now we will start 'cleaning' the data------")
print()
print()
print()

print("-----Hello! Welcome to combined_program.ipynb!-----")
print("-----I would like to thank for taking time to use my predition model for Barnegat Bay!-----")
print("-----If you encounter any problems, please let me know!------")
print("-----(This model is only trained for summer (June-August) and will not perform accurately)-----")
print()
print()
print()
print("-----This program will take a csv of your choosing, (make sure it is formatted as specified in the readme)-----")
print("-----return your orginal data (in the same location), a editied version for the model, and a results csv-----")

width = 20
pos = 0
direction = 1

for _ in range(80):
    bar = [" "] * width
    bar[pos] = "#"
    print("\r[" + "".join(bar) + "]", end="", flush=True)
    time.sleep(0.05)
    pos += direction
    if pos == 0 or pos == width - 1:
        direction *= -1


#Read the file and watch for errors
print("-----Please make sure your file is in the 'observed_data' folder and has been renamed 'RAW_data.csv'-----")

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
print("-----Reading all it's goodies....------")

#replace the direction column with the corresponding degree values
direction_map = {'N': 0,'NNE': 1,'NE': 2,'ENE': 3,'E': 4,'ESE': 5,'SE': 6,'SSE': 7,
            'S': 8,'SSW': 9,'SW': 10,'WSW': 11,'W': 12,'WNW': 13,'NW': 14,'NNW': 15
        }

untouched_csv['Direction (A)'] = untouched_csv['Direction (A)'].replace(direction_map).infer_objects(copy = False)
untouched_csv['Direction (A)'] = untouched_csv['Direction (A)'].astype('int')
untouched_csv.infer_objects(copy=False)
print("-----Remapped Directions to Bins...-----")

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
    print("-----Converted air temperatures from Fahrenheit to Celsius...-----")
else:
    print("-----Thanks! Less work for me-----")

reply = input("-----Are your Water Tempuratures in Celsius (y/n)?").strip().lower()
if reply=='n':
    #convert to Celsius
    untouched_csv['Ocean Temp'] = round((untouched_csv['Ocean Temp']-32) * 5.0/9.0, 1)
    untouched_csv['Bay Temp'] = round((untouched_csv['Bay Temp']-32) * 5.0/9.0, 1)
    print("-----Converted water temperatures from Fahrenheit to Celsius...-----")
else:
    print("-----Thanks! Less work for me-----")

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

#determines if its a onshore breeze and adds a new column for it
onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
untouched_csv['Onshore'] = untouched_csv['Direction (A)'].isin(onshore_degrees)
untouched_csv['Onshore'] = untouched_csv['Onshore'].astype(int)
print("-----Determined Onshore Breezes...------")

total = untouched_csv["Onshore"].sum()
print("-----Found ", total, " onshore wind events...-----")

#thresholds
ocean_thresh = 1.0
wind_thresh = 7.0

#average of two lowest ocean points in a day
untouched_csv['ocean_min1'] = untouched_csv['Ocean Temp'].rolling(24).min()
untouched_csv['ocean_min2'] = (
    untouched_csv['Ocean Temp'].rolling(24).apply(lambda x: np.sort(x)[1], raw=False)
)
untouched_csv['ocean_min'] = (untouched_csv['ocean_min1']+untouched_csv['ocean_min2'])/2
untouched_csv['Ocean Temp'] = round(untouched_csv['Ocean Temp'], 1)
print("-----Determined Upwelling...------")

#Check if its a upwelling direction
upwell_wind = [6, 7, 8, 12]
untouched_csv['upwell_wind'] = untouched_csv['Direction (A)'].isin(upwell_wind)
untouched_csv['upwell_wind'] = untouched_csv['upwell_wind'].astype(int)

#bools:
#There are two tide cycles in a day, if you take the two lowest values and average them together. 
#Taking the difference of the low values and the current temperature and if this is bigger than the threshold the Ocean boolean is true. 
untouched_csv['big_wind'] = (untouched_csv['Wind Speed (A)'] > wind_thresh).astype(int)
untouched_csv['upwell_wind'] = (untouched_csv['upwell_wind'] == 1).astype(int)
untouched_csv['ocean_bool'] = ((untouched_csv['ocean_min'] - untouched_csv['Ocean Temp']) > ocean_thresh).astype(int)
untouched_csv["upwelling_flag"] = ((untouched_csv['ocean_bool'] == 1) & (untouched_csv['big_wind'] == 1)).astype(int)
print("-----Checking my answers...------")

total = untouched_csv["upwelling_flag"].sum()
print("-----Found ", total, " upwelling events...-----")

#drop the colomns
untouched_csv.drop(['ocean_bool', 'big_wind', 'ocean_min1', 'ocean_min2', 'upwell_wind', 'ocean_min'], axis=1, inplace=True)
untouched_csv = untouched_csv.dropna(how="all").reset_index(drop=True)
print("-----Fixing typos....-----")

#Save this data frame to a Finished csv
#Removes first 24 hrs of data to remove NA's
untouched_csv = untouched_csv[24:]
untouched_csv.to_csv('Csv/CLEAN.csv', index=False)



print()
print()
print()
print("-----Hazah! You've made it thus far. Your data has been cleaned for the models-----")
print()
print()
print()


#Opens a new dataframe with the Clean csv
cleancsv = pd.read_csv('Csv/CLEAN.csv')
print("-----Opened our cleaned up csv...-----")
print("-----Gobbling up all the tasty data....-----")

#Convert data into Date time and create date filter
cleancsv['Date'] = pd.to_datetime(cleancsv['Date'])
cleancsv['Date'] = cleancsv['Date'] + pd.to_timedelta(cleancsv["Hr"], unit="h")
cleancsv.drop('Hr', axis=1, inplace=True)
print("-----Converted Date and Hour to DateTime...------")

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

#Create month colomn and restrict to only summer months
summer_mask = cleancsv["Date"].dt.month.isin([6, 7, 8, 9])
cleancsv = cleancsv[summer_mask].reset_index(drop=True)
print("----Making sure your data is summer only...-----")



print()
print()
print()
print("-----Now the fun part! Model time! With your results you also get a four course meal of models with a side of Numpy(pie)-----")



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

print()
print()
print()
print("-----Model One: Wind Speed-----")


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

#Save output data into variables and reshape it to be a 1d array
output_data = data_wind_speed.values
output_data = np.array(output_data).reshape(-1, 1)
print("-----Reshaping our output to a 1D array...-----")

#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.90)) #Use 90% of training data
print("-----Setting the length of the training data...-----")

print("-----Setting the length of the training data...-----")

#Scaler
scaler_x= StandardScaler()
scaler_y= StandardScaler()


scaledx = scaler_x.fit_transform(dataset)
scaledy = scaler_y.fit_transform(output_data)
print("-----Scaling with love...-----")

#Setting to all
X_all = scaledx          # (N, 12)
y_all = output_data
print("-----Scaling with everything in mind...-----")

#Train's and test
X_train = X_all[:training_data_len]
y_train = y_all[:training_data_len]
X_test  = X_all[training_data_len:]
y_test  = y_all[training_data_len:]
print("-----Chooo-chooooo! Setting up our X/y trains-----")
print("-----Don't forget about our X/y tests-----")

#Open the model
reg_speed = load("models/wind_speed_linear.joblib")
print("-----Opening up the model...-----")

#Predict
speed_pred_lr = reg_speed.predict(X_test)
speed_pred_lr = np.maximum(speed_pred_lr, 0.0)
speed_pred_lr = np.round(speed_pred_lr, 1)
print("-----Doing the math...-----")
print("-----It's a lot of math okay?-----")

#MAE
speed_mae_lr = mean_absolute_error(y_test, speed_pred_lr)
print("-----Checking my work...-----")
print("-----Linear MAE of:", speed_mae_lr, "-----")

# attach to dataframe for export
speed_linear = cleancsv.iloc[training_data_len:training_data_len + len(speed_pred_lr)].copy()
speed_linear["wind_speed_pred_linear"] = speed_pred_lr
speed_linear.to_csv("Csv/predictions/predicted_wind_speed.csv", index=False)
print("----Typing up a essay of data for you...-----")



print("-----YES! All saved. Moving on....-----")
print()
print()
print()
print("-----Model Two: Wind Gust Speed-----")


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
#Save output data into variables and reshape it to be a 2d array
output_data = data_gusting.values
output_data = np.array(output_data).reshape(-1, 1)
print("-----Reshaping our output to a 1D array...-----")

#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.90)) #Use 90% of training data
print("-----Setting the length of the training data...-----")

#Scaler
scaler_x= StandardScaler()
scaler_y= StandardScaler()


scaledx = scaler_x.fit_transform(dataset)
scaledy = scaler_y.fit_transform(output_data)
print("-----Scaling with love...-----")

#Setting to all
X_all = scaledx          # (N, 12)
y_all = output_data
print("-----Scaling with everything in mind...-----")

#Train's and test
X_train = X_all[:training_data_len]
y_train = y_all[:training_data_len]
X_test  = X_all[training_data_len:]
y_test  = y_all[training_data_len:]
print("-----Chooo-chooooo! Setting up our X/y trains-----")
print("-----Don't forget about our X/y tests-----")

#Open the model
reg_gust = load("models/wind_gust_linear.joblib")
print("-----Opening up the model...-----")

#Predict
gust_pred_lr = reg_gust.predict(X_test)
gust_pred_lr = np.maximum(gust_pred_lr, 0.0)
gust_pred_lr = np.round(gust_pred_lr, 1)
print("-----Doing the math...-----")
print("-----It's a lot of math okay?-----")

#MAE
gust_mae_lr = mean_absolute_error(y_test, gust_pred_lr)
print("-----Checking my work...-----")
print("-----Linear MAE of:", gust_mae_lr, "-----")

# attach to dataframe for export
linear_df = cleancsv.iloc[training_data_len:training_data_len + len(gust_pred_lr)].copy()
linear_df["wind_gust_pred_linear"] = gust_pred_lr
linear_df.to_csv("Csv/predictions/predicted_wind_gust.csv", index=False)



print("-----YES! All saved. Moving on....-----")
print()
print()
print()
print("-----Model Three: Wind Direction-----")

#Create Naive model
wd = cleancsv['Direction (A)'].values.astype(int)
print("-----Creating a Naive model...-----")

#Train the model
direction_pred = wd[training_data_len-1:-1]
direction_true = wd[training_data_len:]
print("-----Teaching the model a few tricks...-----")   

#Stop negatives and round to X.X
direction_pred_pred = np.maximum(direction_pred, 0.0)
print("-----Stopping negatives and rounding...-----")

#Find MAE
direction_mae_naive = mean_absolute_error(direction_true, direction_pred_pred)
print("-----Checking my work...-----")
print("-----Naive MAE of:", direction_mae_naive, "-----")


print("-----YES! All saved. Moving on....-----")
print()
print()
print()
print("-----Model Four: Onshore Breezes-----")



#Save onshore to variable and check if the predicted direction is onshore
onshore_degrees = [8, 6, 7, 4, 3, 2, 1]
onshore_pred_flag = np.isin(direction_pred, onshore_degrees).astype(int)
print("-----Determining onshore breezes...-----")

#Find MAE of onshore
onshore_check = cleancsv['Onshore'].iloc[training_data_len:].values.astype(int)
mae_onshore = mean_absolute_error(onshore_check, onshore_pred_flag)
print("-----Onshore flag MAE (naive-direction):", mae_onshore, "-----")

#Save to csv
naive_df = cleancsv.iloc[training_data_len:].copy()
naive_df["wind_direction_pred_naive"] = direction_pred
naive_df["onshore_pred_direction"] = onshore_pred_flag
naive_df.to_csv("Csv/predictions/predicted_direction.csv", index=False)


print("-----YES! All saved. Moving on....-----")
print()
print()
print()
print("-----Model Five: Upwelling-----")
print("-----This model uses less data for training and more for testing due to lack of later upwelling events-----") 



#Length of training data
training_data_len = int(np.ceil(len(dataset) * 0.60)) #Use 90% of training data
print("-----Setting the length of the training data...-----")

#Naive model creation
up = cleancsv['upwelling_flag'].values
print("-----Creating a Naive model...-----")   

#Fitting the model
upwell_pred = up[training_data_len-1:-1]
upwell_true = up[training_data_len:]

#Calculate MAE
upwell_mae_naive = mean_absolute_error(upwell_true, upwell_pred)
print("-----Checking my work...-----")
print("-----Naive MAE of:", upwell_mae_naive, "-----")

#Save to csv
upwell_df = cleancsv.iloc[training_data_len:].copy()
upwell_df["upwelling_pred_naive"] = upwell_pred
upwell_df["upwelling_pred_naive"] = upwell_df["upwelling_pred_naive"].astype(int)
upwell_df.to_csv("Csv/predictions/predicted_upwelling.csv", index=False)

print("-----YES! All saved. Moving on....-----")
print()
print()
print()
print("-----Whew! That was a lot! But we are almost there! Just a few more steps-----")


#Load clean csv to use to merge data
clean = pd.read_csv("Csv/CLEAN.csv")
clean["Date"] = pd.to_datetime(clean["Date"]) + pd.to_timedelta(clean["Hr"], unit="h")
clean = clean.drop(columns=["Hr"])
print("-----Opening your csv...-----")
print("-----Time is nothing...But thats why we have Datetime...-----")

#Data frames for every csv
pred_direction = pd.read_csv('Csv/predictions/predicted_direction.csv', sep=',')
pred_upwelling = pd.read_csv('Csv/predictions/predicted_upwelling.csv', sep=',')
pred_wind_gust = pd.read_csv('Csv/predictions/predicted_wind_gust.csv', sep=',')
pred_wind_speed = pd.read_csv('Csv/predictions/predicted_wind_speed.csv', sep=',')

for df in [pred_direction, pred_upwelling, pred_wind_gust, pred_wind_speed]:
    df["Date"] = pd.to_datetime(df["Date"])
    print("-----Opening even more csvs...-----")

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
print("-----Merging all the csvs together...-----")
print("-----Merging all the csvs together...-----")
print("-----Merging all the csvs together...-----")
print("-----Merging all the csvs together...-----")
print("-----Alright that's enough of that-----")

#Sort by date
merged = merged.sort_values("Date").reset_index(drop=True)

#Save to csv
merged.to_csv("Csv/predictions/all_predictions.csv", index=False)
print("-----Saving the final csv with all the predictions...-----")

with open("Csv/predictions/mae_report.txt", "w") as f:
    f.write(f"Wind Speed Linear MAE: {speed_mae_lr:.6f}\n")
    f.write(f"Wind Gust Linear MAE: {gust_mae_lr:.6f}\n")
    f.write(f"Upwelling Naive MAE: {upwell_mae_naive:.6f}\n")
    f.write(f"Wind Direction Naive MAE: {direction_mae_naive:.6f}\n")
    f.write(f"Onshore Breeze from Direction MAE: {mae_onshore:.6f}\n")
    
print("---------")
print("---------")
print("---------")
print("-----Huh. That's it. If you would like to do some more predicting, you know where to go-----"    )
print("-----You can find your 'all_predictions.csv' csv with all the predictions in the 'predictions' folder as well as all the MAE values saved to a 'mae_report.txt' file in the 'predictions' folder-----")  
