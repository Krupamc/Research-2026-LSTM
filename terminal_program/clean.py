from datetime import datetime
import os
import pandas as pd
import numpy as np

#The dataframe
ask = input("What is the file path of your csv?")
untouched_csv = pd.read_csv(ask, sep=',')

#replace the direction column with the corresponding degree values
direction_map = {'N': 0,'NNE': 22.5,'NE': 45,'ENE': 67.5,'E': 90,'ESE': 112.5,'SE': 135,'SSE': 157.5,
            'S': 180,'SSW': 202.5,'SW': 225,'WSW': 247.5,'W': 270,'WNW': 292.5,'NW': 315,'NNW': 337.5
        }

untouched_csv['Direction (A)'] = untouched_csv['Direction (A)'].replace(direction_map).infer_objects(copy = False)
untouched_csv.infer_objects(copy=False)

#convert to Celsius
untouched_csv['Mainland Air Temp'] = round((untouched_csv['Mainland Air Temp']-32) * 5.0/9.0, 1)
untouched_csv['LBI Air Temp'] = round((untouched_csv['LBI Air Temp']-32) * 5.0/9.0, 1)

#rounds all colomns
untouched_csv['Humidity (%)'] = round(untouched_csv['Humidity (%)'], 1)
untouched_csv['Wind Speed (A)'] = round(untouched_csv['Wind Speed (A)'], 1)
untouched_csv['Gusting'] = round(untouched_csv['Gusting'], 1)
untouched_csv['Atmospheric Pressure (IN)'] = round(untouched_csv['Atmospheric Pressure (IN)'], 2)
untouched_csv['Precipitation Rate'] = round(untouched_csv['Precipitation Rate'], 2)
untouched_csv['Bay Temp'] = round(untouched_csv['Bay Temp'], 2)
untouched_csv['Salinity'] = round(untouched_csv['Salinity'], 2)
untouched_csv['LBI Air Temp'] = round(untouched_csv['LBI Air Temp'], 1)
untouched_csv['Ocean Temp'] = round(untouched_csv['Ocean Temp'], 1)

#determines if its a onshore breeze and adds a new column for it
onshore_degrees = [180, 135, 157.5, 90, 67.5, 45, 22.5]
untouched_csv['Onshore'] = untouched_csv['Direction (A)'].isin(onshore_degrees)
untouched_csv['Onshore'] = untouched_csv['Onshore'].astype(int)

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

#Check if its a upwelling direction
upwell_wind = [135, 165, 180, 270]
untouched_csv['upwell_wind'] = untouched_csv['Direction (A)'].isin(upwell_wind)
untouched_csv['upwell_wind'] = untouched_csv['upwell_wind'].astype(int)

#bools:
#   There are two tide cycles in a day, if you take the two lowest values and average them together. 
#   Taking the absolute value of the difference of the low values and the current temperature and if this is bigger than the threshold the Ocean boolean is true. 
untouched_csv['big_wind'] = (untouched_csv['Wind Speed (A)'] > wind_thresh).astype(int)
untouched_csv['upwell_wind'] = (untouched_csv['upwell_wind'] == 1).astype(int)
untouched_csv['ocean_bool'] = (abs(untouched_csv['ocean_min'] - untouched_csv['Ocean Temp']) > ocean_thresh).astype(int)
untouched_csv["upwelling_flag"] = ((untouched_csv['ocean_bool'] == 1) & (untouched_csv['big_wind'] == 1)).astype(int)

#drop the colomns
untouched_csv.drop(['ocean_bool', 'big_wind', 'ocean_min1', 'ocean_min2', 'upwell_wind', 'ocean_min'], axis=1, inplace=True)
#Save this data frame to a Finished csv
#Removes first 24 hrs of data to remove NA's
untouched_csv = untouched_csv[24:]
untouched_csv.to_csv('Csv/CLEAN.csv', index=False)

print("wprked")