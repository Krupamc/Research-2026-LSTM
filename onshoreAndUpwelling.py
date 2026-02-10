# direction_to_degree.py
import pandas as pd
import numpy as np

class DirectionToDegree:
    direction_map = {
        'N': 0,
        'NNE': 22.5,
        'NE': 45,
        'ENE': 67.5,
        'E': 90,
        'ESE': 112.5,
        'SE': 135,
        'SSE': 157.5,
        'S': 180,
        'SSW': 202.5,
        'SW': 225,
        'WSW': 247.5,
        'W': 270,
        'WNW': 292.5,
        'NW': 315,
        'NNW': 337.5
    }
   
    def __init__(self, csv_path, directions_column="Direction (A)", onshore_column="Onshore", upwelling_column="Upwelling", bay_temp_column="Bay Temp", ocean_temp_column="Ocean Temp", wind_speed_column="Wind Speed (A)"):
        self.csv_path = csv_path
        self.directions_column = directions_column
        self.onshore_column = onshore_column
        self.upwelling_column = upwelling_column
        self.bay_temp_column = bay_temp_column
        self.ocean_temp_column = ocean_temp_column
        self.wind_speed_column = wind_speed_column
        self.df = pd.read_csv(self.csv_path)
   
    def convert(self):
        #Replace direction values with their corresponding degree values.
        self.df[self.directions_column] = self.df[self.directions_column].replace(self.direction_map)
        self.df.to_csv(self.csv_path, index=False)

    def onshore(self):
        #Mark onshore wind directions (1 = onshore, 0 = offshore)
        # Winds from S, SE, SSE, E, ENE, NE, NNE are onshore for your coast
        onshore_degrees = [180, 157.5, 135, 112.5, 90, 67.5, 45, 22.5]
        
        # Start with all 0s
        self.df[self.onshore_column] = 0
        
        # Set to 1 where direction column has an onshore degree
        self.df.loc[
            self.df[self.directions_column].isin(onshore_degrees), self.onshore_column] = 1
        self.df.to_csv(self.csv_path, index=False)

    def f_to_c(self, column):
      self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
      self.df[column] = (self.df[column] - 32) * 5.0 / 9.0
      self.df[column] = self.df[column].round(1)
      self.df.to_csv(self.csv_path, index=False)

    def upwelling(self, temp_drop_threshold=0.5, ocean_bay_diff_threshold=0.5, wind_speed_threshold=8.0):
        """
        Mark upwelling events based on:
        1. Wind from S, SE, or SSE
        2. Bay temperature dropping below 24-hour average
        3. Ocean cooler than bay
        4. Wind speed above threshold
        
        Parameters:
        - temp_drop_threshold: How much bay temp must drop below 24h average (°F)
        - ocean_bay_diff_threshold: How much colder ocean must be than bay (°F)
        - wind_speed_threshold: Minimum wind speed (mph)
        """
        
        # Define upwelling-favorable wind directions (S, SSE, SE)
        upwelling_directions = [180, 157.5, 135]
        
        # Calculate rolling 24-hour bay temperature average
        self.df['bay_temp_24h_avg'] = self.df[self.bay_temp_column].rolling(
            window=24, 
            min_periods=1
        ).mean()
        
        # Calculate how much bay temp has dropped from 24h average
        self.df['bay_temp_drop'] = self.df['bay_temp_24h_avg'] - self.df[self.bay_temp_column]
        
        # Calculate ocean-bay temperature difference
        self.df['ocean_bay_diff'] = self.df[self.bay_temp_column] - self.df[self.ocean_temp_column]
        
        # Initialize upwelling column with 0s
        self.df[self.upwelling_column] = 0
        
        # Mark upwelling (1) where all conditions are met
        upwelling_mask = (
            self.df[self.directions_column].isin(upwelling_directions) &
            (self.df['bay_temp_drop'] >= temp_drop_threshold) &
            (self.df['ocean_bay_diff'] >= ocean_bay_diff_threshold) &
            (self.df[self.wind_speed_column] >= wind_speed_threshold)
        )
        
        self.df.loc[upwelling_mask, self.upwelling_column] = 1
        
        # Clean up temporary columns (optional - remove if you want to keep them for analysis)
        #self.df.drop(['bay_temp_24h_avg', 'bay_temp_drop', 'ocean_bay_diff'], axis=1, inplace=True)
        self.df.to_csv(self.csv_path, index=False)

