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

    def upwelling(self):
        # Mark upwelling events (1 = upwelling, 0 = no upwelling)
        upwelling_directions = [180, 157.5, 135]  # S, SSE, SE
        self.df[self.upwelling_column] = 0
        
        
        


