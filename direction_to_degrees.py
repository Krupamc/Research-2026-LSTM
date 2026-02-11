import pandas as pd
import numpy as np

class direction_to_degrees:
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
   
   #initializes the class with the path to the CSV file and the name of the column containing the direction values (default is "Direction (A)")
   def __init__(self, csv_path, directions_column="Direction (A)", onshore_column="Onshore", air_temp_column="Mainland Air Temp", island_temp_column="LBI Air Temp"):
      self.csv_path = csv_path
      self.directions_column = directions_column
      self.onshore_column = onshore_column
      self.air_temp_column = air_temp_column
      self.df = pd.read_csv(self.csv_path)
   
   #replaces the direction values with their corresponding degree values and saves the modified DataFrame back to the CSV file
   def convertdd(self):
      self.df[self.directions_column] = self.df[self.directions_column].replace(self.direction_map)
      self.df[self.onshore_column] = self.df[self.directions_column]
      self.df.to_csv(self.csv_path, index=False)

   def f_to_c(self, column):
      self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
      self.df[column] = (self.df[column] - 32) * 5.0 / 9.0
      self.df[column] = self.df[column].round(1)
      self.df.to_csv(self.csv_path, index=False)


   #replaces the degree values in the onshore column with 1 for onshore directions
   def onshore(self):
      
      #replaces the degree values in the onshore column with 0 for offshore directions
       onshore_degrees = [180, 135, 157.5, 90, 67.5, 45, 22.5]
         #S, SE, SSE, E, ENE, NE, NNE

       # set to 1 where the other column has an onshore degree
       self.df.loc[
       self.df[self.directions_column].isin(onshore_degrees),
       self.onshore_column] = 1
       self.df.to_csv(self.csv_path, index=False)