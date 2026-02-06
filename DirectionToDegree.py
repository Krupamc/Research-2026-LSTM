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
   
   def __init__(self, csv_path, directions_column="Direction (A)"):
      self.csv_path = csv_path
      self.directions_column = directions_column
      self.df = pd.read_csv(self.csv_path)
   
   def convert(self):
      self.df[self.directions_column] = self.df[self.directions_column].replace(self.direction_map)
      self.df.to_csv(self.csv_path, index=False)
      