
from direction_to_degrees import direction_to_degrees

path = "C://Users/Krupam/downloads/data.csv"
degree = direction_to_degrees(path)
degree.convertdd()
degree.onshore()

#ask users if air temps are in Celsius or Fahrenheit
degree.f_to_c("Mainland Air Temp")       # Convert temperatures from Fahrenheit to Celsius
degree.f_to_c("LBI Air Temp")       # Convert temperatures from Fahrenheit to Celsius



print("")
print("Conversion and onshore flagging complete.")
print("")
