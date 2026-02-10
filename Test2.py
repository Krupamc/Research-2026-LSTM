# Test.py
from onshoreAndUpwelling import DirectionToDegree

path = "C://Users/Krupam/downloads/data.csv"

degree = DirectionToDegree(
    path,
    directions_column="Direction (A)",
    onshore_column="Onshore",
    upwelling_column="Upwelling",
    bay_temp_column="Bay Temp",
    ocean_temp_column="Ocean Temp",
    wind_speed_column="Wind Speed (A)"
)

# Run all transformations in order
degree.convert()       # Convert N, NE, etc. to degrees
degree.onshore()       # Mark onshore winds
degree.f_to_c("Mainland Air Temp")       # Convert temperatures from Fahrenheit to Celsius
degree.f_to_c("LBI Air Temp")     # Convert temperatures from Fahrenheit to Celsius
degree.upwelling()     # Mark upwelling events


print("Processing complete! Check your CSV.")
