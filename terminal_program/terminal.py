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
print("South Barnegat Bay Onshore Wind Model Prediction with the Use of Long Short-Term Memory Neural Networks - Terminal Program for Real-Time Predictions\n")
print("Written by ----- \nWritten in Python 3.11.14\n")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("Damp South and Western onshore winds initiate the summertime event known as upwelling.\nUpwelling in small, localized areas like Barnegat Bay has a significant impact on bay temperature, creating a large land-sea temperature difference.\nThis difference can lead to harsh and fast onshore breezes that can “swamp” small watercraft. This study uses an LSTM, a deep learning neural network,\nto predict these large gusts, along with Naive and Linear Regression algorithms to verify its effectiveness. To utilize these models, thirteen variables\nwere collected on an hourly basis for June-August. Once the models were created and trained, the mean absolute error was calculated for each of the\nmodels as a comparison. Shockingly, it seemed that Linear Regressions and Naive models performed marginally better than the LSTM, which had collapsed to\npredicting a value close to the mean in almost all tests. Accurate wind speeds and direction were still predicted, as the hypothesis says, just with\ndifferent models. With the rarity of upwelling events and their spontaneity, it’s a wonder that the models could predict them in any way.\n")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("-----Hello! Welcome to terminal.py!-----")
print("-----I would like to thank for taking time to use my predition model for Barnegat Bay!-----")
print("-----If you encounter any problems, please let me know!------")
print("-----(This model is only trained for summer (June-August) and will not perform if given data outside this period)-----")
print()

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in (""):
        break
    print("-----Please press enter to continue-----")
print()
print()
print()

print("-----This program will ask for the weather conditions, and predict the next hour for you and save the results in a results file-----")


print("-----Several times you may be asked 'Yes' or 'No' questions and reply with (y/n) in the terminal-----")
print()
print()
print()
while True:
    reply = input("------Are you ready to start? (You have no choice in this one :D)-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please enter 'y' to continue-----")
