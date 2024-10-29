# This script plots the precipitation amount over time of each cleaned pickle file

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the relative path to the pickles directory
pickle_dir = os.path.join(os.path.dirname(__file__), '..', 'pickles')
# Concat all pickles
df_list = []
#files = os.listdir(os.path.join("", "data\\pickles"))
files = os.listdir(pickle_dir)

for file in files:
    target_file = os.path.join(pickle_dir, file)
    df = pd.read_pickle(target_file)
    df_list.append(df)

# print(df.head())
feature_list = ['Date/Time (LST)','Temp (degC)','Rel Hum (%)','Precip. Amount (mm)','Stn Press (kPa)','Wind Spd (km/h)']
feature = feature_list[1]
stations = [30165,48374,49608]

# Plot
nb_timesteps = 24*14
for index in range(3):
    plt.plot(df_list[index].index[:nb_timesteps], df_list[index][feature][:nb_timesteps], label=f'Station {stations[index]}')
plt.xlabel('Time steps (h)')
plt.ylabel(feature)
plt.title(feature + ' Over Time')
plt.legend()
plt.show()