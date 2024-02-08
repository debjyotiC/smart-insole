from os import listdir, path
import pandas as pd
import numpy as np

mean_data = []
label_data = []

data_set_path = "data/insole-pressure"

files = listdir(data_set_path)

for file in files:
    full_file_path = path.join(data_set_path, file)
    mean_sensors = pd.read_csv(full_file_path, index_col=False)["Mean"].values.tolist()[:56]
    mean_data.append(mean_sensors)
    label_data.append(file.split(".")[0])

mean_sensor_data = np.array(mean_data)
mean_sensor_label = np.array(label_data)

np.savez("training-data.npz", out_x=mean_sensor_data, out_y=mean_sensor_label)
