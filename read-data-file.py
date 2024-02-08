import numpy as np
import matplotlib.pyplot as plt

data = np.load("training-data.npz", allow_pickle=True)

fig, ax = plt.subplots()

for pos, data_point in enumerate(data['out_x']):
    ax.clear()
    ax.set_title(f"The label for data is {pos}")
    ax.plot(data_point)
    plt.pause(0.5)

plt.show()
