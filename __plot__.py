import pandas as pd
import os
import matplotlib.pyplot as plt


def plot():
    dir = os.listdir("logs")
    dir.sort()
    latestfile = "logs/" + dir[-1]

    data = pd.read_csv(latestfile,
                       names=["Time",
                              "Reward",
                              "Loss"]
                       )

    plt.subplot(3, 1, 1)
    plt.plot(data["Time"])

    plt.subplot(3, 1, 2)
    plt.plot(data["Reward"])

    plt.subplot(3, 1, 3)
    plt.plot(data["Loss"])

    plt.show()


if __name__ == "__main__":
    plot()

