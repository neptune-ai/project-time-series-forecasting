import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"


path = "../data/"

print(os.listdir(path))