import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import seaborn as sb


# item 1 : dataset loading and preview
data = pd.read_csv("./forestfires.csv")
x = data[["X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind", "rain"]]
y = data[["area"]]

# item 4 : dataset splitting
x_train, x_test, y_train, y_test = tts(x, y, random_state=104, test_size=0.30, shuffle=True)