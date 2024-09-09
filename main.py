import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import seaborn as sb

# we're mapping days and months to their corresponding numeric values. not using 0 cause multiplying by 0 is a big nono
day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

# item 2: non linear regression
def h(x, w):
    return np.dot(x, w.T) # returns a column vector

def error(x, w, y, lamb, reg ="none"):
    if(reg == "lasso"):
        return (np.linalg.norm(y - h(x,w)) ** 2) / (2 * len(x)) # todo: add lasso regularization
    elif(reg == "ridge"):
        return (np.linalg.norm(y - h(x,w)) ** 2) / (2 * len(x)) # todo: add ridge regularization
    elif(reg == "none"):
        return (np.linalg.norm(y - h(x,w)) ** 2) / (2 * len(x))
    else:
        raise Exception("error function did not recognize the regularization type " + reg + ".\n Did you mean to use lasso, ridge, or none?")
# todo: fix this thing, everything is nan

def derivative(x, w, y, lamb):
    diff = y - h(x, w) # col - col
    num = np.dot(diff.T, -x) # numerator
    den = len(y) # denominator
    reg = 2 * lamb * w
    return num/den + reg/den

def update(w, d_w, alpha):
    return w - alpha * d_w # alpha measures how much the model reacts to each derivative change

def linear_train(x, y, e, alpha, lamb, debug):
    np.random.seed(2001)

    # notes:
    # shape[0] = number of rows
    # shape[1] = number of columns
    # at first, x has 12 features, so there's 12 columns
    # as we want to have a bias, we add a column of 1s to x. now x has 13 features (the first one is the bias)

    # because w uses the n° of columns in x to build itself, it now has a new element at the start, w0 (the bias)
    # the bias is the first element of w, because it is always multiplied by 1 (from the first added column in x)

    x = np.hstack( (np.ones((x.shape[0], 1)), x)) # adds a first column of 0s, so it can be multiplied with the transposed of w
    w = np.random.rand(1, x.shape[1]) # w is a row vector with as many rows as columns in x. its values are random at first
    loss = [] # array of loss at every iteration
    for i in range(e):
        d_w = derivative(x, w, y, lamb)
        w = update(w, d_w, alpha)
        l = error(x, w, y, lamb, "ridge")
        loss.append(l)
        if (i % debug == 0): 
            print(f"Error at iteration {i}: {l}") # prints the error every "debug" iterations

    return w, loss 

def linear_init():
    # item 1 : dataset loading and preview
    data = pd.read_csv("./forestfires.csv")
    x = data[["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]]
    y = data[["area"]]
    # preview pending lmao

    # map strings to numeric values
    x['day'] = x['day'].map(day_mapping)
    x['month'] = x['month'].map(month_mapping)

    # conversion to np arrays because python is awful
    x = np.array(x)
    y = np.array(y)

    # normalization
    for i in range(x.shape[1]):
        col = x[:, i]  # Get the column
        col = (np.min(col) - col) / (np.max(col) - np.min(col))  # Apply the transformation
        x[:, i] = col  # Update the column in the original matrix
    y  = (min(y) - y)/(max(y) - min(y))

    # item 4 : dataset splitting
    x_train, x_test, y_train, y_test = tts(x, y, random_state=104, test_size=0.30, shuffle=True)

    # item 2: non linear regression (training)
    w, l = linear_train(x, y, 10000, 0.01, 0.1, 500)
    # trains according to features in x and results in y, does 10000 iterations, alpha is 0.9, lambda is 0, prints error every 500 iterations

    plt.plot(l)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Over Iterations')
    plt.show()

linear_init()


# todo: nonlinear shenanigans