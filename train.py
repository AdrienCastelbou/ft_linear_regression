import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from mylinearregression import MyLinearRegression as MyLR

def save_model(model):
    file = open('model.pickle', 'wb')
    pickle.dump(model, file)
    file.close()

def normalize(x):
    norm_x = np.array([])
    for col in x.T:
        mean_col = np.mean(col)
        std_col = np.std(col)
        n_col = ((col - mean_col) / std_col).reshape((-1, 1))
        if norm_x.shape == (0,):
            norm_x = n_col
        else:
            norm_x = np.hstack((norm_x, n_col))
    return norm_x

def load_dataset():
    content = pd.read_csv("data.csv")
    x = np.array(content[["km"]])    
    y = np.array(content[["price"]])
    if x.shape[1] !=  1 or y.shape[1] !=  1:
        raise Exception("Datas are missing in data.csv")   
    return x, y

def vizualize(x, y, y_hat):
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    plt.show()

def unormalize_thetas(n_thetas, x, y):
    x_std = np.std(x)
    y_std = np.std(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    thetas = np.zeros(n_thetas.shape)
    thetas[0] = y_std * n_thetas[0] + y_mean - (y_std / x_std) * n_thetas[1] * x_mean
    thetas[1] = (y_std / x_std) * n_thetas[1]
    return thetas

def train_model(x, y):
    x_train = normalize(x)
    y_train = normalize(y)
    thetas = np.random.rand(x.shape[1] + 1, 1).reshape(-1, 1)
    myLR = MyLR(thetas=thetas, max_iter=100000)
    myLR.fit_(x_train, y_train)
    myLR.unormalize_thetas(x, y)
    return myLR

def main():
    try:
        x, y = load_dataset()
    except Exception as e:
        print("Error in data loading : ", e)
    model = train_model(x, y)
    save_model(model)
    


if __name__ == "__main__":
    main()