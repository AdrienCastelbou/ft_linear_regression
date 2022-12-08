import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from mylinearregression import MyLinearRegression as MyLR

def load_model():
    try:
        file = open('model.pickle', 'rb')
        model = pickle.load(file)
        file.close()
    except Exception as e:
        print("No model found, default model will be used")
        thetas = np.zeros((2, 1))
        model = MyLR(thetas=thetas)
    return model

def load_dataset():
    content = pd.read_csv("data.csv")
    x = np.array(content[["km"]])    
    y = np.array(content[["price"]])
    if x.shape[1] !=  1 or y.shape[1] !=  1:
        raise Exception("Datas are missing in data.csv")   
    return x, y

def evaluate_model(model, x, y):
    y_hat = model.predict_(x)
    model_mse = MyLR.mse_(y=y, y_hat=y_hat)
    print(f"The avergage squared distance between our predictions and the real values, aka the \033[95mMSE\033[0m, is equal to \033[92m {model_mse} \033[0m.")

def main():
    try:
        x, y = load_dataset()
    except Exception as e:
        print("Error in data loading : ", e)
    model = load_model()
    evaluate_model(model, x, y)
    pass

if __name__ == "__main__":
    main()