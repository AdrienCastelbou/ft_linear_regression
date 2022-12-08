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


def make_prediction(model):
    try:
        mileage = float(input("Enter a mileage 🚗 : "))
        x = np.array(mileage).reshape(-1, 1)
        y_hat = model.predict_(x)
        print(f"The price of your car, with a mileage of {mileage}, can be estimated at {y_hat[0][0]} 🤑")
    except ValueError:
        print("Please enter a valid mileage")
    except Exception as e:
        print("An error occured")


def main():
    model = load_model()
    make_prediction(model)
    pass

if __name__ == "__main__":
    main()