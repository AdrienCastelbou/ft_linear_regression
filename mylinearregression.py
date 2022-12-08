import numpy as np

def arg_checker(fnc):
    def ckeck_args(*args, **kwargs):
        for arg in args:
            if type(arg) == MyLinearRegression:
                if type(arg.alpha) != float or type(arg.max_iter) != int or type(arg.thetas) != np.ndarray:
                    print("Bad params for your model")
                    return None
            else:
                if type(arg) != np.ndarray:
                    print(f"Bad param for {fnc.__name__}")
                    return None
        return fnc(*args, **kwargs)
    return ckeck_args

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        try:
            if type(thetas) != np.ndarray or type(alpha) != float or type(max_iter) != int:
                return None
            self.alpha = alpha
            self.max_iter = max_iter
            self.thetas = thetas.astype(float)
        except:
            return None


    def unormalize_thetas(self, x, y):
        try:
            x_std = np.std(x)
            y_std = np.std(y)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            thetas = np.zeros(self.thetas.shape)
            thetas[0] = y_std * self.thetas[0] + y_mean - (y_std / x_std) * self.thetas[1] * x_mean
            thetas[1:] = (y_std / x_std) * self.thetas[1:]
            self.thetas = thetas
        except:
            return None


    @arg_checker
    def gradient(self, x, y):
        try:
            l = len(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            nabla_J = x.T.dot(x.dot(self.thetas) - y) / l
            return nabla_J
        except:
            return None


    @arg_checker
    def fit_(self, x, y):
        try:
            for i in range(self.max_iter):
                nabla_J = self.gradient(x, y)
                self.thetas -= self.alpha * nabla_J
            return self.thetas
        except:
            return None
    
    @arg_checker
    def predict_(self, x):
        try:
            if not len(x) or not len(self.thetas):
                return None
            extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
            return extended_x.dot(self.thetas)
        except:
            return None

    @arg_checker
    def loss_elem_(self, y, y_hat):
        try:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                return None
            return (y_hat - y) ** 2
        except:
            return None

    @arg_checker
    def loss_(self, y, y_hat):
        try:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                return None
            return float(1 / (2 * y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
        except:
            return None

    @staticmethod
    @arg_checker
    def mse_(y, y_hat):
        try:
            return float(1 / (y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
        except:
            return None
