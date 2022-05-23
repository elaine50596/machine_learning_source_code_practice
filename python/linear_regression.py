## Source Code Practice for linear regression
# Objective: estimate the coefficients from linear regression
# method 1: ols: gradient descent using analytical gradient
# method 2: ols: close-form calculation from matrix arithmetic
# method 3: maximum likelihood estimation using numerical optimization (numerical gradient)
# notes: sample size might be too small, estimation might be biased
# hyperparameter: learning_rate - worth tuning via cross validation
import numpy as np
x_train = np.array([[0.1, 0.2, 1],
                    [0.2, 1, 0.5],
                    [2, 0.2, 2],
                    [0.5, 1, 0.5]]) 

target_beta = np.array([2.5, 1.5, -2, 3])
scale = 0.5

all_ones = np.ones((x_train.shape[0],1))
x_matrix = np.hstack((all_ones, x_train))
y_train = np.matmul(x_matrix, target_beta) + np.random.normal(0, scale)

# below class covers method 1 and method 2
class linear_regression:
    def __init__(self) -> None:
        pass
    
    def initialize_beta(self, x):
        # beta: all zeros (inclusive of intercept beta0)
        return np.array([0]*(1+x.shape[1]))

    def evaluate_gradient(self, x, y, yhat):  
        error = y-yhat   
        gradient_beta_0 = -np.mean(2*error)
        gradient_beta_x=list(np.apply_along_axis(np.mean, axis = 1, arr = -x.T*(2*error)))
        gradient_beta = np.array([gradient_beta_0] + gradient_beta_x)
        return gradient_beta

    # in this case, use iterations; can also use rss movement criteria 
    def estimation_gradient_descent(self, x, y, lr = 0.1, iterations = 3000):
        all_ones = np.ones((x.shape[0],1))
        x_matrix = np.hstack((all_ones, x))

        beta = self.initialize_beta(x)
        yhat = np.matmul(x_matrix, beta)

        for i in range(iterations):
            gradient_beta = self.evaluate_gradient(x, y, yhat)
            beta = beta-lr*gradient_beta
            yhat = np.matmul(x_matrix, beta)

        return beta

    def estimation_linear_algebra(self, x, y):
        all_ones = np.ones((x.shape[0],1))
        x_matrix = np.hstack((all_ones, x))
        beta = np.matmul(np.linalg.inv(np.matmul(x_matrix.T,x_matrix)), np.matmul(x_matrix.T, y))

        return beta
    
lm = linear_regression()
beta_linear_algebra = lm.estimation_linear_algebra(x_train,y_train)
beta_gradient_descent = lm.estimation_gradient_descent(x_train,y_train)

# below class covers method 3 using numerical optimization solver (without providing the analytical gradient)
from scipy.optimize import minimize 

class linear_regression_mle:
    def __init__(self) -> None:
        pass

    def initialize_parameters(self, x, y):
        # beta: all zeros (inclusive of intercept beta0)
        # epsilon7
        betas = [0]*(1+x.shape[1])
        epsilon = [np.std(y)*0.5]
        return betas + epsilon

    def estimation_mle(self, x, y):
        all_ones = np.ones((x.shape[0],1))
        x_matrix = np.hstack((all_ones, x))
        params = self.initialize_parameters(x,y)
        
        def mle_norm(params):
            beta, eps = params[:-1], params[-1]
            yhat = np.matmul(x_matrix, beta)
            return 0.5*np.log(2*np.pi*eps**2)*len(x) + sum((0.5/(eps**2))*(y-yhat)**2)
        
        mle_model = minimize(mle_norm, params, method='L-BFGS-B')
        beta = mle_model.x
        
        yhat = np.matmul(x_matrix, beta)
        lss = sum((y-yhat)**2)

        return beta, lss

lm_mle = linear_regression_mle(x_train, y_train)
beta_mle, lss = lm_mle.estimation_mle(x_train, y_train)